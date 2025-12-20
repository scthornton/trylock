#!/usr/bin/env python3
"""
TRYLOCK Gateway: Three-Layer Defense Integration

Integrates all three TRYLOCK defense layers:
- Layer 1: DPO-trained model (KNOWLEDGE)
- Layer 2: RepE steering (INSTINCT)
- Layer 3: Sidecar classifier (OVERSIGHT)

The gateway provides dynamic alpha adjustment based on sidecar threat detection,
enabling adaptive security posture without model reloading.

Usage:
    python deployment/trylock_gateway.py \
        --model_name mistralai/Mistral-7B-Instruct-v0.3 \
        --adapter_path outputs/trylock-mistral-7b \
        --vectors_path outputs/repe \
        --sidecar_path outputs/trylock-sidecar \
        --port 8080

    # API:
    curl -X POST http://localhost:8080/generate \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "Hello"}]}'
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from concurrent.futures import ThreadPoolExecutor
import logging

import yaml

try:
    import torch
    from torch import Tensor
except ImportError:
    torch = None
    Tensor = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    FastAPI = None
    BaseModel = object
    uvicorn = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TRYLOCKConfig:
    """Configuration for TRYLOCK Gateway."""

    # Layer 1: DPO Model
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    adapter_path: str | None = None
    trust_remote_code: bool = True

    # Layer 2: RepE Steering
    vectors_path: str | None = None
    steering_layers: list[int] = field(
        default_factory=lambda: [12, 14, 16, 18, 20, 22, 24, 26]
    )

    # Layer 3: Sidecar Classifier
    sidecar_path: str | None = None
    sidecar_enabled: bool = True

    # Dynamic Alpha Configuration
    alpha_safe: float = 0.5      # Alpha when SAFE
    alpha_warn: float = 1.5      # Alpha when WARN
    alpha_attack: float = 2.5    # Alpha when ATTACK
    alpha_default: float = 1.0   # Default when sidecar disabled

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    do_sample: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Hardware
    device: str = "auto"
    bf16: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TRYLOCKConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("trylock", data))


# ============================================================================
# Request/Response Models
# ============================================================================

class Message(BaseModel if BaseModel != object else object):
    role: str = "user"
    content: str = ""


class GenerateRequest(BaseModel if BaseModel != object else object):
    """Generation request."""
    messages: list[dict] = Field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = None
    alpha_override: float | None = None  # Override dynamic alpha
    security_mode: Literal["auto", "research", "balanced", "elevated", "lockdown"] = "auto"


class GenerateResponse(BaseModel if BaseModel != object else object):
    """Generation response."""
    response: str
    sidecar_classification: str | None = None
    applied_alpha: float
    security_mode: str
    latency_ms: float


class HealthResponse(BaseModel if BaseModel != object else object):
    """Health check response."""
    status: str
    layers_loaded: dict[str, bool]
    device: str


# ============================================================================
# Sidecar Client
# ============================================================================

class SidecarClient:
    """Client for sidecar classifier."""

    def __init__(self, model_path: str, device: str = "auto"):
        if torch is None:
            raise ImportError("torch is required")

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = Path(model_path)

        # Check if this is a PEFT adapter
        adapter_config_path = model_path / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)

            base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")

            # Load tokenizer from adapter path (has all tokens)
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model for sequence classification
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=3,
                device_map="auto" if device == "auto" else device,
                torch_dtype=torch.bfloat16,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Load PEFT adapter
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            # Load as full model (non-PEFT)
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                device_map="auto" if device == "auto" else device,
                torch_dtype=torch.bfloat16,
            )

        self.model.eval()

        # Load thresholds
        thresholds_path = model_path / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                thresholds = json.load(f)
            self.label_names = thresholds.get("label_names", ["SAFE", "WARN", "ATTACK"])
        else:
            self.label_names = ["SAFE", "WARN", "ATTACK"]

    def classify(self, messages: list[dict]) -> dict:
        """Classify conversation."""
        text = self._format_conversation(messages)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        predicted_class = int(probs.argmax())
        risk_score = probs[1] * 0.3 + probs[2] * 0.7

        return {
            "classification": self.label_names[predicted_class],
            "class_id": predicted_class,
            "probabilities": {
                name: float(p) for name, p in zip(self.label_names, probs)
            },
            "risk_score": float(risk_score),
        }

    def _format_conversation(self, messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<{role}>\n{content}\n</{role}>")
        return "\n".join(parts)


# ============================================================================
# RepE Steering Hook
# ============================================================================

class RepESteeringHook:
    """
    Hook for Representation Engineering steering.

    Subtracts steering vectors from activations at specified layers
    to suppress attack compliance behavior.
    """

    def __init__(
        self,
        steering_vectors: dict[int, Tensor],
        target_layers: list[int],
        alpha: float = 1.0,
    ):
        self.steering_vectors = steering_vectors
        self.target_layers = set(target_layers)
        self.alpha = alpha
        self.hooks = []
        self.enabled = True

    def set_alpha(self, alpha: float):
        """Update the alpha coefficient."""
        self.alpha = alpha

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if not self.enabled or self.alpha == 0.0:
                return output

            # Get steering vector for this layer
            if layer_idx not in self.steering_vectors:
                return output

            vector = self.steering_vectors[layer_idx]

            # Extract hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # Move vector to correct device
            vector = vector.to(hidden_states.device).to(hidden_states.dtype)

            # Apply steering: subtract alpha * direction from all positions
            steered = hidden_states - self.alpha * vector.unsqueeze(0).unsqueeze(0)

            if rest is not None:
                return (steered,) + rest
            return steered

        return hook

    def register(self, model):
        """Register hooks on target layers."""
        self.clear()

        # Unwrap PEFT model if needed
        unwrapped = model
        if hasattr(model, 'base_model'):
            unwrapped = model.base_model
        if hasattr(unwrapped, 'model'):
            unwrapped = unwrapped.model

        # Find layers
        layers = None
        if hasattr(unwrapped, 'model') and hasattr(unwrapped.model, 'layers'):
            layers = unwrapped.model.layers
        elif hasattr(unwrapped, 'layers'):
            layers = unwrapped.layers
        elif hasattr(unwrapped, 'transformer') and hasattr(unwrapped.transformer, 'h'):
            layers = unwrapped.transformer.h

        if layers is None:
            raise ValueError("Cannot find transformer layers in model")

        for idx in self.target_layers:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(self._make_hook(idx))
                self.hooks.append(hook)

    def clear(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# TRYLOCK Gateway
# ============================================================================

class TRYLOCKGateway:
    """
    Main TRYLOCK Gateway integrating all three defense layers.
    """

    def __init__(self, config: TRYLOCKConfig):
        self.config = config
        self.logger = logging.getLogger("trylock.gateway")

        # Components
        self.model = None
        self.tokenizer = None
        self.sidecar = None
        self.steering_hook = None

        # Alpha modes
        self.alpha_modes = {
            "research": 0.0,
            "balanced": 1.0,
            "elevated": 1.5,
            "lockdown": 2.5,
        }

    def load(self):
        """Load all TRYLOCK components."""
        if torch is None:
            raise ImportError("torch is required")
        if AutoModelForCausalLM is None:
            raise ImportError("transformers is required")

        # Determine device
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device

        self.logger.info(f"Loading TRYLOCK Gateway on {device}")

        # Layer 1: Load base model with DPO adapter
        self.logger.info("[Layer 1] Loading DPO-trained model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
        )

        if self.config.adapter_path and PeftModel is not None:
            self.logger.info(f"   Loading adapter from {self.config.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.config.adapter_path)

        self.model.eval()
        self.logger.info("   Layer 1 loaded")

        # Layer 2: Load RepE steering vectors
        if self.config.vectors_path and load_file is not None:
            self.logger.info("[Layer 2] Loading RepE steering vectors...")
            vectors_path = Path(self.config.vectors_path)
            vectors_file = vectors_path / "steering_vectors.safetensors"

            if vectors_file.exists():
                vectors_data = load_file(str(vectors_file))

                # Parse steering vectors by layer
                steering_vectors = {}
                for key, tensor in vectors_data.items():
                    if key.startswith("layer_"):
                        layer_idx = int(key.split("_")[1])
                        steering_vectors[layer_idx] = tensor

                self.steering_hook = RepESteeringHook(
                    steering_vectors,
                    self.config.steering_layers,
                    alpha=self.config.alpha_default,
                )
                self.steering_hook.register(self.model)
                self.logger.info(f"   Loaded {len(steering_vectors)} steering vectors")
            else:
                self.logger.warning(f"   Steering vectors not found at {vectors_file}")
        else:
            self.logger.info("[Layer 2] RepE steering disabled")

        # Layer 3: Load sidecar classifier
        if self.config.sidecar_path and self.config.sidecar_enabled:
            self.logger.info("[Layer 3] Loading sidecar classifier...")
            try:
                self.sidecar = SidecarClient(
                    self.config.sidecar_path,
                    device=device,
                )
                self.logger.info("   Sidecar loaded")
            except Exception as e:
                self.logger.warning(f"   Failed to load sidecar: {e}")
                self.sidecar = None
        else:
            self.logger.info("[Layer 3] Sidecar disabled")

        self.logger.info("TRYLOCK Gateway loaded successfully")

    def _get_alpha_for_classification(self, classification: str) -> float:
        """Map sidecar classification to alpha value."""
        if classification == "SAFE":
            return self.config.alpha_safe
        elif classification == "WARN":
            return self.config.alpha_warn
        else:  # ATTACK
            return self.config.alpha_attack

    def generate(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        alpha_override: float | None = None,
        security_mode: str = "auto",
    ) -> dict:
        """
        Generate response with full TRYLOCK protection.

        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            alpha_override: Force specific alpha value
            security_mode: One of "auto", "research", "balanced", "elevated", "lockdown"

        Returns:
            Generation result with metadata
        """
        start_time = time.time()

        # Determine alpha value
        sidecar_result = None
        if alpha_override is not None:
            alpha = alpha_override
            mode = "override"
        elif security_mode != "auto":
            alpha = self.alpha_modes.get(security_mode, self.config.alpha_default)
            mode = security_mode
        elif self.sidecar is not None:
            # Dynamic alpha from sidecar
            sidecar_result = self.sidecar.classify(messages)
            alpha = self._get_alpha_for_classification(sidecar_result["classification"])
            mode = f"auto-{sidecar_result['classification'].lower()}"
        else:
            alpha = self.config.alpha_default
            mode = "default"

        # Apply alpha to steering hook
        if self.steering_hook is not None:
            self.steering_hook.set_alpha(alpha)

        # Format prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.config.max_new_tokens,
                temperature=temperature or self.config.temperature,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        latency_ms = (time.time() - start_time) * 1000

        return {
            "response": response,
            "sidecar_classification": sidecar_result["classification"] if sidecar_result else None,
            "applied_alpha": alpha,
            "security_mode": mode,
            "latency_ms": latency_ms,
        }

    def get_health(self) -> dict:
        """Get gateway health status."""
        return {
            "status": "healthy" if self.model is not None else "unhealthy",
            "layers_loaded": {
                "layer1_dpo": self.model is not None,
                "layer2_repe": self.steering_hook is not None,
                "layer3_sidecar": self.sidecar is not None,
            },
            "device": str(self.model.device) if self.model else "none",
        }

    def create_app(self) -> "FastAPI":
        """Create FastAPI application."""
        if FastAPI is None:
            raise ImportError("fastapi required")

        app = FastAPI(
            title="TRYLOCK Gateway",
            description="Three-Layer AI Security Defense Stack",
            version="2.0.0",
        )

        gateway = self

        @app.on_event("startup")
        async def startup():
            gateway.load()

        @app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            try:
                result = gateway.generate(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    alpha_override=request.alpha_override,
                    security_mode=request.security_mode,
                )
                return GenerateResponse(**result)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health", response_model=HealthResponse)
        async def health():
            return HealthResponse(**gateway.get_health())

        @app.get("/")
        async def root():
            return {
                "service": "TRYLOCK Gateway",
                "version": "2.0.0",
                "layers": {
                    "layer1": "DPO (KNOWLEDGE)",
                    "layer2": "RepE (INSTINCT)",
                    "layer3": "Sidecar (OVERSIGHT)",
                },
                "security_modes": list(gateway.alpha_modes.keys()) + ["auto"],
            }

        return app

    def run(self):
        """Run the gateway server."""
        if uvicorn is None:
            raise ImportError("uvicorn required")

        app = self.create_app()
        uvicorn.run(
            app,
            host=self.config.host,
            port=self.config.port,
        )


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="TRYLOCK Gateway")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--vectors_path", type=str, default=None)
    parser.add_argument("--sidecar_path", type=str, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()

    if args.config:
        config = TRYLOCKConfig.from_yaml(args.config)
    else:
        config = TRYLOCKConfig(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            vectors_path=args.vectors_path,
            sidecar_path=args.sidecar_path,
            port=args.port,
            host=args.host,
        )

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("TRYLOCK GATEWAY - Three-Layer Defense Stack")
    print("=" * 60)
    print(f"Layer 1 (DPO):    {config.model_name}")
    print(f"   Adapter:       {config.adapter_path or 'None'}")
    print(f"Layer 2 (RepE):   {config.vectors_path or 'Disabled'}")
    print(f"Layer 3 (Sidecar): {config.sidecar_path or 'Disabled'}")
    print(f"Server:           http://{config.host}:{config.port}")
    print("=" * 60)

    gateway = TRYLOCKGateway(config)
    gateway.run()


if __name__ == "__main__":
    main()
