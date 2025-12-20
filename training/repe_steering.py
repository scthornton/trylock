#!/usr/bin/env python3
"""
TRYLOCK Layer 2: RepE Steering Module

Runtime steering module that applies representation engineering
to dampen "attack compliance" during model inference.

This module provides:
1. SteeringWrapper - wraps a model to apply steering during forward pass
2. Dynamic alpha adjustment - tune sensitivity at runtime
3. Integration with PEFT adapters (works with Layer 1)

Based on:
- Representation Engineering (Zou et al., 2023)
- Activation Addition (Turner et al., 2023)
- Circuit Breakers (Zou et al., 2024)

Usage:
    # As a wrapper
    from repe_steering import SteeringWrapper

    model = AutoModelForCausalLM.from_pretrained(...)
    steered_model = SteeringWrapper(
        model,
        vectors_path="outputs/repe/steering_vectors.safetensors",
        alpha=1.0,  # Balanced mode
    )

    # Generate with steering
    outputs = steered_model.generate(**inputs)

    # Adjust sensitivity at runtime
    steered_model.set_alpha(2.5)  # Lockdown mode
"""

import json
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from safetensors.torch import load_file


@dataclass
class SteeringConfig:
    """Configuration for runtime steering."""

    # Alpha coefficient (0 = no steering, higher = more aggressive)
    alpha: float = 1.0

    # Which layers to steer (subset of available vectors)
    active_layers: Optional[list[int]] = None

    # Steering modes
    MODES = {
        "research": 0.0,     # No steering, maximum utility
        "balanced": 1.0,     # Default enterprise setting
        "elevated": 1.5,     # Heightened security
        "lockdown": 2.5,     # Maximum protection, blocks grey areas
    }


class SteeringHook:
    """
    Forward hook that applies steering vector subtraction to activations.

    During the forward pass, this hook intercepts the output of specific
    transformer layers and subtracts the steering vector (scaled by alpha)
    from the residual stream.

    This has the effect of dampening the "attack compliance" direction
    in the model's representation, making it less likely to comply with
    attacks without requiring explicit refusal behavior.
    """

    def __init__(
        self,
        steering_vector: Tensor,
        alpha: float = 1.0,
        layer_idx: int = 0,
    ):
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.layer_idx = layer_idx
        self._enabled = True

    def __call__(self, module, input, output):
        """Apply steering during forward pass."""
        if not self._enabled or self.alpha == 0:
            return output

        # Handle tuple output (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Apply steering: subtract alpha * steering_vector from all positions
        # steering_vector shape: [hidden_dim]
        # hidden_states shape: [batch, seq_len, hidden_dim]
        device = hidden_states.device
        steering = self.steering_vector.to(device)

        # Subtract steering direction (scaled by alpha)
        steered = hidden_states - self.alpha * steering.unsqueeze(0).unsqueeze(0)

        if rest is not None:
            return (steered,) + rest
        return steered

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def set_alpha(self, alpha: float):
        self.alpha = alpha


class SteeringWrapper(nn.Module):
    """
    Wrapper that applies representation engineering steering to a model.

    This wrapper:
    1. Loads pre-trained steering vectors
    2. Registers hooks on target layers
    3. Applies steering during forward pass
    4. Supports runtime alpha adjustment

    Usage:
        model = AutoModelForCausalLM.from_pretrained(...)
        steered = SteeringWrapper(model, vectors_path, alpha=1.0)
        outputs = steered.generate(**inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        vectors_path: str,
        config_path: Optional[str] = None,
        alpha: float = 1.0,
        active_layers: Optional[list[int]] = None,
    ):
        super().__init__()

        self.model = model
        self.alpha = alpha
        self.hooks: dict[int, tuple[SteeringHook, torch.utils.hooks.RemovableHandle]] = {}

        # Load steering vectors
        vectors_path = Path(vectors_path)
        self.steering_vectors = self._load_vectors(vectors_path)

        # Load config if available
        if config_path:
            config_path = Path(config_path)
        else:
            # Try to find config next to vectors
            config_path = vectors_path.parent / "repe_config.json"

        self.config = self._load_config(config_path) if config_path.exists() else None

        # Determine active layers
        available_layers = sorted(self.steering_vectors.keys())
        if active_layers:
            self.active_layers = [l for l in active_layers if l in available_layers]
        else:
            self.active_layers = available_layers

        # Register hooks
        self._register_hooks()

        print(f"SteeringWrapper initialized:")
        print(f"  Active layers: {self.active_layers}")
        print(f"  Alpha: {self.alpha}")

    def _load_vectors(self, path: Path) -> dict[int, Tensor]:
        """Load steering vectors from safetensors file."""
        data = load_file(str(path))
        vectors = {}
        for key, tensor in data.items():
            # Key format: "layer_12" -> 12
            layer_idx = int(key.split("_")[1])
            vectors[layer_idx] = tensor
        return vectors

    def _load_config(self, path: Path) -> dict:
        """Load config JSON."""
        with open(path) as f:
            return json.load(f)

    def _find_layers(self) -> list[nn.Module]:
        """Find transformer layers in model architecture."""
        model = self.model

        # Handle PEFT wrapped models
        if hasattr(model, 'base_model'):
            model = model.base_model
        if hasattr(model, 'model'):
            model = model.model

        # Common architectures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return list(model.model.layers)
        elif hasattr(model, 'layers'):
            return list(model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return list(model.transformer.h)
        else:
            raise ValueError("Cannot find transformer layers in model")

    def _register_hooks(self):
        """Register steering hooks on target layers."""
        layers = self._find_layers()

        for layer_idx in self.active_layers:
            if layer_idx >= len(layers):
                print(f"  Warning: Layer {layer_idx} not found (model has {len(layers)} layers)")
                continue

            if layer_idx not in self.steering_vectors:
                print(f"  Warning: No steering vector for layer {layer_idx}")
                continue

            hook = SteeringHook(
                steering_vector=self.steering_vectors[layer_idx],
                alpha=self.alpha,
                layer_idx=layer_idx,
            )
            handle = layers[layer_idx].register_forward_hook(hook)
            self.hooks[layer_idx] = (hook, handle)

    def set_alpha(self, alpha: float):
        """Set steering strength at runtime."""
        self.alpha = alpha
        for layer_idx, (hook, _) in self.hooks.items():
            hook.set_alpha(alpha)

    def set_mode(self, mode: str):
        """Set steering mode using predefined alpha values."""
        if mode not in SteeringConfig.MODES:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(SteeringConfig.MODES.keys())}")
        self.set_alpha(SteeringConfig.MODES[mode])

    def enable_steering(self):
        """Enable steering (after disable)."""
        for hook, _ in self.hooks.values():
            hook.enable()

    def disable_steering(self):
        """Temporarily disable steering."""
        for hook, _ in self.hooks.values():
            hook.disable()

    def remove_hooks(self):
        """Remove all hooks (permanent)."""
        for _, handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()

    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped model."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generate with steering applied."""
        return self.model.generate(*args, **kwargs)

    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def apply_steering_to_model(
    model: nn.Module,
    vectors_path: str,
    alpha: float = 1.0,
    active_layers: Optional[list[int]] = None,
) -> SteeringWrapper:
    """
    Convenience function to wrap a model with steering.

    Args:
        model: The model to wrap
        vectors_path: Path to steering_vectors.safetensors
        alpha: Steering strength (0=none, 1=balanced, 2.5=max)
        active_layers: Specific layers to steer (default: all available)

    Returns:
        SteeringWrapper instance
    """
    return SteeringWrapper(
        model=model,
        vectors_path=vectors_path,
        alpha=alpha,
        active_layers=active_layers,
    )


# =============================================================================
# Testing and Demo
# =============================================================================

def test_steering():
    """Test steering functionality with a small model."""
    import argparse

    parser = argparse.ArgumentParser(description="Test RepE Steering")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--vectors", type=str, required=True,
                        help="Path to steering_vectors.safetensors")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Test prompt (optional)")

    args = parser.parse_args()

    print("=" * 60)
    print("TRYLOCK REPE STEERING TEST")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Load model
    print("\n[1/3] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if args.adapter_path:
        print(f"[1b/3] Loading adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    # Apply steering
    print("\n[2/3] Applying steering wrapper...")
    steered_model = apply_steering_to_model(
        model,
        vectors_path=args.vectors,
        alpha=args.alpha,
    )

    # Test generation
    print("\n[3/3] Testing generation...")

    test_prompt = args.prompt or "What is the capital of France?"
    messages = [{"role": "user", "content": test_prompt}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\nPrompt: {test_prompt}")
    print("-" * 40)

    # Generate without steering
    print("\n[Without steering (alpha=0)]:")
    steered_model.set_alpha(0.0)
    with torch.no_grad():
        outputs = steered_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(response)

    # Generate with steering
    print(f"\n[With steering (alpha={args.alpha})]:")
    steered_model.set_alpha(args.alpha)
    with torch.no_grad():
        outputs = steered_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(response)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_steering()
