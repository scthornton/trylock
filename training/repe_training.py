"""
TRYLOCK RepE (Representation Engineering) Training

Circuit breaker training using representation engineering to identify and
dampen "attack compliance" directions in the model's latent space.

Key concepts:
1. Capture activations at pivot points (attack vs safe)
2. Compute control direction via PCA on activation differences
3. Apply control vectors at inference with tunable alpha
   - α=0.0: Research mode (no steering)
   - α=1.0: Balanced mode (standard defense)
   - α=2.5: Lockdown mode (maximum defense)

Usage:
    python -m training.repe_training --config configs/llama3_8b_repe.yaml

    # Or programmatically:
    from training.repe_training import train_repe_vectors
    vectors = train_repe_vectors(config)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
except ImportError:
    PCA = None
    LogisticRegression = None


@dataclass
class RepEConfig:
    """Configuration for RepE training."""

    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    dpo_checkpoint: str | None = None  # Load from DPO if available
    trust_remote_code: bool = True

    # Layer selection
    target_layers: list[int] = field(
        default_factory=lambda: [10, 14, 18, 22]  # Middle-to-late layers
    )
    layer_selection_method: Literal["fixed", "auto"] = "fixed"
    auto_select_top_k: int = 4

    # Control vector computation
    method: Literal["pca", "mean_diff", "logistic", "caa"] = "pca"
    n_components: int = 1  # For PCA
    normalize_vectors: bool = True

    # Data
    activations_file: str = "data/activations/pivot_activations.npz"
    train_file: str | None = "data/tier1_open/attacks/train.jsonl"
    max_samples: int | None = None

    # Alpha presets
    alpha_presets: dict[str, float] = field(
        default_factory=lambda: {
            "research": 0.0,
            "balanced": 1.0,
            "enhanced": 1.5,
            "lockdown": 2.5,
        }
    )

    # Output
    output_dir: str = "./outputs/trylock-repe"

    # Hardware
    bf16: bool = True
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RepEConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("repe", data))


class ActivationExtractor:
    """
    Extract activations from specific layers during forward pass.

    Used to collect activation data for computing control vectors.
    """

    def __init__(
        self,
        model,
        tokenizer,
        target_layers: list[int],
        device: str = "auto",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = target_layers

        if device == "auto":
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        self.activations = {}
        self._hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        """Register forward hooks on target layers."""
        for layer_idx in self.target_layers:
            # Handle different model architectures
            if hasattr(self.model, "model"):
                # Llama-style
                if hasattr(self.model.model, "layers"):
                    layer = self.model.model.layers[layer_idx]
                else:
                    continue
            elif hasattr(self.model, "transformer"):
                # GPT-style
                if hasattr(self.model.transformer, "h"):
                    layer = self.model.transformer.h[layer_idx]
                else:
                    continue
            else:
                continue

            def make_hook(idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    self.activations[idx] = hidden.detach()
                return hook

            handle = layer.register_forward_hook(make_hook(layer_idx))
            self._hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def extract(self, text: str, token_position: int = -1) -> dict[int, torch.Tensor]:
        """
        Extract activations for a given text.

        Args:
            text: Input text
            token_position: Which token position to extract (-1 for last)

        Returns:
            Dictionary mapping layer index to activation tensor
        """
        self.activations = {}

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        with torch.no_grad():
            self.model(**inputs)

        # Extract activations at specified token position
        result = {}
        for layer_idx, activation in self.activations.items():
            # activation shape: (batch, seq_len, hidden_dim)
            if token_position == -1:
                # Last non-padding token
                seq_len = inputs.attention_mask.sum().item()
                result[layer_idx] = activation[0, seq_len - 1, :].cpu()
            else:
                result[layer_idx] = activation[0, token_position, :].cpu()

        return result

    def extract_pair(
        self,
        prompt: str,
        safe_response: str,
        unsafe_response: str,
    ) -> tuple[dict, dict]:
        """
        Extract activations for a safe/unsafe response pair.

        Returns:
            Tuple of (safe_activations, unsafe_activations)
        """
        safe_text = prompt + safe_response
        unsafe_text = prompt + unsafe_response

        safe_acts = self.extract(safe_text)
        unsafe_acts = self.extract(unsafe_text)

        return safe_acts, unsafe_acts


class ControlVector:
    """
    Represents a control vector for steering model behavior.

    The vector can be added to activations at inference time
    with a configurable alpha coefficient.
    """

    def __init__(
        self,
        layer_idx: int,
        vector: np.ndarray | torch.Tensor,
        method: str = "pca",
        metadata: dict | None = None,
    ):
        self.layer_idx = layer_idx
        if isinstance(vector, torch.Tensor):
            self.vector = vector.numpy()
        else:
            self.vector = vector
        self.method = method
        self.metadata = metadata or {}

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert to PyTorch tensor."""
        return torch.tensor(self.vector, device=device, dtype=torch.float32)

    def apply(
        self,
        activations: torch.Tensor,
        alpha: float = 1.0,
        direction: str = "subtract",
    ) -> torch.Tensor:
        """
        Apply control vector to activations.

        Args:
            activations: Input activations (batch, seq_len, hidden_dim)
            alpha: Steering strength coefficient
            direction: "subtract" to reduce unsafe, "add" to increase

        Returns:
            Modified activations
        """
        vector = self.to_tensor(activations.device)

        if direction == "subtract":
            return activations - alpha * vector
        else:
            return activations + alpha * vector

    def save(self, path: str | Path):
        """Save control vector to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "layer_idx": self.layer_idx,
            "vector": self.vector.tolist(),
            "method": self.method,
            "metadata": self.metadata,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ControlVector":
        """Load control vector from file."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            layer_idx=data["layer_idx"],
            vector=np.array(data["vector"]),
            method=data["method"],
            metadata=data.get("metadata", {}),
        )


class ControlVectorSet:
    """Collection of control vectors for multiple layers."""

    def __init__(self, vectors: list[ControlVector] | None = None):
        self.vectors = {v.layer_idx: v for v in (vectors or [])}

    def add(self, vector: ControlVector):
        """Add a control vector."""
        self.vectors[vector.layer_idx] = vector

    def get(self, layer_idx: int) -> ControlVector | None:
        """Get control vector for a layer."""
        return self.vectors.get(layer_idx)

    def save(self, path: str | Path):
        """Save all control vectors."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for layer_idx, vector in self.vectors.items():
            vector.save(path / f"layer_{layer_idx}.json")

        # Save metadata
        metadata = {
            "layers": list(self.vectors.keys()),
            "methods": {k: v.method for k, v in self.vectors.items()},
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ControlVectorSet":
        """Load control vectors from directory."""
        path = Path(path)

        vectors = []
        for vector_file in path.glob("layer_*.json"):
            vectors.append(ControlVector.load(vector_file))

        return cls(vectors)


def compute_control_vectors_pca(
    safe_activations: list[dict],
    unsafe_activations: list[dict],
    target_layers: list[int],
    n_components: int = 1,
    normalize: bool = True,
) -> ControlVectorSet:
    """
    Compute control vectors using PCA on activation differences.

    The first principal component of (unsafe - safe) differences
    represents the "attack compliance" direction.
    """
    if np is None or PCA is None:
        raise ImportError("numpy and sklearn are required for PCA")

    vectors = ControlVectorSet()

    for layer_idx in target_layers:
        # Collect activation differences
        diffs = []
        for safe, unsafe in zip(safe_activations, unsafe_activations):
            if layer_idx in safe and layer_idx in unsafe:
                diff = unsafe[layer_idx].numpy() - safe[layer_idx].numpy()
                diffs.append(diff)

        if len(diffs) < 2:
            print(f"Warning: Not enough samples for layer {layer_idx}")
            continue

        diffs = np.stack(diffs)

        # PCA to find main direction
        pca = PCA(n_components=n_components)
        pca.fit(diffs)

        # First component is the control direction
        control_direction = pca.components_[0]

        if normalize:
            control_direction = control_direction / np.linalg.norm(control_direction)

        vector = ControlVector(
            layer_idx=layer_idx,
            vector=control_direction,
            method="pca",
            metadata={
                "explained_variance_ratio": pca.explained_variance_ratio_[0],
                "n_samples": len(diffs),
            },
        )
        vectors.add(vector)

    return vectors


def compute_control_vectors_mean_diff(
    safe_activations: list[dict],
    unsafe_activations: list[dict],
    target_layers: list[int],
    normalize: bool = True,
) -> ControlVectorSet:
    """
    Compute control vectors as mean difference between unsafe and safe.

    Simple but effective baseline method.
    """
    if np is None:
        raise ImportError("numpy is required")

    vectors = ControlVectorSet()

    for layer_idx in target_layers:
        safe_acts = []
        unsafe_acts = []

        for safe, unsafe in zip(safe_activations, unsafe_activations):
            if layer_idx in safe and layer_idx in unsafe:
                safe_acts.append(safe[layer_idx].numpy())
                unsafe_acts.append(unsafe[layer_idx].numpy())

        if len(safe_acts) < 2:
            print(f"Warning: Not enough samples for layer {layer_idx}")
            continue

        safe_mean = np.mean(safe_acts, axis=0)
        unsafe_mean = np.mean(unsafe_acts, axis=0)

        control_direction = unsafe_mean - safe_mean

        if normalize:
            control_direction = control_direction / np.linalg.norm(control_direction)

        vector = ControlVector(
            layer_idx=layer_idx,
            vector=control_direction,
            method="mean_diff",
            metadata={"n_samples": len(safe_acts)},
        )
        vectors.add(vector)

    return vectors


def compute_control_vectors_logistic(
    safe_activations: list[dict],
    unsafe_activations: list[dict],
    target_layers: list[int],
    normalize: bool = True,
) -> ControlVectorSet:
    """
    Compute control vectors using logistic regression weights.

    Train a classifier to distinguish safe from unsafe, then use
    the weight vector as the control direction.
    """
    if np is None or LogisticRegression is None:
        raise ImportError("numpy and sklearn are required")

    vectors = ControlVectorSet()

    for layer_idx in target_layers:
        X = []
        y = []

        for safe, unsafe in zip(safe_activations, unsafe_activations):
            if layer_idx in safe and layer_idx in unsafe:
                X.append(safe[layer_idx].numpy())
                y.append(0)  # safe
                X.append(unsafe[layer_idx].numpy())
                y.append(1)  # unsafe

        if len(X) < 4:
            print(f"Warning: Not enough samples for layer {layer_idx}")
            continue

        X = np.stack(X)
        y = np.array(y)

        # Train classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X, y)

        # Weight vector points toward unsafe class
        control_direction = clf.coef_[0]

        if normalize:
            control_direction = control_direction / np.linalg.norm(control_direction)

        vector = ControlVector(
            layer_idx=layer_idx,
            vector=control_direction,
            method="logistic",
            metadata={
                "accuracy": clf.score(X, y),
                "n_samples": len(X),
            },
        )
        vectors.add(vector)

    return vectors


def auto_select_layers(
    model,
    tokenizer,
    sample_prompts: list[dict],
    top_k: int = 4,
    device: str = "auto",
) -> list[int]:
    """
    Automatically select the best layers for control vectors.

    Evaluates layer activations on sample prompts and selects
    layers with highest discriminative power.
    """
    if AutoModelForCausalLM is None:
        raise ImportError("transformers is required")

    # Get total number of layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n_layers = len(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        n_layers = len(model.transformer.h)
    else:
        # Default to checking middle-to-late layers
        return [10, 14, 18, 22]

    # Test all layers
    all_layers = list(range(n_layers))
    extractor = ActivationExtractor(model, tokenizer, all_layers, device)

    layer_scores = {i: 0.0 for i in all_layers}

    try:
        for sample in sample_prompts:
            prompt = sample.get("prompt", "")
            safe = sample.get("safe_response", "")
            unsafe = sample.get("unsafe_response", "")

            if not all([prompt, safe, unsafe]):
                continue

            safe_acts, unsafe_acts = extractor.extract_pair(prompt, safe, unsafe)

            for layer_idx in all_layers:
                if layer_idx in safe_acts and layer_idx in unsafe_acts:
                    # Score by activation difference magnitude
                    diff = (
                        unsafe_acts[layer_idx].numpy() - safe_acts[layer_idx].numpy()
                    )
                    layer_scores[layer_idx] += np.linalg.norm(diff)
    finally:
        extractor.remove_hooks()

    # Select top-k layers
    sorted_layers = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    selected = [layer for layer, score in sorted_layers[:top_k]]

    print(f"Auto-selected layers: {selected}")
    return sorted(selected)


def collect_activation_pairs(
    model,
    tokenizer,
    data_path: str | Path,
    target_layers: list[int],
    max_samples: int | None = None,
    device: str = "auto",
) -> tuple[list[dict], list[dict]]:
    """
    Collect activation pairs from trajectory data.

    Returns:
        Tuple of (safe_activations, unsafe_activations) lists
    """
    extractor = ActivationExtractor(model, tokenizer, target_layers, device)

    safe_activations = []
    unsafe_activations = []

    try:
        with open(data_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                if not line.strip():
                    continue

                traj = json.loads(line)
                conversation = traj.get("conversation", [])
                responses = traj.get("responses", {})

                chosen = responses.get("chosen", {})
                rejected_unsafe = responses.get("rejected_unsafe", {})

                if not chosen.get("content") or not rejected_unsafe.get("content"):
                    continue

                # Build prompt
                messages = []
                for turn in conversation:
                    role = turn.get("role")
                    content = turn.get("content")
                    if role in ["user", "assistant", "system"]:
                        messages.append({"role": role, "content": content})

                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Extract activations
                safe_acts, unsafe_acts = extractor.extract_pair(
                    prompt,
                    chosen["content"],
                    rejected_unsafe["content"],
                )

                safe_activations.append(safe_acts)
                unsafe_activations.append(unsafe_acts)

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1} trajectories...")

    finally:
        extractor.remove_hooks()

    print(f"Collected {len(safe_activations)} activation pairs")
    return safe_activations, unsafe_activations


def train_repe_vectors(config: RepEConfig) -> ControlVectorSet:
    """
    Train RepE control vectors.

    Args:
        config: Training configuration

    Returns:
        ControlVectorSet with trained vectors
    """
    if torch is None:
        raise ImportError("torch is required")
    if AutoModelForCausalLM is None:
        raise ImportError("transformers is required")

    print("TRYLOCK RepE Training")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Target layers: {config.target_layers}")
    print(f"Method: {config.method}")
    print(f"Output: {config.output_dir}")
    print()

    # Load model
    model_path = config.dpo_checkpoint or config.model_name
    print(f"Loading model from {model_path}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-select layers if configured
    target_layers = config.target_layers
    if config.layer_selection_method == "auto" and config.train_file:
        # Load sample prompts for layer selection
        sample_prompts = []
        with open(config.train_file) as f:
            for i, line in enumerate(f):
                if i >= 20:  # Use 20 samples for selection
                    break
                if line.strip():
                    traj = json.loads(line)
                    conv = traj.get("conversation", [])
                    responses = traj.get("responses", {})

                    messages = [
                        {"role": t["role"], "content": t["content"]}
                        for t in conv
                        if t.get("role") in ["user", "assistant", "system"]
                    ]

                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    sample_prompts.append({
                        "prompt": prompt,
                        "safe_response": responses.get("chosen", {}).get("content", ""),
                        "unsafe_response": responses.get("rejected_unsafe", {}).get(
                            "content", ""
                        ),
                    })

        target_layers = auto_select_layers(
            model,
            tokenizer,
            sample_prompts,
            top_k=config.auto_select_top_k,
            device=config.device,
        )

    # Check for pre-computed activations
    activations_path = Path(config.activations_file)
    if activations_path.exists() and activations_path.suffix == ".npz":
        print(f"Loading pre-computed activations from {config.activations_file}...")
        data = np.load(config.activations_file, allow_pickle=True)
        safe_activations = data["safe"].tolist()
        unsafe_activations = data["unsafe"].tolist()
    elif config.train_file:
        print(f"Collecting activations from {config.train_file}...")
        safe_activations, unsafe_activations = collect_activation_pairs(
            model,
            tokenizer,
            config.train_file,
            target_layers,
            max_samples=config.max_samples,
            device=config.device,
        )

        # Save activations for reuse
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path / "activations.npz",
            safe=safe_activations,
            unsafe=unsafe_activations,
        )
    else:
        raise ValueError("Either activations_file or train_file must be provided")

    # Compute control vectors
    print(f"\nComputing control vectors using {config.method} method...")

    if config.method == "pca":
        vectors = compute_control_vectors_pca(
            safe_activations,
            unsafe_activations,
            target_layers,
            n_components=config.n_components,
            normalize=config.normalize_vectors,
        )
    elif config.method == "mean_diff":
        vectors = compute_control_vectors_mean_diff(
            safe_activations,
            unsafe_activations,
            target_layers,
            normalize=config.normalize_vectors,
        )
    elif config.method == "logistic":
        vectors = compute_control_vectors_logistic(
            safe_activations,
            unsafe_activations,
            target_layers,
            normalize=config.normalize_vectors,
        )
    else:
        raise ValueError(f"Unknown method: {config.method}")

    # Save vectors
    output_dir = Path(config.output_dir)
    vectors.save(output_dir / "control_vectors")

    # Save alpha presets
    with open(output_dir / "alpha_presets.json", "w") as f:
        json.dump(config.alpha_presets, f, indent=2)

    print(f"\nSaved control vectors to {config.output_dir}")
    print(f"Vectors computed for layers: {list(vectors.vectors.keys())}")

    # Print metadata
    for layer_idx, vector in vectors.vectors.items():
        print(f"  Layer {layer_idx}: {vector.metadata}")

    return vectors


class RepEInferenceWrapper:
    """
    Wrapper for applying RepE control vectors during inference.

    Usage:
        wrapper = RepEInferenceWrapper(model, vectors, alpha=1.0)
        wrapper.enable()
        # Run inference
        wrapper.disable()
    """

    def __init__(
        self,
        model,
        control_vectors: ControlVectorSet,
        alpha: float = 1.0,
    ):
        self.model = model
        self.control_vectors = control_vectors
        self.alpha = alpha
        self._hooks = []
        self._enabled = False

    def set_alpha(self, alpha: float):
        """Update steering strength."""
        self.alpha = alpha

    def set_alpha_preset(self, preset: str, presets_path: str | None = None):
        """Set alpha from named preset."""
        presets = {
            "research": 0.0,
            "balanced": 1.0,
            "enhanced": 1.5,
            "lockdown": 2.5,
        }

        if presets_path:
            with open(presets_path) as f:
                presets.update(json.load(f))

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

        self.alpha = presets[preset]
        print(f"Set alpha to {self.alpha} ({preset} mode)")

    def enable(self):
        """Enable control vector application."""
        if self._enabled:
            return

        for layer_idx, vector in self.control_vectors.vectors.items():
            # Get layer module
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                layer = self.model.model.layers[layer_idx]
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                layer = self.model.transformer.h[layer_idx]
            else:
                continue

            def make_hook(cv):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        modified = cv.apply(hidden, self.alpha, direction="subtract")
                        return (modified,) + output[1:]
                    else:
                        return cv.apply(output, self.alpha, direction="subtract")
                return hook

            handle = layer.register_forward_hook(make_hook(vector))
            self._hooks.append(handle)

        self._enabled = True
        print(f"RepE enabled with alpha={self.alpha}")

    def disable(self):
        """Disable control vector application."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._enabled = False
        print("RepE disabled")

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRYLOCK RepE Training")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--dpo-checkpoint",
        type=str,
        default=None,
        help="Path to DPO checkpoint",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/tier1_open/attacks/train.jsonl",
        help="Training data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/trylock-repe",
        help="Output directory",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["pca", "mean_diff", "logistic"],
        default="pca",
        help="Control vector computation method",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="10,14,18,22",
        help="Comma-separated list of target layers",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum training samples",
    )

    args = parser.parse_args()

    if args.config:
        config = RepEConfig.from_yaml(args.config)
    else:
        target_layers = [int(x) for x in args.layers.split(",")]
        config = RepEConfig(
            model_name=args.model,
            dpo_checkpoint=args.dpo_checkpoint,
            train_file=args.train_file,
            output_dir=args.output_dir,
            method=args.method,
            target_layers=target_layers,
            max_samples=args.max_samples,
        )

    train_repe_vectors(config)
