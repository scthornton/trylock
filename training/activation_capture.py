#!/usr/bin/env python3
"""
TRYLOCK Layer 2: Activation Capture Pipeline

Captures model activations at pivot points during attack trajectories.
These activations are used to train linear probes that identify the
"attack compliance" direction in latent space.

Based on Representation Engineering (Zou et al., 2023) and Circuit Breakers (2024).

Usage:
    python training/activation_capture.py \
        --model_name mistralai/Mistral-7B-Instruct-v0.3 \
        --adapter_path outputs/trylock-mistral-7b \
        --data_path data/dpo \
        --output_dir activations \
        --target_layers 12,14,16,18,20
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import warnings

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import save_file, load_file
from tqdm import tqdm


@dataclass
class ActivationCapture:
    """Stores captured activations for a single sample."""
    sample_id: str
    attack_family: str
    attack_type: str
    difficulty: str
    is_attack: bool

    # Activations at each target layer (shape: [seq_len, hidden_dim])
    layer_activations: dict[int, Tensor]

    # Position of the last token before response (pivot point)
    pivot_position: int

    # The actual conversation that was processed
    conversation_length: int


class ActivationCaptureHook:
    """
    Hook to capture activations at specified layers during forward pass.

    Uses the residual stream (output of each transformer block) as the
    activation to capture, following RepE methodology.
    """

    def __init__(self, target_layers: list[int]):
        self.target_layers = set(target_layers)
        self.captured: dict[int, Tensor] = {}
        self.hooks = []

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # Capture the hidden states (residual stream)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Store on CPU to save GPU memory
            self.captured[layer_idx] = hidden_states.detach().cpu()
        return hook

    def register(self, model):
        """Register hooks on target layers."""
        self.clear()

        # Find the transformer layers - handle various model architectures including PEFT
        layers = None

        # Try to unwrap PEFT model first
        unwrapped = model
        if hasattr(model, 'base_model'):
            unwrapped = model.base_model
        if hasattr(unwrapped, 'model'):
            unwrapped = unwrapped.model

        # Now find layers in the unwrapped model
        if hasattr(unwrapped, 'model') and hasattr(unwrapped.model, 'layers'):
            # Mistral/Llama structure: model.model.layers
            layers = unwrapped.model.layers
        elif hasattr(unwrapped, 'layers'):
            # Direct layers access
            layers = unwrapped.layers
        elif hasattr(unwrapped, 'transformer') and hasattr(unwrapped.transformer, 'h'):
            # GPT-2/GPT-J structure
            layers = unwrapped.transformer.h

        if layers is None:
            # Debug: print model structure
            print(f"DEBUG: Model type: {type(model)}")
            print(f"DEBUG: Unwrapped type: {type(unwrapped)}")
            print(f"DEBUG: Model attributes: {[a for a in dir(unwrapped) if not a.startswith('_')][:20]}")
            raise ValueError("Cannot find transformer layers in model architecture")

        for idx in self.target_layers:
            if idx < len(layers):
                hook = layers[idx].register_forward_hook(self._make_hook(idx))
                self.hooks.append(hook)
            else:
                warnings.warn(f"Layer {idx} does not exist (model has {len(layers)} layers)")

    def clear(self):
        """Remove all hooks and clear captured activations."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.captured = {}

    def get_pivot_activations(self, pivot_position: int) -> dict[int, Tensor]:
        """
        Extract activations at the pivot position from all captured layers.

        The pivot position is the last token before the model generates a response.
        This is where the model has "understood" the full prompt and is about to
        decide how to respond - the critical point for detecting attack compliance.
        """
        result = {}
        for layer_idx, activations in self.captured.items():
            # activations shape: [batch=1, seq_len, hidden_dim]
            # Extract at pivot position
            result[layer_idx] = activations[0, pivot_position, :].clone()
        return result


def format_conversation(conversation: list[dict], tokenizer) -> tuple[str, int]:
    """
    Format conversation for model input and find pivot position.

    Returns:
        - Formatted prompt string
        - Approximate token position of pivot (last token before response)
    """
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize to find length (pivot is at end of prompt, before generation)
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    pivot_position = tokens.shape[1] - 1  # Last token position

    return prompt, pivot_position


def capture_sample(
    model,
    tokenizer,
    hook: ActivationCaptureHook,
    conversation: list[dict],
    device: str = "cuda",
) -> tuple[dict[int, Tensor], int]:
    """
    Capture activations for a single conversation.

    Returns:
        - Dictionary mapping layer index to activation tensor at pivot
        - Pivot position in sequence
    """
    # Format conversation
    prompt, pivot_position = format_conversation(conversation, tokenizer)

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)

    # Run forward pass (hooks will capture activations)
    hook.captured.clear()

    with torch.no_grad():
        _ = model(**inputs)

    # Extract activations at pivot position
    activations = hook.get_pivot_activations(pivot_position)

    return activations, pivot_position


def load_dpo_data(data_path: Path) -> list[dict]:
    """Load DPO-formatted data for activation capture."""
    samples = []

    for split in ["train", "val", "test"]:
        filepath = data_path / f"{split}.jsonl"
        if filepath.exists():
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        item["_split"] = split
                        samples.append(item)

    return samples


def main():
    parser = argparse.ArgumentParser(description="TRYLOCK Activation Capture")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="Base model name")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to PEFT adapter (optional, for fine-tuned model)")
    parser.add_argument("--data_path", type=str, default="data/dpo",
                        help="Path to DPO data directory")
    parser.add_argument("--output_dir", type=str, default="activations",
                        help="Output directory for captured activations")
    parser.add_argument("--target_layers", type=str, default="12,14,16,18,20",
                        help="Comma-separated list of layer indices to capture")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to process (None = all)")
    parser.add_argument("--batch_save_size", type=int, default=100,
                        help="Save activations every N samples")

    args = parser.parse_args()

    # Parse target layers
    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    print("=" * 60)
    print("TRYLOCK ACTIVATION CAPTURE")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Adapter: {args.adapter_path or 'None (base model)'}")
    print(f"Target layers: {target_layers}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("[2/4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load adapter if specified
    if args.adapter_path:
        print(f"[2b/4] Loading PEFT adapter from {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    # Setup capture hook
    print("[3/4] Setting up activation hooks...")
    hook = ActivationCaptureHook(target_layers)
    hook.register(model)

    # Load data
    print("[4/4] Loading data...")
    data_path = Path(args.data_path)
    samples = load_dpo_data(data_path)
    print(f"   Loaded {len(samples)} samples")

    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"   Limited to {len(samples)} samples")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Capture activations
    print("\n" + "=" * 60)
    print("CAPTURING ACTIVATIONS")
    print("=" * 60)

    # Store all captures
    all_activations = {
        "attack": {layer: [] for layer in target_layers},
        "benign": {layer: [] for layer in target_layers},
    }
    metadata = []

    for i, sample in enumerate(tqdm(samples, desc="Capturing")):
        try:
            conversation = sample.get("prompt", [])
            sample_id = sample.get("id", f"sample_{i}")
            meta = sample.get("metadata", {})

            # Determine if this is an attack or benign
            attack_family = meta.get("family", "unknown")
            is_attack = attack_family not in ["benign_hard_negatives", "benign", "legitimate"]

            # Capture activations
            activations, pivot_pos = capture_sample(
                model, tokenizer, hook, conversation, device
            )

            # Store activations by category
            category = "attack" if is_attack else "benign"
            for layer_idx, act in activations.items():
                all_activations[category][layer_idx].append(act)

            # Store metadata
            metadata.append({
                "id": sample_id,
                "index": i,
                "family": attack_family,
                "type": meta.get("type", "unknown"),
                "difficulty": meta.get("difficulty", "unknown"),
                "is_attack": is_attack,
                "pivot_position": pivot_pos,
                "conversation_length": len(conversation),
            })

            # Periodic save
            if (i + 1) % args.batch_save_size == 0:
                print(f"\n   Checkpoint at {i + 1} samples...")

        except Exception as e:
            print(f"\n   Error processing sample {i}: {e}")
            continue

    # Clean up hooks
    hook.clear()

    # Save activations
    print("\n" + "=" * 60)
    print("SAVING ACTIVATIONS")
    print("=" * 60)

    # Stack and save activations for each category and layer
    for category in ["attack", "benign"]:
        for layer_idx in target_layers:
            acts = all_activations[category][layer_idx]
            if acts:
                stacked = torch.stack(acts)  # [num_samples, hidden_dim]
                filepath = output_dir / f"{category}_layer{layer_idx}.safetensors"
                save_file({"activations": stacked}, str(filepath))
                print(f"   Saved {filepath}: shape {stacked.shape}")

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "adapter_path": args.adapter_path,
            "target_layers": target_layers,
            "total_samples": len(metadata),
            "attack_count": sum(1 for m in metadata if m["is_attack"]),
            "benign_count": sum(1 for m in metadata if not m["is_attack"]),
            "samples": metadata,
        }, f, indent=2)
    print(f"   Saved metadata: {meta_path}")

    # Summary
    print("\n" + "=" * 60)
    print("CAPTURE COMPLETE")
    print("=" * 60)
    print(f"Total samples: {len(metadata)}")
    print(f"Attack samples: {sum(1 for m in metadata if m['is_attack'])}")
    print(f"Benign samples: {sum(1 for m in metadata if not m['is_attack'])}")
    print(f"Output directory: {output_dir}")
    print("\nNext step: Train RepE probes")
    print(f"  python training/train_repe.py --activations_dir {output_dir}")


if __name__ == "__main__":
    main()
