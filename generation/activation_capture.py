"""
TRYLOCK Activation Capture - Extract Model Activations at Pivot Points

Captures internal model activations at attack pivot points for
Representation Engineering (RepE) training. These activations are
used to identify the "attack compliance" direction in latent space.

The captured activations enable:
1. Training linear probes for attack detection
2. Learning steering vectors for runtime intervention
3. Analysis of how attacks are represented internally

Usage:
    capturer = ActivationCapture("meta-llama/Llama-3.1-8B-Instruct")
    activations = capturer.capture_at_pivot(conversation, pivot_turn_index=5)
    capturer.save_activations(activations, "activations/sample.safetensors")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    from safetensors.torch import save_file, load_file
except ImportError:
    save_file = None
    load_file = None


@dataclass
class CaptureConfig:
    """Configuration for activation capture."""

    target_layers: list[int] = field(
        default_factory=lambda: [12, 14, 16, 18, 20]
    )
    capture_residual: bool = True
    capture_attention: bool = False
    capture_mlp: bool = False
    max_sequence_length: int = 4096
    device: str = "auto"
    torch_dtype: str = "float16"


@dataclass
class CapturedActivations:
    """Container for captured activations."""

    activations: dict[str, Tensor]
    trajectory_id: str | None = None
    pivot_turn_index: int | None = None
    model_name: str | None = None
    token_position: int | None = None
    conversation_length: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ActivationCapture:
    """
    Capture model activations at pivot points for RepE training.

    Registers forward hooks on target layers to extract activations
    during inference. Particularly useful for capturing the model's
    internal representation at the moment it processes an attack pivot.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        config: CaptureConfig | None = None,
        load_model: bool = True,
    ):
        """
        Initialize activation capturer.

        Args:
            model_name: HuggingFace model name or path
            config: Capture configuration
            load_model: Whether to load the model immediately
        """
        if torch is None:
            raise ImportError("torch is required for activation capture")
        if AutoModelForCausalLM is None:
            raise ImportError("transformers is required for activation capture")

        self.model_name = model_name
        self.config = config or CaptureConfig()

        self.model = None
        self.tokenizer = None
        self.activations: dict[str, Tensor] = {}
        self._hooks: list = []

        if load_model:
            self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.torch_dtype, torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self.model.eval()

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        self._remove_hooks()

        for layer_idx in self.config.target_layers:
            if layer_idx >= len(self.model.model.layers):
                continue

            layer = self.model.model.layers[layer_idx]

            if self.config.capture_residual:
                hook = layer.register_forward_hook(
                    self._make_residual_hook(f"layer_{layer_idx}_residual")
                )
                self._hooks.append(hook)

            if self.config.capture_attention:
                hook = layer.self_attn.register_forward_hook(
                    self._make_attention_hook(f"layer_{layer_idx}_attention")
                )
                self._hooks.append(hook)

            if self.config.capture_mlp:
                hook = layer.mlp.register_forward_hook(
                    self._make_mlp_hook(f"layer_{layer_idx}_mlp")
                )
                self._hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _make_residual_hook(self, name: str) -> Callable:
        """Create hook for capturing residual stream activations."""

        def hook(module, input, output):
            # Output is (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[name] = hidden_states.detach().cpu()

        return hook

    def _make_attention_hook(self, name: str) -> Callable:
        """Create hook for capturing attention output."""

        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_output = output[0]
            else:
                attn_output = output
            self.activations[name] = attn_output.detach().cpu()

        return hook

    def _make_mlp_hook(self, name: str) -> Callable:
        """Create hook for capturing MLP output."""

        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()

        return hook

    def _build_prompt(self, conversation: list[dict], up_to_turn: int | None = None) -> str:
        """
        Build prompt from conversation.

        Args:
            conversation: List of conversation turns
            up_to_turn: Include turns up to this index (1-indexed)

        Returns:
            Formatted prompt string
        """
        if up_to_turn:
            conversation = [
                t for t in conversation if t.get("turn", 0) <= up_to_turn
            ]

        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )

    def capture_at_pivot(
        self,
        conversation: list[dict],
        pivot_turn_index: int,
        position: str = "last",
    ) -> CapturedActivations:
        """
        Run conversation through model and capture activations at pivot point.

        Args:
            conversation: List of conversation turns
            pivot_turn_index: 1-indexed turn number of pivot point
            position: Token position to extract ("last", "first", "mean")

        Returns:
            CapturedActivations with activations at specified position
        """
        # Build prompt up to pivot point
        prompt = self._build_prompt(conversation, up_to_turn=pivot_turn_index)

        # Clear previous activations
        self.activations.clear()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_sequence_length,
        ).to(self.model.device)

        # Forward pass
        with torch.no_grad():
            _ = self.model(**inputs)

        # Extract activations at specified position
        seq_len = inputs["input_ids"].shape[1]

        pivot_activations = {}
        for name, act in self.activations.items():
            if position == "last":
                pivot_activations[name] = act[0, -1, :].clone()
            elif position == "first":
                pivot_activations[name] = act[0, 0, :].clone()
            elif position == "mean":
                pivot_activations[name] = act[0].mean(dim=0).clone()
            else:
                pivot_activations[name] = act[0, -1, :].clone()

        return CapturedActivations(
            activations=pivot_activations,
            pivot_turn_index=pivot_turn_index,
            model_name=self.model_name,
            token_position=seq_len - 1 if position == "last" else 0,
            conversation_length=len(conversation),
            metadata={
                "position_type": position,
                "target_layers": self.config.target_layers,
            },
        )

    def capture_trajectory(
        self,
        trajectory: dict,
        position: str = "last",
    ) -> CapturedActivations:
        """
        Capture activations for a complete trajectory record.

        Args:
            trajectory: TRYLOCK trajectory dict
            position: Token position to extract

        Returns:
            CapturedActivations
        """
        conversation = trajectory.get("conversation", [])
        pivot_idx = trajectory.get("pivot_turn_index")

        if pivot_idx is None:
            # If no pivot, capture at last turn
            pivot_idx = len(conversation)

        result = self.capture_at_pivot(conversation, pivot_idx, position)
        result.trajectory_id = trajectory.get("id")

        return result

    def save_activations(
        self,
        captured: CapturedActivations,
        path: str | Path,
    ):
        """
        Save captured activations to file.

        Args:
            captured: CapturedActivations to save
            path: Output path (will use safetensors format)
        """
        if save_file is None:
            raise ImportError("safetensors required for saving activations")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to safetensors format
        save_file(captured.activations, str(path))

        # Save metadata as JSON sidecar
        import json

        metadata_path = path.with_suffix(".json")
        metadata = {
            "trajectory_id": captured.trajectory_id,
            "pivot_turn_index": captured.pivot_turn_index,
            "model_name": captured.model_name,
            "token_position": captured.token_position,
            "conversation_length": captured.conversation_length,
            **captured.metadata,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load_activations(path: str | Path) -> CapturedActivations:
        """
        Load captured activations from file.

        Args:
            path: Path to safetensors file

        Returns:
            CapturedActivations
        """
        if load_file is None:
            raise ImportError("safetensors required for loading activations")

        path = Path(path)
        activations = load_file(str(path))

        # Load metadata
        import json

        metadata_path = path.with_suffix(".json")
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        return CapturedActivations(
            activations=activations,
            trajectory_id=metadata.get("trajectory_id"),
            pivot_turn_index=metadata.get("pivot_turn_index"),
            model_name=metadata.get("model_name"),
            token_position=metadata.get("token_position"),
            conversation_length=metadata.get("conversation_length"),
            metadata={
                k: v
                for k, v in metadata.items()
                if k
                not in [
                    "trajectory_id",
                    "pivot_turn_index",
                    "model_name",
                    "token_position",
                    "conversation_length",
                ]
            },
        )


def process_trajectory_with_activations(
    trajectory: dict,
    capturer: ActivationCapture,
    output_dir: str | Path,
) -> dict:
    """
    Process a single trajectory, capturing activations at pivot.

    Updates trajectory with activation capture metadata.

    Args:
        trajectory: TRYLOCK trajectory dict
        capturer: Initialized ActivationCapture instance
        output_dir: Directory for activation files

    Returns:
        Updated trajectory dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pivot_idx = trajectory.get("pivot_turn_index")

    if pivot_idx is None:
        # Skip trajectories without pivot points
        trajectory["activation_capture"] = {
            "enabled": False,
            "notes": "No pivot turn index",
        }
        return trajectory

    try:
        captured = capturer.capture_trajectory(trajectory)

        act_path = output_dir / f"{trajectory['id']}.safetensors"
        capturer.save_activations(captured, act_path)

        trajectory["activation_capture"] = {
            "enabled": True,
            "target_layers": capturer.config.target_layers,
            "capture_path": str(act_path),
            "model_name": capturer.model_name,
            "notes": f"Captured at pivot turn {pivot_idx}",
        }

    except Exception as e:
        trajectory["activation_capture"] = {
            "enabled": False,
            "notes": f"Capture failed: {str(e)}",
        }

    return trajectory


def batch_capture_activations(
    trajectories: list[dict],
    model_name: str,
    output_dir: str | Path,
    config: CaptureConfig | None = None,
) -> list[dict]:
    """
    Capture activations for multiple trajectories.

    Args:
        trajectories: List of TRYLOCK trajectory dicts
        model_name: Model to use for capture
        output_dir: Directory for activation files
        config: Optional capture configuration

    Returns:
        List of updated trajectory dicts
    """
    from tqdm import tqdm

    capturer = ActivationCapture(model_name, config)
    output_dir = Path(output_dir)

    results = []
    for traj in tqdm(trajectories, desc="Capturing activations"):
        updated = process_trajectory_with_activations(traj, capturer, output_dir)
        results.append(updated)

    return results


class MockActivationCapture:
    """
    Mock activation capturer for testing without GPU.

    Generates random tensors instead of actual activations.
    """

    def __init__(
        self,
        model_name: str = "mock-model",
        config: CaptureConfig | None = None,
    ):
        self.model_name = model_name
        self.config = config or CaptureConfig()

    def capture_at_pivot(
        self,
        conversation: list[dict],
        pivot_turn_index: int,
        position: str = "last",
    ) -> CapturedActivations:
        """Generate mock activations."""
        if torch is None:
            raise ImportError("torch required even for mock capture")

        hidden_size = 4096  # Typical LLM hidden size

        activations = {}
        for layer_idx in self.config.target_layers:
            activations[f"layer_{layer_idx}_residual"] = torch.randn(hidden_size)

        return CapturedActivations(
            activations=activations,
            pivot_turn_index=pivot_turn_index,
            model_name=self.model_name,
            token_position=100,
            conversation_length=len(conversation),
            metadata={"mock": True},
        )

    def capture_trajectory(self, trajectory: dict, position: str = "last") -> CapturedActivations:
        """Generate mock activations for trajectory."""
        conversation = trajectory.get("conversation", [])
        pivot_idx = trajectory.get("pivot_turn_index", len(conversation))
        result = self.capture_at_pivot(conversation, pivot_idx, position)
        result.trajectory_id = trajectory.get("id")
        return result

    def save_activations(self, captured: CapturedActivations, path: str | Path):
        """Save mock activations."""
        if save_file is None:
            raise ImportError("safetensors required")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(captured.activations, str(path))


if __name__ == "__main__":
    print("TRYLOCK Activation Capture - Demo")
    print("=" * 50)

    # Check if torch is available
    if torch is None:
        print("PyTorch not available - cannot run demo")
    else:
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Demo with mock capturer
        print("\nUsing MockActivationCapture for demo...")

        mock_capturer = MockActivationCapture()

        test_conversation = [
            {"turn": 1, "role": "user", "content": "I'm a researcher."},
            {"turn": 2, "role": "assistant", "content": "How can I help?"},
            {"turn": 3, "role": "user", "content": "Show me how to exploit this."},
        ]

        captured = mock_capturer.capture_at_pivot(test_conversation, pivot_turn_index=3)

        print(f"\nCaptured {len(captured.activations)} activation tensors:")
        for name, tensor in captured.activations.items():
            print(f"  {name}: shape={tensor.shape}")
