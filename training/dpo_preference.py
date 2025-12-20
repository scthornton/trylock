"""
TRYLOCK DPO Preference Training

Direct Preference Optimization for teaching models to prefer secure responses
over unsafe/overblocking alternatives.

Key training strategy:
1. Use (chosen, rejected_unsafe) pairs to teach attack resistance
2. Use (chosen, rejected_overblock) pairs to prevent over-refusal
3. Curriculum learning: easy → medium → hard → expert

Usage:
    python -m training.dpo_preference --config configs/llama3_8b_dpo.yaml

    # Or programmatically:
    from training.dpo_preference import run_dpo_training
    run_dpo_training(config)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except ImportError:
    LoraConfig = None
    get_peft_model = None
    PeftModel = None

try:
    from trl import DPOTrainer, DPOConfig
except ImportError:
    DPOTrainer = None
    DPOConfig = None


@dataclass
class DPOTrainingConfig:
    """Configuration for DPO preference training."""

    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    sft_checkpoint: str | None = None  # Load from SFT warmup
    trust_remote_code: bool = True

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # DPO-specific
    beta: float = 0.1  # KL penalty coefficient
    loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid"
    label_smoothing: float = 0.0
    reference_free: bool = False  # If True, skip reference model

    # Training
    learning_rate: float = 5e-6  # Lower than SFT
    num_epochs: int = 1
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 4096
    max_prompt_length: int = 2048

    # Data
    train_file: str = "data/tier1_open/attacks/train.jsonl"
    eval_file: str | None = None
    max_samples: int | None = None

    # Preference pair strategy
    unsafe_weight: float = 0.6  # Weight for (chosen, rejected_unsafe) pairs
    overblock_weight: float = 0.4  # Weight for (chosen, rejected_overblock) pairs

    # Curriculum learning
    use_curriculum: bool = True
    curriculum_schedule: list[str] = field(
        default_factory=lambda: ["easy", "medium", "hard", "expert"]
    )
    curriculum_epochs_per_stage: int = 1

    # Output
    output_dir: str = "./outputs/trylock-dpo"
    save_steps: int = 500
    logging_steps: int = 10

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DPOTrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("dpo", data))


class TRYLOCKDPODataset(Dataset):
    """
    Dataset for DPO preference training.

    Converts TRYLOCK trajectories into (prompt, chosen, rejected) triplets
    for preference learning.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int = 4096,
        max_prompt_length: int = 2048,
        max_samples: int | None = None,
        pair_type: Literal["unsafe", "overblock", "mixed"] = "mixed",
        unsafe_weight: float = 0.6,
        difficulty_filter: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.pair_type = pair_type
        self.unsafe_weight = unsafe_weight
        self.examples = []

        # Load trajectories
        with open(data_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    traj = json.loads(line)

                    # Apply difficulty filter
                    if difficulty_filter:
                        diff = traj.get("attack_metadata", {}).get("difficulty")
                        if diff not in difficulty_filter:
                            continue

                    examples = self._process_trajectory(traj)
                    self.examples.extend(examples)

    def _process_trajectory(self, trajectory: dict) -> list[dict]:
        """Convert trajectory to preference pairs."""
        conversation = trajectory.get("conversation", [])
        responses = trajectory.get("responses", {})
        chosen = responses.get("chosen", {})

        if not chosen.get("content"):
            return []

        # Build prompt from conversation
        messages = []
        for turn in conversation:
            role = turn.get("role")
            content = turn.get("content")
            if role in ["user", "assistant", "system"]:
                messages.append({"role": role, "content": content})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        chosen_content = chosen["content"]
        examples = []

        # Create pairs based on configuration
        rejected_unsafe = responses.get("rejected_unsafe", {})
        rejected_overblock = responses.get("rejected_overblock", {})

        is_attack = trajectory.get("attack_metadata", {}).get("family") != "benign"

        # For attacks: prefer (chosen, rejected_unsafe)
        # For benign: prefer (chosen, rejected_overblock)
        if self.pair_type == "unsafe" or (self.pair_type == "mixed" and is_attack):
            if rejected_unsafe.get("content"):
                examples.append({
                    "prompt": prompt,
                    "chosen": chosen_content,
                    "rejected": rejected_unsafe["content"],
                    "pair_type": "unsafe",
                    "trajectory_id": trajectory.get("id"),
                    "difficulty": trajectory.get("attack_metadata", {}).get("difficulty"),
                })

        if self.pair_type == "overblock" or (self.pair_type == "mixed" and not is_attack):
            if rejected_overblock.get("content"):
                examples.append({
                    "prompt": prompt,
                    "chosen": chosen_content,
                    "rejected": rejected_overblock["content"],
                    "pair_type": "overblock",
                    "trajectory_id": trajectory.get("id"),
                    "difficulty": trajectory.get("attack_metadata", {}).get("difficulty"),
                })

        # Mixed mode with both pair types for attacks
        if self.pair_type == "mixed" and is_attack:
            # Also add overblock pairs for attack trajectories (prevents false positives)
            if rejected_overblock.get("content"):
                examples.append({
                    "prompt": prompt,
                    "chosen": chosen_content,
                    "rejected": rejected_overblock["content"],
                    "pair_type": "overblock",
                    "trajectory_id": trajectory.get("id"),
                    "difficulty": trajectory.get("attack_metadata", {}).get("difficulty"),
                })

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        # Return format expected by DPOTrainer
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }


class CurriculumDPODataset(Dataset):
    """
    Dataset that implements curriculum learning for DPO.

    Progressively introduces harder examples during training.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int = 4096,
        max_prompt_length: int = 2048,
        max_samples: int | None = None,
        pair_type: Literal["unsafe", "overblock", "mixed"] = "mixed",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.pair_type = pair_type

        # Load all trajectories grouped by difficulty
        self.examples_by_difficulty = {
            "easy": [],
            "medium": [],
            "hard": [],
            "expert": [],
        }

        with open(data_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    traj = json.loads(line)
                    difficulty = traj.get("attack_metadata", {}).get("difficulty", "medium")
                    examples = self._process_trajectory(traj)

                    if difficulty in self.examples_by_difficulty:
                        self.examples_by_difficulty[difficulty].extend(examples)
                    else:
                        self.examples_by_difficulty["medium"].extend(examples)

        # Track current curriculum stage
        self.current_difficulties = ["easy"]
        self._rebuild_active_examples()

    def _process_trajectory(self, trajectory: dict) -> list[dict]:
        """Convert trajectory to preference pairs."""
        conversation = trajectory.get("conversation", [])
        responses = trajectory.get("responses", {})
        chosen = responses.get("chosen", {})

        if not chosen.get("content"):
            return []

        messages = []
        for turn in conversation:
            role = turn.get("role")
            content = turn.get("content")
            if role in ["user", "assistant", "system"]:
                messages.append({"role": role, "content": content})

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        chosen_content = chosen["content"]
        examples = []

        rejected_unsafe = responses.get("rejected_unsafe", {})
        rejected_overblock = responses.get("rejected_overblock", {})
        is_attack = trajectory.get("attack_metadata", {}).get("family") != "benign"

        if self.pair_type in ["unsafe", "mixed"] and is_attack:
            if rejected_unsafe.get("content"):
                examples.append({
                    "prompt": prompt,
                    "chosen": chosen_content,
                    "rejected": rejected_unsafe["content"],
                    "difficulty": trajectory.get("attack_metadata", {}).get("difficulty"),
                })

        if self.pair_type in ["overblock", "mixed"]:
            if rejected_overblock.get("content"):
                examples.append({
                    "prompt": prompt,
                    "chosen": chosen_content,
                    "rejected": rejected_overblock["content"],
                    "difficulty": trajectory.get("attack_metadata", {}).get("difficulty"),
                })

        return examples

    def _rebuild_active_examples(self):
        """Rebuild active example list based on current curriculum."""
        self.active_examples = []
        for diff in self.current_difficulties:
            self.active_examples.extend(self.examples_by_difficulty.get(diff, []))
        random.shuffle(self.active_examples)

    def advance_curriculum(self, new_difficulty: str):
        """Add a new difficulty level to the curriculum."""
        if new_difficulty not in self.current_difficulties:
            self.current_difficulties.append(new_difficulty)
            self._rebuild_active_examples()
            print(f"Curriculum advanced: now including {self.current_difficulties}")

    def set_curriculum(self, difficulties: list[str]):
        """Set specific difficulties for curriculum."""
        self.current_difficulties = difficulties
        self._rebuild_active_examples()

    def __len__(self) -> int:
        return len(self.active_examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.active_examples[idx]
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }


def setup_model_and_tokenizer(config: DPOTrainingConfig):
    """Load and configure model and tokenizer for DPO."""
    if AutoModelForCausalLM is None:
        raise ImportError("transformers is required")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (from SFT checkpoint if provided)
    model_path = config.sft_checkpoint or config.model_name

    if config.sft_checkpoint and PeftModel is not None:
        # Load base model with LoRA from SFT
        print(f"Loading from SFT checkpoint: {config.sft_checkpoint}")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=config.trust_remote_code,
        )
        model = PeftModel.from_pretrained(base_model, config.sft_checkpoint)
        # Merge and unload for DPO training
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=config.trust_remote_code,
        )

    # Apply new LoRA for DPO
    if config.use_lora:
        if LoraConfig is None:
            raise ImportError("peft is required for LoRA")

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Reference model (for DPO KL penalty)
    if not config.reference_free:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=config.trust_remote_code,
        )
        ref_model.eval()
    else:
        ref_model = None

    return model, ref_model, tokenizer


def run_dpo_training(config: DPOTrainingConfig) -> str:
    """
    Run DPO preference training.

    Args:
        config: Training configuration

    Returns:
        Path to saved model
    """
    if torch is None:
        raise ImportError("torch is required")
    if DPOTrainer is None:
        raise ImportError("trl is required for DPO training")

    print("TRYLOCK DPO Preference Training")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"SFT Checkpoint: {config.sft_checkpoint or 'None'}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"DPO Beta: {config.beta}")
    print(f"Data: {config.train_file}")
    print(f"Output: {config.output_dir}")
    print(f"Curriculum: {config.use_curriculum}")
    print()

    # Setup model and tokenizer
    model, ref_model, tokenizer = setup_model_and_tokenizer(config)

    # Load dataset
    if config.use_curriculum:
        train_dataset = CurriculumDPODataset(
            config.train_file,
            tokenizer,
            max_length=config.max_seq_length,
            max_prompt_length=config.max_prompt_length,
            max_samples=config.max_samples,
            pair_type="mixed",
        )
    else:
        train_dataset = TRYLOCKDPODataset(
            config.train_file,
            tokenizer,
            max_length=config.max_seq_length,
            max_prompt_length=config.max_prompt_length,
            max_samples=config.max_samples,
            pair_type="mixed",
            unsafe_weight=config.unsafe_weight,
        )

    print(f"Training examples: {len(train_dataset)}")

    eval_dataset = None
    if config.eval_file:
        eval_dataset = TRYLOCKDPODataset(
            config.eval_file,
            tokenizer,
            max_length=config.max_seq_length,
            max_prompt_length=config.max_prompt_length,
        )
        print(f"Evaluation examples: {len(eval_dataset)}")

    # DPO Training arguments
    if DPOConfig is not None:
        training_args = DPOConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=3,
            bf16=config.bf16,
            remove_unused_columns=False,
            report_to="none",
            beta=config.beta,
            loss_type=config.loss_type,
            label_smoothing=config.label_smoothing,
            max_length=config.max_seq_length,
            max_prompt_length=config.max_prompt_length,
        )
    else:
        # Fallback to TrainingArguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.per_device_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            save_total_limit=3,
            bf16=config.bf16,
            remove_unused_columns=False,
            report_to="none",
        )

    # Curriculum training loop
    if config.use_curriculum and isinstance(train_dataset, CurriculumDPODataset):
        print("\nStarting curriculum training...")

        for stage_idx, difficulty in enumerate(config.curriculum_schedule):
            print(f"\n--- Curriculum Stage {stage_idx + 1}: {difficulty} ---")

            # Update curriculum
            train_dataset.advance_curriculum(difficulty)

            # Create trainer for this stage
            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
            )

            # Train for configured epochs per stage
            for epoch in range(config.curriculum_epochs_per_stage):
                print(f"Stage {difficulty}, Epoch {epoch + 1}")
                trainer.train()

            # Save checkpoint after each stage
            stage_dir = Path(config.output_dir) / f"stage_{difficulty}"
            trainer.save_model(str(stage_dir))
    else:
        # Standard training
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        print("\nStarting training...")
        trainer.train()

    # Save final model
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    return config.output_dir


def compute_dpo_metrics(
    model,
    ref_model,
    tokenizer,
    eval_dataset,
    beta: float = 0.1,
) -> dict:
    """
    Compute DPO-specific metrics.

    Returns:
        Dictionary with accuracy, reward margins, etc.
    """
    if torch is None:
        raise ImportError("torch is required")

    model.eval()
    if ref_model is not None:
        ref_model.eval()

    correct = 0
    total = 0
    reward_margins = []

    with torch.no_grad():
        for example in eval_dataset:
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]

            # Compute log probs for chosen and rejected
            chosen_logprob = _compute_log_prob(
                model, tokenizer, prompt, chosen
            )
            rejected_logprob = _compute_log_prob(
                model, tokenizer, prompt, rejected
            )

            if ref_model is not None:
                ref_chosen_logprob = _compute_log_prob(
                    ref_model, tokenizer, prompt, chosen
                )
                ref_rejected_logprob = _compute_log_prob(
                    ref_model, tokenizer, prompt, rejected
                )

                # DPO reward difference
                reward_chosen = beta * (chosen_logprob - ref_chosen_logprob)
                reward_rejected = beta * (rejected_logprob - ref_rejected_logprob)
            else:
                reward_chosen = chosen_logprob
                reward_rejected = rejected_logprob

            if reward_chosen > reward_rejected:
                correct += 1
            total += 1

            reward_margins.append(reward_chosen - reward_rejected)

    return {
        "accuracy": correct / total if total > 0 else 0,
        "mean_reward_margin": sum(reward_margins) / len(reward_margins) if reward_margins else 0,
        "total_examples": total,
    }


def _compute_log_prob(model, tokenizer, prompt: str, response: str) -> float:
    """Compute log probability of response given prompt."""
    full_text = prompt + response
    inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prompt_inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get log probs for response tokens only
    prompt_len = prompt_inputs.input_ids.shape[1]
    response_logits = logits[:, prompt_len - 1:-1]
    response_ids = inputs.input_ids[:, prompt_len:]

    log_probs = torch.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum().item()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRYLOCK DPO Preference Training")
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
        "--sft-checkpoint",
        type=str,
        default=None,
        help="Path to SFT warmup checkpoint",
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
        default="./outputs/trylock-dpo",
        help="Output directory",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta (KL penalty)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum training samples",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum learning",
    )

    args = parser.parse_args()

    if args.config:
        config = DPOTrainingConfig.from_yaml(args.config)
    else:
        config = DPOTrainingConfig(
            model_name=args.model,
            sft_checkpoint=args.sft_checkpoint,
            train_file=args.train_file,
            output_dir=args.output_dir,
            beta=args.beta,
            num_epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            max_samples=args.max_samples,
            use_curriculum=not args.no_curriculum,
        )

    run_dpo_training(config)
