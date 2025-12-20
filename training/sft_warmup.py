"""
TRYLOCK SFT Warmup Training

Light supervised fine-tuning on chosen responses to establish baseline
security-aware behavior before DPO preference learning.

This warmup phase:
1. Teaches the model the format of ideal responses
2. Establishes basic security awareness
3. Prepares the model for preference learning

Usage:
    python -m training.sft_warmup --config configs/llama3_8b_lora.yaml

    # Or programmatically:
    from training.sft_warmup import run_sft_warmup
    run_sft_warmup(config)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
        Trainer,
        DataCollatorForLanguageModeling,
    )
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None
    Trainer = None

try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    LoraConfig = None
    get_peft_model = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


@dataclass
class SFTConfig:
    """Configuration for SFT warmup training."""

    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    trust_remote_code: bool = True

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 2
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 4096

    # Data
    train_file: str = "data/tier1_open/attacks/train.jsonl"
    eval_file: str | None = None
    max_samples: int | None = None

    # Training mix ratios
    attack_ratio: float = 0.40  # Ideal responses to attacks
    benign_ratio: float = 0.35  # Responses to benign hard negatives
    capability_ratio: float = 0.25  # General helpfulness (optional)

    # Output
    output_dir: str = "./outputs/trylock-sft"
    save_steps: int = 500
    logging_steps: int = 10

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SFTConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("sft", data))


class TRYLOCKSFTDataset(Dataset):
    """
    Dataset for SFT warmup training.

    Converts TRYLOCK trajectories into (prompt, completion) pairs
    using the chosen responses.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int = 4096,
        max_samples: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load trajectories
        with open(data_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    traj = json.loads(line)
                    example = self._process_trajectory(traj)
                    if example:
                        self.examples.append(example)

    def _process_trajectory(self, trajectory: dict) -> dict | None:
        """Convert trajectory to training example."""
        conversation = trajectory.get("conversation", [])
        responses = trajectory.get("responses", {})
        chosen = responses.get("chosen", {})

        if not chosen.get("content"):
            return None

        # Build prompt from conversation (up to last user turn)
        messages = []
        for turn in conversation:
            role = turn.get("role")
            content = turn.get("content")
            if role in ["user", "assistant", "system"]:
                messages.append({"role": role, "content": content})

        # The chosen response is what we want the model to learn
        chosen_content = chosen["content"]

        # Format as chat
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Full text includes the response
        full_text = prompt + chosen_content + self.tokenizer.eos_token

        return {
            "text": full_text,
            "prompt": prompt,
            "response": chosen_content,
            "trajectory_id": trajectory.get("id"),
            "attack_family": trajectory.get("attack_metadata", {}).get("family"),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        # Tokenize
        encodings = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # Create labels (mask prompt tokens with -100)
        prompt_encodings = self.tokenizer(
            example["prompt"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        labels = encodings["input_ids"].copy()
        prompt_len = len(prompt_encodings["input_ids"])

        # Mask prompt tokens
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }


class SFTDataCollator:
    """Data collator for SFT that handles variable length sequences."""

    def __init__(self, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict]) -> dict:
        # Find max length in batch
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length,
        )

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for feature in features:
            # Pad or truncate
            input_ids = feature["input_ids"][:max_len]
            attention_mask = feature["attention_mask"][:max_len]
            labels = feature["labels"][:max_len]

            # Pad to max_len
            padding_len = max_len - len(input_ids)
            if padding_len > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
                attention_mask = attention_mask + [0] * padding_len
                labels = labels + [-100] * padding_len

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch


def setup_model_and_tokenizer(config: SFTConfig):
    """Load and configure model and tokenizer."""
    if AutoModelForCausalLM is None:
        raise ImportError("transformers is required")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
    )

    # Apply LoRA if configured
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

    return model, tokenizer


def run_sft_warmup(config: SFTConfig) -> str:
    """
    Run SFT warmup training.

    Args:
        config: Training configuration

    Returns:
        Path to saved model
    """
    if torch is None:
        raise ImportError("torch is required")
    if Trainer is None:
        raise ImportError("transformers is required")

    print(f"TRYLOCK SFT Warmup Training")
    print(f"=" * 50)
    print(f"Model: {config.model_name}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Data: {config.train_file}")
    print(f"Output: {config.output_dir}")
    print()

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load dataset
    train_dataset = TRYLOCKSFTDataset(
        config.train_file,
        tokenizer,
        max_length=config.max_seq_length,
        max_samples=config.max_samples,
    )

    print(f"Training examples: {len(train_dataset)}")

    eval_dataset = None
    if config.eval_file:
        eval_dataset = TRYLOCKSFTDataset(
            config.eval_file,
            tokenizer,
            max_length=config.max_seq_length,
        )
        print(f"Evaluation examples: {len(eval_dataset)}")

    # Data collator
    data_collator = SFTDataCollator(tokenizer, config.max_seq_length)

    # Training arguments
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
        report_to="none",  # Disable wandb by default
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    return config.output_dir


def prepare_mixed_dataset(
    attack_file: str,
    benign_file: str,
    output_file: str,
    attack_ratio: float = 0.5,
    max_samples: int | None = None,
):
    """
    Prepare a mixed dataset from attack and benign trajectories.

    Args:
        attack_file: Path to attack trajectories
        benign_file: Path to benign hard negatives
        output_file: Output path for mixed dataset
        attack_ratio: Ratio of attack examples
        max_samples: Maximum total samples
    """
    import random

    attacks = []
    benign = []

    # Load attacks
    with open(attack_file) as f:
        for line in f:
            if line.strip():
                attacks.append(json.loads(line))

    # Load benign
    with open(benign_file) as f:
        for line in f:
            if line.strip():
                benign.append(json.loads(line))

    # Calculate counts
    if max_samples:
        attack_count = int(max_samples * attack_ratio)
        benign_count = max_samples - attack_count
    else:
        attack_count = len(attacks)
        benign_count = len(benign)

    # Sample
    attack_sample = random.sample(attacks, min(attack_count, len(attacks)))
    benign_sample = random.sample(benign, min(benign_count, len(benign)))

    # Combine and shuffle
    combined = attack_sample + benign_sample
    random.shuffle(combined)

    # Save
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for traj in combined:
            f.write(json.dumps(traj) + "\n")

    print(f"Created mixed dataset: {len(combined)} examples")
    print(f"  Attacks: {len(attack_sample)}")
    print(f"  Benign: {len(benign_sample)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRYLOCK SFT Warmup Training")
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
        "--train-file",
        type=str,
        default="data/tier1_open/attacks/train.jsonl",
        help="Training data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/trylock-sft",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum training samples",
    )

    args = parser.parse_args()

    if args.config:
        config = SFTConfig.from_yaml(args.config)
    else:
        config = SFTConfig(
            model_name=args.model,
            train_file=args.train_file,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            per_device_batch_size=args.batch_size,
            max_samples=args.max_samples,
        )

    run_sft_warmup(config)
