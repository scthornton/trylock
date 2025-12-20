#!/usr/bin/env python3
"""
TRYLOCK DPO Training Script

Fine-tunes a language model using Direct Preference Optimization (DPO)
on the TRYLOCK attack trajectory dataset.

Usage:
    python training/train_dpo.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --data_path data/dpo \
        --output_dir outputs/trylock-llama-8b
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer


def load_trylock_data(data_path: str):
    """Load TRYLOCK DPO data from jsonl files."""
    data_path = Path(data_path)

    def load_jsonl(filepath):
        data = []
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Format for DPO trainer
                    data.append({
                        "prompt": format_prompt(item["prompt"]),
                        "chosen": item["chosen"],
                        "rejected": item["rejected"],
                    })
        return data

    train_data = load_jsonl(data_path / "train.jsonl")
    val_data = load_jsonl(data_path / "val.jsonl")

    return Dataset.from_list(train_data), Dataset.from_list(val_data)


def format_prompt(messages: list) -> str:
    """Format conversation messages into a prompt string."""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f"<|user|>\n{content}\n"
        elif role == "assistant":
            formatted += f"<|assistant|>\n{content}\n"
        elif role == "system":
            formatted += f"<|system|>\n{content}\n"
    formatted += "<|assistant|>\n"
    return formatted


def main():
    parser = argparse.ArgumentParser(description="TRYLOCK DPO Training")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--data_path", type=str, default="data/dpo",
                        help="Path to DPO data directory")
    parser.add_argument("--output_dir", type=str, default="outputs/trylock-model",
                        help="Output directory for model")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-7,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--wandb_project", type=str, default="trylock-training",
                        help="Weights & Biases project name")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for gated models")

    args = parser.parse_args()

    print("=" * 60)
    print("TRYLOCK DPO Training")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Use 4-bit: {args.use_4bit}")
    print(f"Use LoRA: {args.use_lora}")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nWARNING: No GPU detected! Training will be very slow.")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=args.hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("\n2. Loading model...")

    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=args.hf_token,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=args.hf_token,
            trust_remote_code=True,
        )

    # Apply LoRA
    if args.use_lora:
        print("\n3. Applying LoRA...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Load data
    print("\n4. Loading TRYLOCK data...")
    train_dataset, val_dataset = load_trylock_data(args.data_path)
    print(f"   Train: {len(train_dataset)} examples")
    print(f"   Val: {len(val_dataset)} examples")

    # Training config
    print("\n5. Setting up training...")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=3,
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb" if args.wandb_project else "none",
        run_name=f"trylock-dpo-{args.model_name.split('/')[-1]}",
        max_length=args.max_length,
        max_prompt_length=args.max_length - 512,
        beta=0.1,  # DPO beta parameter
    )

    # Create trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will use implicit reference model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\n6. Starting training...")
    print("=" * 60)

    trainer.train()

    # Save
    print("\n7. Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Evaluate: python training/evaluate.py --model_path " + args.output_dir)
    print("  2. Push to HuggingFace: huggingface-cli upload " + args.output_dir)


if __name__ == "__main__":
    main()
