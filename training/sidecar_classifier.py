"""
TRYLOCK Sidecar Classifier Training

Train a lightweight security classifier that runs in parallel with the main
model to provide conversation state scoring. The classifier is invisible to
attackers and provides real-time threat assessment.

Classification levels:
    - SAFE (0.0-0.3): Normal conversation, no intervention needed
    - WARN (0.3-0.7): Elevated risk, enable enhanced monitoring
    - ATTACK (0.7-1.0): Active attack detected, intervention required

Usage:
    python -m training.sidecar_classifier --config configs/sidecar_8b.yaml

    # Or programmatically:
    from training.sidecar_classifier import train_sidecar
    model = train_sidecar(config)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    nn = None
    Dataset = object
    DataLoader = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
    )
except ImportError:
    AutoModelForSequenceClassification = None
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
    from sklearn.metrics import (
        classification_report,
        precision_recall_fscore_support,
        roc_auc_score,
    )
except ImportError:
    classification_report = None
    precision_recall_fscore_support = None
    roc_auc_score = None


@dataclass
class SidecarConfig:
    """Configuration for sidecar classifier training."""

    # Model - use smaller model for inference efficiency
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    model_type: Literal["sequence_classification", "causal_lm"] = "sequence_classification"
    trust_remote_code: bool = True

    # Classification
    num_labels: int = 3  # SAFE, WARN, ATTACK
    label_names: list[str] = field(
        default_factory=lambda: ["SAFE", "WARN", "ATTACK"]
    )
    problem_type: Literal["single_label", "multi_label", "regression"] = "single_label"

    # LoRA - lighter config for smaller model
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 3
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048

    # Data
    train_file: str = "data/tier1_open/attacks/train.jsonl"
    eval_file: str | None = None
    max_samples: int | None = None

    # Class balancing
    use_class_weights: bool = True
    oversample_minority: bool = False

    # Thresholds for classification
    warn_threshold: float = 0.3
    attack_threshold: float = 0.7

    # Output
    output_dir: str = "./outputs/trylock-sidecar"
    save_steps: int = 500
    logging_steps: int = 10

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SidecarConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("sidecar", data))


class SidecarDataset(Dataset):
    """
    Dataset for sidecar classifier training.

    Converts TRYLOCK trajectories into classification examples based on
    attack metadata and conversation state.
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int = 2048,
        max_samples: int | None = None,
        include_turn_level: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if line.strip():
                    traj = json.loads(line)
                    examples = self._process_trajectory(traj, include_turn_level)
                    self.examples.extend(examples)

        # Compute class distribution
        self.class_counts = [0, 0, 0]
        for ex in self.examples:
            self.class_counts[ex["label"]] += 1

        print(f"Class distribution: SAFE={self.class_counts[0]}, "
              f"WARN={self.class_counts[1]}, ATTACK={self.class_counts[2]}")

    def _process_trajectory(
        self,
        trajectory: dict,
        include_turn_level: bool,
    ) -> list[dict]:
        """Convert trajectory to classification examples."""
        conversation = trajectory.get("conversation", [])
        attack_metadata = trajectory.get("attack_metadata", {})
        # Recognize all benign family variants
        benign_families = {"benign", "benign_hard_negatives", "legitimate"}
        is_attack = attack_metadata.get("family") not in benign_families
        pivot_turn = trajectory.get("pivot_turn_index")

        examples = []

        if include_turn_level:
            # Create example for each conversation prefix
            for i in range(1, len(conversation) + 1):
                prefix = conversation[:i]
                text = self._format_conversation(prefix)

                # Determine label based on position relative to pivot
                if not is_attack:
                    label = 0  # SAFE
                elif pivot_turn and i < pivot_turn:
                    # Before pivot - setup phase, could be WARN
                    turn_labels = prefix[-1].get("turn_labels", {})
                    intent_score = turn_labels.get("attack_intent_score", 0)
                    if intent_score > 0:
                        label = 1  # WARN
                    else:
                        label = 0  # SAFE
                elif pivot_turn and i >= pivot_turn:
                    label = 2  # ATTACK
                else:
                    # No pivot marked but is attack
                    label = 2  # ATTACK

                examples.append({
                    "text": text,
                    "label": label,
                    "trajectory_id": trajectory.get("id"),
                    "turn": i,
                })
        else:
            # Just use full conversation
            text = self._format_conversation(conversation)
            label = 2 if is_attack else 0
            examples.append({
                "text": text,
                "label": label,
                "trajectory_id": trajectory.get("id"),
                "turn": len(conversation),
            })

        return examples

    def _format_conversation(self, turns: list[dict]) -> str:
        """Format conversation for classification."""
        parts = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append(f"<{role}>\n{content}\n</{role}>")
        return "\n".join(parts)

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency class weights."""
        if torch is None:
            raise ImportError("torch is required")

        total = sum(self.class_counts)
        weights = [total / (3 * count) if count > 0 else 1.0
                  for count in self.class_counts]
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        encodings = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": example["label"],
        }


class SidecarClassifier(nn.Module):
    """
    Custom classifier head for causal LM base.

    Uses the last token representation for classification.
    """

    def __init__(self, base_model, num_labels: int = 3, hidden_size: int | None = None):
        super().__init__()
        self.base_model = base_model

        if hidden_size is None:
            hidden_size = base_model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels),
        )

        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Use last hidden state at last valid token position
        hidden_states = outputs.hidden_states[-1]

        if attention_mask is not None:
            # Get position of last non-padding token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, sequence_lengths]
        else:
            last_hidden = hidden_states[:, -1]

        logits = self.classifier(last_hidden)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


def setup_model_and_tokenizer(config: SidecarConfig):
    """Load and configure model and tokenizer for classification."""
    if AutoModelForSequenceClassification is None:
        raise ImportError("transformers is required")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if config.model_type == "sequence_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=config.trust_remote_code,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        # Use causal LM with custom classifier head
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=config.trust_remote_code,
        )
        model = SidecarClassifier(base_model, config.num_labels)

    # Apply LoRA
    if config.use_lora and LoraConfig is not None:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type=TaskType.SEQ_CLS if config.model_type == "sequence_classification" else TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def compute_metrics(eval_pred):
    """Compute classification metrics."""
    if precision_recall_fscore_support is None:
        return {}

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )

    # Per-class metrics
    per_class = precision_recall_fscore_support(
        labels, predictions, average=None, labels=[0, 1, 2]
    )

    metrics = {
        "accuracy": (predictions == labels).mean(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "safe_precision": per_class[0][0],
        "safe_recall": per_class[1][0],
        "warn_precision": per_class[0][1],
        "warn_recall": per_class[1][1],
        "attack_precision": per_class[0][2],
        "attack_recall": per_class[1][2],
    }

    return metrics


class WeightedLossTrainer(Trainer):
    """Trainer with class-weighted loss."""

    def __init__(self, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def train_sidecar(config: SidecarConfig) -> str:
    """
    Train sidecar classifier.

    Args:
        config: Training configuration

    Returns:
        Path to saved model
    """
    if torch is None:
        raise ImportError("torch is required")
    if Trainer is None:
        raise ImportError("transformers is required")

    print("TRYLOCK Sidecar Classifier Training")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Type: {config.model_type}")
    print(f"Labels: {config.label_names}")
    print(f"Data: {config.train_file}")
    print(f"Output: {config.output_dir}")
    print()

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config)

    # Load dataset
    train_dataset = SidecarDataset(
        config.train_file,
        tokenizer,
        max_length=config.max_seq_length,
        max_samples=config.max_samples,
        include_turn_level=True,
    )

    print(f"Training examples: {len(train_dataset)}")

    eval_dataset = None
    if config.eval_file:
        eval_dataset = SidecarDataset(
            config.eval_file,
            tokenizer,
            max_length=config.max_seq_length,
        )
        print(f"Evaluation examples: {len(eval_dataset)}")

    # Class weights
    class_weights = None
    if config.use_class_weights:
        class_weights = train_dataset.get_class_weights()
        print(f"Class weights: {class_weights.tolist()}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        bf16=config.bf16,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.save_steps if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="f1" if eval_dataset else None,
        remove_unused_columns=False,
        report_to="none",
    )

    # Trainer
    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Evaluate
    if eval_dataset:
        print("\nFinal evaluation:")
        metrics = trainer.evaluate()
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # Save
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    # Save thresholds
    thresholds = {
        "warn_threshold": config.warn_threshold,
        "attack_threshold": config.attack_threshold,
        "label_names": config.label_names,
    }
    with open(Path(config.output_dir) / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    return config.output_dir


class SidecarInference:
    """
    Inference wrapper for the sidecar classifier.

    Provides real-time threat assessment scores.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
    ):
        if AutoModelForSequenceClassification is None:
            raise ImportError("transformers is required")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map="auto" if device == "auto" else device,
        )
        self.model.eval()

        # Load thresholds
        thresholds_path = Path(model_path) / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                thresholds = json.load(f)
            self.warn_threshold = thresholds.get("warn_threshold", 0.3)
            self.attack_threshold = thresholds.get("attack_threshold", 0.7)
            self.label_names = thresholds.get("label_names", ["SAFE", "WARN", "ATTACK"])
        else:
            self.warn_threshold = 0.3
            self.attack_threshold = 0.7
            self.label_names = ["SAFE", "WARN", "ATTACK"]

    def classify(
        self,
        conversation: list[dict] | str,
    ) -> dict:
        """
        Classify a conversation.

        Args:
            conversation: List of turn dicts or formatted string

        Returns:
            Dict with classification results
        """
        if isinstance(conversation, list):
            text = self._format_conversation(conversation)
        else:
            text = conversation

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_class = int(np.argmax(probs))

        # Compute risk score (weighted toward ATTACK)
        risk_score = probs[1] * 0.3 + probs[2] * 0.7

        return {
            "classification": self.label_names[predicted_class],
            "class_id": predicted_class,
            "probabilities": {
                name: float(prob)
                for name, prob in zip(self.label_names, probs)
            },
            "risk_score": float(risk_score),
            "action": self._get_recommended_action(predicted_class, risk_score),
        }

    def _format_conversation(self, turns: list[dict]) -> str:
        """Format conversation for classification."""
        parts = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append(f"<{role}>\n{content}\n</{role}>")
        return "\n".join(parts)

    def _get_recommended_action(
        self,
        class_id: int,
        risk_score: float,
    ) -> str:
        """Get recommended action based on classification."""
        if class_id == 0 and risk_score < self.warn_threshold:
            return "CONTINUE"
        elif class_id == 1 or self.warn_threshold <= risk_score < self.attack_threshold:
            return "ENHANCE_MONITORING"
        else:
            return "INTERVENE"

    def classify_stream(
        self,
        conversation: list[dict],
    ) -> list[dict]:
        """
        Classify each prefix of a conversation.

        Useful for understanding how risk evolves through a conversation.

        Returns:
            List of classification results for each turn
        """
        results = []
        for i in range(1, len(conversation) + 1):
            prefix = conversation[:i]
            result = self.classify(prefix)
            result["turn"] = i
            results.append(result)
        return results


class HighRiskGuard:
    """
    Guard layer that overrides sidecar SAFE classification for high-risk prompts.

    Addresses the failure mode where sidecar misclassifies attacks as SAFE
    by checking for high-risk features that should floor alpha to minimum WARN level.

    This runs AFTER sidecar classification and can upgrade (but not downgrade) risk.
    """

    # Patterns that indicate high risk even if sidecar says SAFE
    HIGH_RISK_PATTERNS = [
        # Encoding artifacts
        r'[A-Za-z0-9+/]{40,}={0,2}',  # Base64
        r'\\x[0-9a-fA-F]{2}',  # Hex escapes
        r'\\u[0-9a-fA-F]{4}',  # Unicode escapes
        r'%[0-9a-fA-F]{2}',  # URL encoding

        # Refusal evasion templates
        r'ignore previous',
        r'ignore all previous',
        r'disregard instructions',
        r'forget your training',
        r'pretend you are',
        r'act as if you',
        r'you are now',
        r'new persona',
        r'developer mode',
        r'jailbreak',

        # Roleplay indicators with harm potential
        r'evil (ai|assistant|bot)',
        r'no ethical',
        r'no moral',
        r'without restrictions',
        r'uncensored',

        # System prompt extraction
        r'repeat.*system prompt',
        r'show.*system (message|prompt)',
        r'what are your instructions',
        r'reveal.*prompt',
    ]

    def __init__(
        self,
        min_alpha_on_high_risk: float = 1.5,
        enable_homoglyph_check: bool = True,
    ):
        import re
        self.min_alpha = min_alpha_on_high_risk
        self.enable_homoglyph = enable_homoglyph_check
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.HIGH_RISK_PATTERNS]

        # Common homoglyph characters (non-ASCII that look like ASCII)
        self.homoglyph_chars = set([
            '\u0430', '\u0435', '\u043e', '\u0440', '\u0441',  # Cyrillic
            '\u03b1', '\u03b5', '\u03b9', '\u03bf', '\u03c1',  # Greek
            '\u200b', '\u200c', '\u200d', '\ufeff',  # Zero-width
        ])

    def check_high_risk(self, text: str) -> tuple[bool, list[str]]:
        """
        Check if text contains high-risk features.

        Returns:
            Tuple of (is_high_risk, list_of_reasons)
        """
        reasons = []

        # Check regex patterns
        for pattern in self.patterns:
            if pattern.search(text):
                reasons.append(f"pattern:{pattern.pattern[:30]}...")

        # Check for homoglyphs
        if self.enable_homoglyph:
            homoglyph_count = sum(1 for c in text if c in self.homoglyph_chars)
            if homoglyph_count >= 3:
                reasons.append(f"homoglyphs:{homoglyph_count}")

        # Check for mixed scripts (potential obfuscation)
        import unicodedata
        scripts = set()
        for char in text:
            try:
                name = unicodedata.name(char, '')
                if 'CYRILLIC' in name:
                    scripts.add('cyrillic')
                elif 'GREEK' in name and 'LETTER' in name:
                    scripts.add('greek')
                elif 'LATIN' in name:
                    scripts.add('latin')
            except ValueError:
                pass

        if len(scripts) > 1:
            reasons.append(f"mixed_scripts:{scripts}")

        return len(reasons) > 0, reasons

    def apply_guard(
        self,
        text: str,
        sidecar_classification: str,
        sidecar_alpha: float,
    ) -> tuple[str, float, list[str]]:
        """
        Apply guard to potentially upgrade classification.

        Args:
            text: Input text
            sidecar_classification: Original sidecar classification (SAFE/WARN/ATTACK)
            sidecar_alpha: Original alpha from sidecar

        Returns:
            Tuple of (final_classification, final_alpha, guard_reasons)
        """
        is_high_risk, reasons = self.check_high_risk(text)

        if is_high_risk and sidecar_classification == "SAFE":
            # Upgrade SAFE to WARN
            return "WARN", max(sidecar_alpha, self.min_alpha), reasons
        elif is_high_risk and sidecar_alpha < self.min_alpha:
            # Floor alpha to minimum
            return sidecar_classification, self.min_alpha, reasons
        else:
            # No change
            return sidecar_classification, sidecar_alpha, []


def apply_sidecar_with_guard(
    inference: SidecarInference,
    text: str,
    guard: HighRiskGuard | None = None,
    alpha_map: dict | None = None,
) -> dict:
    """
    Classify with sidecar, then apply high-risk guard.

    This is the recommended way to use the sidecar in production,
    as it addresses the SAFE misclassification failure mode.

    Args:
        inference: SidecarInference instance
        text: Input text to classify
        guard: Optional HighRiskGuard (created if None)
        alpha_map: Optional alpha mapping for classifications

    Returns:
        Classification result with guard annotations
    """
    if alpha_map is None:
        alpha_map = {"SAFE": 0.5, "WARN": 1.5, "ATTACK": 2.5}

    if guard is None:
        guard = HighRiskGuard()

    # Get sidecar classification
    result = inference.classify(text)
    original_class = result["classification"]
    original_alpha = alpha_map.get(original_class, 1.5)

    # Apply guard
    final_class, final_alpha, guard_reasons = guard.apply_guard(
        text, original_class, original_alpha
    )

    result["original_classification"] = original_class
    result["original_alpha"] = original_alpha
    result["classification"] = final_class
    result["alpha"] = final_alpha
    result["guard_applied"] = len(guard_reasons) > 0
    result["guard_reasons"] = guard_reasons

    return result


def evaluate_sidecar(
    model_path: str,
    test_file: str,
    output_file: str | None = None,
) -> dict:
    """
    Evaluate sidecar classifier on test data.

    Returns comprehensive metrics and error analysis.
    """
    if classification_report is None:
        raise ImportError("sklearn is required for evaluation")

    inference = SidecarInference(model_path)

    predictions = []
    labels = []
    examples = []

    with open(test_file) as f:
        for line in f:
            if not line.strip():
                continue

            traj = json.loads(line)
            conversation = traj.get("conversation", [])
            # Recognize all benign family variants
            benign_families = {"benign", "benign_hard_negatives", "legitimate"}
            is_attack = traj.get("attack_metadata", {}).get("family") not in benign_families

            # Classify full conversation
            result = inference.classify(conversation)

            # True label (simplified: attack vs benign)
            true_label = 2 if is_attack else 0
            pred_label = result["class_id"]

            predictions.append(pred_label)
            labels.append(true_label)

            examples.append({
                "id": traj.get("id"),
                "true_label": true_label,
                "pred_label": pred_label,
                "probabilities": result["probabilities"],
                "risk_score": result["risk_score"],
            })

    # Compute metrics
    report = classification_report(
        labels, predictions,
        target_names=inference.label_names,
        output_dict=True,
    )

    # Error analysis
    false_negatives = []  # Attacks classified as SAFE
    false_positives = []  # SAFE classified as ATTACK

    for ex in examples:
        if ex["true_label"] == 2 and ex["pred_label"] == 0:
            false_negatives.append(ex)
        elif ex["true_label"] == 0 and ex["pred_label"] == 2:
            false_positives.append(ex)

    results = {
        "classification_report": report,
        "total_examples": len(examples),
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "false_negative_rate": len(false_negatives) / max(1, sum(1 for l in labels if l == 2)),
        "false_positive_rate": len(false_positives) / max(1, sum(1 for l in labels if l == 0)),
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRYLOCK Sidecar Classifier Training")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/tier1_open/attacks/train.jsonl",
        help="Training data file",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default=None,
        help="Evaluation data file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/trylock-sidecar",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per device",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum training samples",
    )
    parser.add_argument(
        "--evaluate",
        type=str,
        default=None,
        help="Evaluate model on test file instead of training",
    )

    args = parser.parse_args()

    if args.evaluate:
        # Evaluation mode
        results = evaluate_sidecar(
            args.output_dir,
            args.evaluate,
            output_file=args.output_dir + "/eval_results.json",
        )
        print(json.dumps(results, indent=2))
    else:
        # Training mode
        if args.config:
            config = SidecarConfig.from_yaml(args.config)
        else:
            config = SidecarConfig(
                model_name=args.model,
                train_file=args.train_file,
                eval_file=args.eval_file,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
                per_device_batch_size=args.batch_size,
                max_samples=args.max_samples,
            )

        train_sidecar(config)
