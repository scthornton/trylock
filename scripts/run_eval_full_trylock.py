#!/usr/bin/env python3
"""
Full TRYLOCK Evaluation Script (Layer 1 + 2 + 3)

Evaluates the complete TRYLOCK system with adaptive α selection:
- Layer 1: DPO-trained model (KNOWLEDGE)
- Layer 2: RepE steering (INSTINCT)
- Layer 3: Sidecar classifier (OVERSIGHT) selecting α ∈ {0.5, 1.5, 2.5}

This script measures the performance claimed in Table 2 of the paper.

Usage:
    python scripts/run_eval_full_trylock.py \
        --test-file data/dpo/test.jsonl \
        --dpo-adapter outputs/trylock-mistral-7b \
        --repe-vectors outputs/repe/steering_vectors.safetensors \
        --sidecar outputs/trylock-sidecar \
        --output eval_full_trylock.json

Output:
    - Full TRYLOCK ASR and ORR for Table 2
    - Detailed per-sample results
    - Comparison to baseline
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from safetensors.torch import load_file
from tqdm import tqdm

# Import judges from existing evaluation infrastructure
sys.path.append(str(Path(__file__).parent.parent))
from scripts.run_eval import RuleBasedJudge, LLMJudge, EnsembleJudge, wilson_score_interval


# ============================================================================
# Full TRYLOCK Model Integration
# ============================================================================

class FullTRYLOCKModel:
    """
    Complete TRYLOCK defense system with all three layers.
    """

    def __init__(
        self,
        base_model_name: str,
        dpo_adapter_path: str,
        repe_vectors_path: str,
        sidecar_path: str,
        steering_layers: list[int] = None,
        alpha_map: dict = None,
        device: str = "auto",
    ):
        self.base_model_name = base_model_name
        self.dpo_adapter_path = dpo_adapter_path
        self.repe_vectors_path = repe_vectors_path
        self.sidecar_path = sidecar_path
        self.steering_layers = steering_layers or [12, 14, 16, 18, 20, 22, 24, 26]
        self.alpha_map = alpha_map or {"SAFE": 0.5, "WARN": 1.5, "ATTACK": 2.5}
        self.device = device

        self.tokenizer = None
        self.main_model = None
        self.sidecar_model = None
        self.sidecar_tokenizer = None
        self.steering_vectors = None
        self.hooks = []

    def setup(self):
        """Initialize all three layers."""
        print("\n" + "=" * 60)
        print("SETTING UP FULL TRYLOCK (Layer 1 + 2 + 3)")
        print("=" * 60)

        # Layer 1: Load DPO-trained model
        print("\n[Layer 1] Loading DPO-trained model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )

        # Load DPO adapter
        print(f"   Loading DPO adapter from {self.dpo_adapter_path}...")
        self.main_model = PeftModel.from_pretrained(
            base_model,
            self.dpo_adapter_path,
        )
        self.main_model.eval()
        print(f"   ✓ Layer 1 ready (DPO model on {self.main_model.device})")

        # Layer 2: Load RepE steering vectors
        print("\n[Layer 2] Loading RepE steering vectors...")
        self.steering_vectors = load_file(self.repe_vectors_path)
        print(f"   ✓ Layer 2 ready ({len(self.steering_vectors)} vectors loaded)")

        # Layer 3: Load sidecar classifier
        print("\n[Layer 3] Loading sidecar classifier...")
        sidecar_path = Path(self.sidecar_path)

        # Check if PEFT adapter
        adapter_config = sidecar_path / "adapter_config.json"
        if adapter_config.exists():
            with open(adapter_config) as f:
                config = json.load(f)
            base_model_name = config.get("base_model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")

            # Load base + adapter
            self.sidecar_tokenizer = AutoTokenizer.from_pretrained(str(sidecar_path))
            if self.sidecar_tokenizer.pad_token is None:
                self.sidecar_tokenizer.pad_token = self.sidecar_tokenizer.eos_token

            sidecar_base = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=3,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.sidecar_model = PeftModel.from_pretrained(
                sidecar_base,
                str(sidecar_path),
            )
        else:
            # Load full model
            self.sidecar_tokenizer = AutoTokenizer.from_pretrained(str(sidecar_path))
            self.sidecar_model = AutoModelForSequenceClassification.from_pretrained(
                str(sidecar_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        self.sidecar_model.eval()
        print(f"   ✓ Layer 3 ready (sidecar classifier)")

        print("\n" + "=" * 60)
        print("ALL LAYERS INITIALIZED")
        print("=" * 60)

    def classify_threat(self, messages: list[dict]) -> tuple[str, float]:
        """
        Layer 3: Classify threat level using sidecar.

        Returns:
            (classification, confidence) where classification ∈ {SAFE, WARN, ATTACK}
        """
        # Format conversation for classification
        conversation_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in messages
        ])

        inputs = self.sidecar_tokenizer(
            conversation_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.sidecar_model.device)

        with torch.no_grad():
            outputs = self.sidecar_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().float()

        labels = ["SAFE", "WARN", "ATTACK"]
        predicted_idx = probs.argmax().item()
        classification = labels[predicted_idx]
        confidence = probs[predicted_idx].item()

        return classification, confidence

    def apply_steering(self, alpha: float):
        """
        Layer 2: Apply RepE steering hooks at given alpha.
        """
        self.remove_hooks()

        if alpha == 0:
            return

        def make_hook(layer_idx: int):
            key = f"layer_{layer_idx}"
            if key not in self.steering_vectors:
                return None

            vector = self.steering_vectors[key]

            def hook(module, input, output):
                hidden_states = output[0]
                device = hidden_states.device
                dtype = hidden_states.dtype
                steering = vector.to(device).to(dtype)

                # Apply steering: h' = h - α * v
                hidden_states = hidden_states - alpha * steering
                return (hidden_states,) + output[1:]

            return hook

        # Get layers from PEFT model
        try:
            layers = self.main_model.base_model.model.model.layers
        except AttributeError:
            layers = self.main_model.model.layers

        for layer_idx in self.steering_layers:
            hook_fn = make_hook(layer_idx)
            if hook_fn and layer_idx < len(layers):
                h = layers[layer_idx].register_forward_hook(hook_fn)
                self.hooks.append(h)

    def remove_hooks(self):
        """Remove all steering hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        use_adaptive: bool = True,
    ) -> tuple[str, str, float]:
        """
        Generate response with full TRYLOCK protection.

        Returns:
            (response, classification, applied_alpha)
        """
        # Layer 3: Classify threat
        if use_adaptive:
            classification, confidence = self.classify_threat(messages)
            alpha = self.alpha_map[classification]
        else:
            classification = "DISABLED"
            alpha = 0.0

        # Layer 2: Apply steering
        self.apply_steering(alpha)

        # Layer 1: Generate with DPO model
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.main_model.device)

        with torch.no_grad():
            outputs = self.main_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Cleanup
        self.remove_hooks()

        return response.strip(), classification, alpha

    def cleanup(self):
        """Free GPU memory."""
        self.remove_hooks()
        del self.main_model
        del self.sidecar_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# Evaluation Logic
# ============================================================================

@dataclass
class TRYLOCKResult:
    """Result for single sample."""
    id: str
    attack_family: str
    attack_type: str

    # TRYLOCK outputs
    response: str = ""
    classification: str = ""
    alpha_used: float = 0.0

    # Judgment
    complied: bool = False
    refused: bool = False
    judgment: dict = field(default_factory=dict)

    # Timing
    generation_time: float = 0.0
    error: Optional[str] = None


def run_full_trylock_evaluation(
    test_file: str,
    trylock_model: FullTRYLOCKModel,
    judge,
    max_samples: Optional[int] = None,
) -> tuple[list[TRYLOCKResult], dict]:
    """
    Run evaluation on full TRYLOCK system.
    """
    # Load test data
    test_data = []
    with open(test_file) as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"\nEvaluating {len(test_data)} samples with full TRYLOCK...")
    print("-" * 60)

    results = []
    attack_count = 0
    benign_count = 0

    for item in tqdm(test_data, desc="Full TRYLOCK Evaluation"):
        metadata = item.get("metadata", {})

        result = TRYLOCKResult(
            id=item.get("id", "unknown"),
            attack_family=metadata.get("family", "unknown"),
            attack_type=metadata.get("type", "unknown"),
        )

        try:
            messages = item.get("prompt", [])
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")

            # Generate with full TRYLOCK
            start = time.time()
            response, classification, alpha = trylock_model.generate(
                messages,
                use_adaptive=True,
            )
            elapsed = time.time() - start

            result.response = response
            result.classification = classification
            result.alpha_used = alpha
            result.generation_time = elapsed

            # Judge response
            if isinstance(judge, RuleBasedJudge):
                judgment = judge.judge(response, messages, chosen, rejected)
            else:
                judgment = judge.judge(
                    response, messages, chosen, rejected,
                    result.attack_family, result.attack_type,
                )

            result.judgment = judgment
            result.complied = judgment.get("complied", False)
            result.refused = judgment.get("refused", False)

            # Track attack vs benign
            is_benign = metadata.get("is_benign", False) or result.attack_family == "benign"
            if is_benign:
                benign_count += 1
            else:
                attack_count += 1

        except Exception as e:
            result.error = str(e)

        results.append(result)

    # Compute metrics
    valid = [r for r in results if not r.error]
    n_valid = len(valid)

    # Separate attack and benign samples
    attack_results = [r for r in valid if r.attack_family != "benign"]
    benign_results = [r for r in valid if r.attack_family == "benign"]

    # ASR (on attack samples only)
    asr_successes = sum(1 for r in attack_results if r.complied)
    asr = asr_successes / len(attack_results) if attack_results else 0.0
    asr_ci = wilson_score_interval(asr_successes, len(attack_results))

    # ORR (over-refusal on benign samples)
    orr_count = sum(1 for r in benign_results if r.refused)
    orr = orr_count / len(benign_results) if benign_results else 0.0
    orr_ci = wilson_score_interval(orr_count, len(benign_results))

    # Alpha distribution
    alpha_distribution = {}
    for r in valid:
        alpha_distribution[r.classification] = alpha_distribution.get(r.classification, 0) + 1

    metrics = {
        "total_samples": len(test_data),
        "valid_samples": n_valid,
        "errors": len(test_data) - n_valid,
        "attack_samples": len(attack_results),
        "benign_samples": len(benign_results),
        "asr": asr,
        "asr_ci": asr_ci,
        "asr_successes": asr_successes,
        "orr": orr,
        "orr_ci": orr_ci,
        "orr_count": orr_count,
        "alpha_distribution": alpha_distribution,
        "avg_generation_time": sum(r.generation_time for r in valid) / n_valid if n_valid else 0,
    }

    return results, metrics


def print_report(metrics: dict):
    """Print evaluation report."""
    print("\n" + "=" * 70)
    print("FULL TRYLOCK EVALUATION REPORT (Layer 1 + 2 + 3)")
    print("=" * 70)

    print(f"\nSamples:")
    print(f"  Total: {metrics['total_samples']}")
    print(f"  Valid: {metrics['valid_samples']}")
    print(f"  Errors: {metrics['errors']}")
    print(f"  Attack prompts: {metrics['attack_samples']}")
    print(f"  Benign prompts: {metrics['benign_samples']}")

    print(f"\n{'─' * 70}")
    print("PRIMARY METRICS (FOR TABLE 2)")
    print(f"{'─' * 70}")

    asr = metrics['asr']
    asr_ci = metrics['asr_ci']
    orr = metrics['orr']
    orr_ci = metrics['orr_ci']

    print(f"\nAttack Success Rate (ASR):")
    print(f"  Point estimate: {asr:.1%}")
    print(f"  95% CI: [{asr_ci['lower']:.1%}, {asr_ci['upper']:.1%}]")
    print(f"  Attacks succeeded: {metrics['asr_successes']}/{metrics['attack_samples']}")

    print(f"\nOver-Refusal Rate (ORR):")
    print(f"  Point estimate: {orr:.1%}")
    print(f"  95% CI: [{orr_ci['lower']:.1%}, {orr_ci['upper']:.1%}]")
    print(f"  Benign refused: {metrics['orr_count']}/{metrics['benign_samples']}")

    print(f"\nAdaptive Alpha Distribution:")
    for classification, count in sorted(metrics['alpha_distribution'].items()):
        pct = count / metrics['valid_samples'] * 100
        print(f"  {classification}: {count} ({pct:.1f}%)")

    print(f"\nAverage generation time: {metrics['avg_generation_time']:.2f}s")

    print("\n" + "=" * 70)
    print("TABLE 2 ENTRY")
    print("=" * 70)
    print(f"\nFull TRYLOCK (adaptive α)  |  {asr:.1%}  |  {orr:.1%}  |  Adaptive")
    print("\nUse these numbers for the paper!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Full TRYLOCK Evaluation (Layer 1 + 2 + 3)",
    )

    parser.add_argument(
        "--test-file", type=str,
        default="data/dpo/test.jsonl",
        help="Test data file (DPO format)",
    )
    parser.add_argument(
        "--base-model", type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Base model name",
    )
    parser.add_argument(
        "--dpo-adapter", type=str,
        required=True,
        help="Path to DPO adapter (Layer 1)",
    )
    parser.add_argument(
        "--repe-vectors", type=str,
        required=True,
        help="Path to RepE steering vectors (Layer 2)",
    )
    parser.add_argument(
        "--sidecar", type=str,
        required=True,
        help="Path to sidecar classifier (Layer 3)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum samples (None = all)",
    )
    parser.add_argument(
        "--judge", type=str,
        choices=["rule", "llm", "ensemble"],
        default="ensemble",
        help="Judge type",
    )
    parser.add_argument(
        "--output", type=str,
        default="eval_full_trylock.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FULL TRYLOCK EVALUATION")
    print("=" * 70)

    # Initialize full TRYLOCK model
    trylock = FullTRYLOCKModel(
        base_model_name=args.base_model,
        dpo_adapter_path=args.dpo_adapter,
        repe_vectors_path=args.repe_vectors,
        sidecar_path=args.sidecar,
    )
    trylock.setup()

    # Setup judge
    print(f"\nSetting up {args.judge} judge...")
    if args.judge == "rule":
        judge = RuleBasedJudge()
    elif args.judge == "llm":
        judge = LLMJudge()
    else:
        judge = EnsembleJudge()

    # Run evaluation
    results, metrics = run_full_trylock_evaluation(
        test_file=args.test_file,
        trylock_model=trylock,
        judge=judge,
        max_samples=args.max_samples,
    )

    # Print report
    print_report(metrics)

    # Save results
    output_data = {
        "config": {
            "base_model": args.base_model,
            "dpo_adapter": args.dpo_adapter,
            "repe_vectors": args.repe_vectors,
            "sidecar": args.sidecar,
            "test_file": args.test_file,
            "judge_type": args.judge,
        },
        "metrics": metrics,
        "detailed_results": [
            {
                "id": r.id,
                "family": r.attack_family,
                "type": r.attack_type,
                "classification": r.classification,
                "alpha_used": r.alpha_used,
                "complied": r.complied,
                "refused": r.refused,
                "response": r.response[:1000],
                "error": r.error,
            }
            for r in results
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {args.output}")

    # Cleanup
    trylock.cleanup()


if __name__ == "__main__":
    main()
