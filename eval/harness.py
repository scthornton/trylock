"""
TRYLOCK Evaluation Harness

Comprehensive evaluation framework for assessing:
1. Attack Success Rate (ASR) across attack families
2. Over-refusal rate on benign requests
3. Capability preservation on general tasks
4. Response quality and coherence

Usage:
    python -m eval.harness --model ./outputs/trylock-dpo --test-file data/eval.jsonl

    # Or programmatically:
    from eval.harness import TRYLOCKHarness
    harness = TRYLOCKHarness(model_path)
    results = harness.evaluate(test_data)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


@dataclass
class EvalConfig:
    """Configuration for evaluation harness."""

    # Model
    model_path: str = "./outputs/trylock-dpo"
    model_name: str | None = None  # For baseline comparison
    trust_remote_code: bool = True

    # RepE (optional)
    repe_vectors_path: str | None = None
    repe_alpha: float = 1.0

    # Sidecar (optional)
    sidecar_model_path: str | None = None

    # Test data
    test_file: str = "data/tier1_open/attacks/eval.jsonl"
    benign_file: str | None = "data/tier1_open/benign/eval.jsonl"
    capability_file: str | None = None  # General capability benchmark

    # Evaluation settings
    max_samples: int | None = None
    batch_size: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.0  # Deterministic for eval

    # Judge configuration
    use_llm_judge: bool = True
    judge_model: str = "claude-sonnet-4-20250514"
    judge_concurrency: int = 5

    # Output
    output_file: str = "eval_results.json"
    detailed_output: bool = True

    # Hardware
    bf16: bool = True
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("eval", data))


@dataclass
class EvalResult:
    """Result for a single evaluation example."""

    trajectory_id: str
    attack_family: str | None
    attack_type: str | None
    difficulty: str | None
    model_response: str
    is_attack: bool

    # Judgments
    attack_successful: bool | None = None
    was_refusal: bool | None = None
    is_overblock: bool | None = None
    response_quality: float | None = None

    # Scores
    sidecar_classification: str | None = None
    sidecar_risk_score: float | None = None

    # Metadata
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class AggregateResults:
    """Aggregated evaluation metrics."""

    # Attack Success Rates (lower is better)
    overall_asr: float = 0.0
    asr_by_family: dict[str, float] = field(default_factory=dict)
    asr_by_difficulty: dict[str, float] = field(default_factory=dict)
    asr_by_type: dict[str, float] = field(default_factory=dict)

    # Over-refusal (lower is better)
    over_refusal_rate: float = 0.0
    over_refusal_by_category: dict[str, float] = field(default_factory=dict)

    # Quality metrics
    average_response_quality: float = 0.0
    refusal_rate_on_attacks: float = 0.0

    # Capability preservation
    capability_score: float | None = None

    # Sidecar metrics
    sidecar_accuracy: float | None = None
    sidecar_false_negative_rate: float | None = None
    sidecar_false_positive_rate: float | None = None

    # Summary
    total_attack_samples: int = 0
    total_benign_samples: int = 0
    total_errors: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "overall_asr": self.overall_asr,
            "asr_by_family": self.asr_by_family,
            "asr_by_difficulty": self.asr_by_difficulty,
            "asr_by_type": self.asr_by_type,
            "over_refusal_rate": self.over_refusal_rate,
            "over_refusal_by_category": self.over_refusal_by_category,
            "average_response_quality": self.average_response_quality,
            "refusal_rate_on_attacks": self.refusal_rate_on_attacks,
            "capability_score": self.capability_score,
            "sidecar_accuracy": self.sidecar_accuracy,
            "sidecar_false_negative_rate": self.sidecar_false_negative_rate,
            "sidecar_false_positive_rate": self.sidecar_false_positive_rate,
            "total_attack_samples": self.total_attack_samples,
            "total_benign_samples": self.total_benign_samples,
            "total_errors": self.total_errors,
        }


class TRYLOCKHarness:
    """
    Main evaluation harness for TRYLOCK models.

    Coordinates model loading, inference, judging, and metric computation.
    """

    def __init__(
        self,
        config: EvalConfig | None = None,
        model_path: str | None = None,
    ):
        if config is None:
            config = EvalConfig(model_path=model_path or "./outputs/trylock-dpo")
        self.config = config
        self.model = None
        self.tokenizer = None
        self.repe_wrapper = None
        self.sidecar = None
        self.judge = None

    def setup(self):
        """Initialize model and components."""
        if AutoModelForCausalLM is None:
            raise ImportError("transformers is required")

        print(f"Loading model from {self.config.model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="auto" if self.config.device == "auto" else self.config.device,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.eval()

        # Setup RepE if configured
        if self.config.repe_vectors_path:
            from training.repe_training import ControlVectorSet, RepEInferenceWrapper

            vectors = ControlVectorSet.load(self.config.repe_vectors_path)
            self.repe_wrapper = RepEInferenceWrapper(
                self.model, vectors, alpha=self.config.repe_alpha
            )
            self.repe_wrapper.enable()

        # Setup sidecar if configured
        if self.config.sidecar_model_path:
            from training.sidecar_classifier import SidecarInference

            self.sidecar = SidecarInference(self.config.sidecar_model_path)

        # Setup judge if configured
        if self.config.use_llm_judge:
            from eval.judges import LLMJudge

            self.judge = LLMJudge(model=self.config.judge_model)

        print("Setup complete")

    def generate_response(
        self,
        conversation: list[dict],
    ) -> tuple[str, float]:
        """
        Generate model response for a conversation.

        Returns:
            Tuple of (response_text, latency_ms)
        """
        # Format conversation
        messages = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in conversation
            if turn.get("role") in ["user", "assistant", "system"]
        ]

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
        ).to(self.model.device)

        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Decode response (exclude prompt)
        response_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return response_text, latency_ms

    def evaluate_single(
        self,
        trajectory: dict,
    ) -> EvalResult:
        """Evaluate a single trajectory."""
        trajectory_id = trajectory.get("id", "unknown")
        attack_metadata = trajectory.get("attack_metadata", {})
        conversation = trajectory.get("conversation", [])

        is_attack = attack_metadata.get("family") != "benign"

        result = EvalResult(
            trajectory_id=trajectory_id,
            attack_family=attack_metadata.get("family"),
            attack_type=attack_metadata.get("type"),
            difficulty=attack_metadata.get("difficulty"),
            model_response="",
            is_attack=is_attack,
        )

        try:
            # Generate response
            response, latency = self.generate_response(conversation)
            result.model_response = response
            result.latency_ms = latency

            # Sidecar classification
            if self.sidecar:
                sidecar_result = self.sidecar.classify(conversation)
                result.sidecar_classification = sidecar_result["classification"]
                result.sidecar_risk_score = sidecar_result["risk_score"]

            # LLM judge evaluation
            if self.judge:
                judgment = self.judge.judge_response(
                    conversation=conversation,
                    response=response,
                    is_attack=is_attack,
                    attack_family=attack_metadata.get("family"),
                )
                result.attack_successful = judgment.get("attack_successful")
                result.was_refusal = judgment.get("was_refusal")
                result.is_overblock = judgment.get("is_overblock")
                result.response_quality = judgment.get("quality_score")

        except Exception as e:
            result.error = str(e)

        return result

    def evaluate(
        self,
        test_file: str | None = None,
        progress_callback: Callable | None = None,
    ) -> tuple[list[EvalResult], AggregateResults]:
        """
        Run full evaluation.

        Returns:
            Tuple of (individual_results, aggregate_metrics)
        """
        test_file = test_file or self.config.test_file
        results = []

        # Load test data
        trajectories = []
        with open(test_file) as f:
            for line in f:
                if line.strip():
                    trajectories.append(json.loads(line))

        if self.config.max_samples:
            trajectories = trajectories[: self.config.max_samples]

        print(f"Evaluating {len(trajectories)} trajectories...")

        # Evaluate each trajectory
        for i, traj in enumerate(tqdm(trajectories, desc="Evaluating")):
            result = self.evaluate_single(traj)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(trajectories), result)

        # Compute aggregate metrics
        aggregate = self._compute_aggregate_metrics(results)

        return results, aggregate

    def evaluate_benign(
        self,
        benign_file: str | None = None,
    ) -> tuple[list[EvalResult], float]:
        """
        Evaluate over-refusal on benign examples.

        Returns:
            Tuple of (results, over_refusal_rate)
        """
        benign_file = benign_file or self.config.benign_file
        if not benign_file:
            return [], 0.0

        results = []

        with open(benign_file) as f:
            trajectories = [json.loads(line) for line in f if line.strip()]

        if self.config.max_samples:
            trajectories = trajectories[: self.config.max_samples]

        print(f"Evaluating {len(trajectories)} benign trajectories...")

        for traj in tqdm(trajectories, desc="Evaluating benign"):
            result = self.evaluate_single(traj)
            results.append(result)

        # Count over-refusals
        over_refusals = sum(1 for r in results if r.is_overblock)
        over_refusal_rate = over_refusals / len(results) if results else 0.0

        return results, over_refusal_rate

    def _compute_aggregate_metrics(
        self,
        results: list[EvalResult],
    ) -> AggregateResults:
        """Compute aggregate metrics from individual results."""
        aggregate = AggregateResults()

        # Split by attack/benign
        attack_results = [r for r in results if r.is_attack]
        benign_results = [r for r in results if not r.is_attack]

        aggregate.total_attack_samples = len(attack_results)
        aggregate.total_benign_samples = len(benign_results)
        aggregate.total_errors = sum(1 for r in results if r.error)

        # Overall ASR
        if attack_results:
            successful_attacks = sum(
                1 for r in attack_results if r.attack_successful
            )
            aggregate.overall_asr = successful_attacks / len(attack_results)

            # Refusal rate on attacks (higher is better)
            refusals = sum(1 for r in attack_results if r.was_refusal)
            aggregate.refusal_rate_on_attacks = refusals / len(attack_results)

        # ASR by family
        family_counts = {}
        family_successes = {}
        for r in attack_results:
            family = r.attack_family or "unknown"
            family_counts[family] = family_counts.get(family, 0) + 1
            if r.attack_successful:
                family_successes[family] = family_successes.get(family, 0) + 1

        for family in family_counts:
            aggregate.asr_by_family[family] = (
                family_successes.get(family, 0) / family_counts[family]
            )

        # ASR by difficulty
        diff_counts = {}
        diff_successes = {}
        for r in attack_results:
            diff = r.difficulty or "unknown"
            diff_counts[diff] = diff_counts.get(diff, 0) + 1
            if r.attack_successful:
                diff_successes[diff] = diff_successes.get(diff, 0) + 1

        for diff in diff_counts:
            aggregate.asr_by_difficulty[diff] = (
                diff_successes.get(diff, 0) / diff_counts[diff]
            )

        # ASR by type
        type_counts = {}
        type_successes = {}
        for r in attack_results:
            atype = r.attack_type or "unknown"
            type_counts[atype] = type_counts.get(atype, 0) + 1
            if r.attack_successful:
                type_successes[atype] = type_successes.get(atype, 0) + 1

        for atype in type_counts:
            aggregate.asr_by_type[atype] = (
                type_successes.get(atype, 0) / type_counts[atype]
            )

        # Over-refusal rate
        if benign_results:
            over_refusals = sum(1 for r in benign_results if r.is_overblock)
            aggregate.over_refusal_rate = over_refusals / len(benign_results)

        # Response quality
        quality_scores = [
            r.response_quality for r in results if r.response_quality is not None
        ]
        if quality_scores:
            aggregate.average_response_quality = sum(quality_scores) / len(quality_scores)

        # Sidecar metrics
        if any(r.sidecar_classification for r in results):
            sidecar_correct = 0
            sidecar_fn = 0  # Attacks classified as SAFE
            sidecar_fp = 0  # SAFE classified as ATTACK

            for r in results:
                if r.sidecar_classification:
                    predicted_attack = r.sidecar_classification == "ATTACK"
                    if predicted_attack == r.is_attack:
                        sidecar_correct += 1
                    elif r.is_attack and not predicted_attack:
                        sidecar_fn += 1
                    elif not r.is_attack and predicted_attack:
                        sidecar_fp += 1

            total_sidecar = sum(1 for r in results if r.sidecar_classification)
            aggregate.sidecar_accuracy = sidecar_correct / total_sidecar if total_sidecar else None
            aggregate.sidecar_false_negative_rate = (
                sidecar_fn / aggregate.total_attack_samples
                if aggregate.total_attack_samples
                else None
            )
            aggregate.sidecar_false_positive_rate = (
                sidecar_fp / aggregate.total_benign_samples
                if aggregate.total_benign_samples
                else None
            )

        return aggregate

    def save_results(
        self,
        results: list[EvalResult],
        aggregate: AggregateResults,
        output_file: str | None = None,
    ):
        """Save evaluation results to file."""
        output_file = output_file or self.config.output_file

        output = {
            "config": {
                "model_path": self.config.model_path,
                "test_file": self.config.test_file,
                "repe_alpha": self.config.repe_alpha if self.config.repe_vectors_path else None,
            },
            "aggregate": aggregate.to_dict(),
        }

        if self.config.detailed_output:
            output["results"] = [
                {
                    "trajectory_id": r.trajectory_id,
                    "attack_family": r.attack_family,
                    "attack_type": r.attack_type,
                    "difficulty": r.difficulty,
                    "is_attack": r.is_attack,
                    "attack_successful": r.attack_successful,
                    "was_refusal": r.was_refusal,
                    "is_overblock": r.is_overblock,
                    "response_quality": r.response_quality,
                    "sidecar_classification": r.sidecar_classification,
                    "sidecar_risk_score": r.sidecar_risk_score,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                    "model_response": r.model_response[:500] if r.model_response else None,
                }
                for r in results
            ]

        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Results saved to {output_file}")

    def cleanup(self):
        """Clean up resources."""
        if self.repe_wrapper:
            self.repe_wrapper.disable()


def run_evaluation(config: EvalConfig) -> AggregateResults:
    """
    Run complete evaluation pipeline.

    Args:
        config: Evaluation configuration

    Returns:
        Aggregate evaluation metrics
    """
    harness = TRYLOCKHarness(config)
    harness.setup()

    try:
        # Evaluate attacks
        results, aggregate = harness.evaluate()

        # Evaluate benign (over-refusal)
        if config.benign_file:
            benign_results, over_refusal = harness.evaluate_benign()
            results.extend(benign_results)
            aggregate.over_refusal_rate = over_refusal

        # Save results
        harness.save_results(results, aggregate)

        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Overall ASR: {aggregate.overall_asr:.2%}")
        print(f"Over-refusal Rate: {aggregate.over_refusal_rate:.2%}")
        print(f"Average Response Quality: {aggregate.average_response_quality:.2f}")
        print("\nASR by Family:")
        for family, asr in aggregate.asr_by_family.items():
            print(f"  {family}: {asr:.2%}")
        print("\nASR by Difficulty:")
        for diff, asr in aggregate.asr_by_difficulty.items():
            print(f"  {diff}: {asr:.2%}")

        return aggregate

    finally:
        harness.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRYLOCK Evaluation Harness")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./outputs/trylock-dpo",
        help="Model path",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/tier1_open/attacks/eval.jsonl",
        help="Test data file",
    )
    parser.add_argument(
        "--benign-file",
        type=str,
        default=None,
        help="Benign data file for over-refusal testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--repe-vectors",
        type=str,
        default=None,
        help="Path to RepE control vectors",
    )
    parser.add_argument(
        "--repe-alpha",
        type=float,
        default=1.0,
        help="RepE steering strength",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLM judge (faster but less accurate)",
    )

    args = parser.parse_args()

    if args.config:
        config = EvalConfig.from_yaml(args.config)
    else:
        config = EvalConfig(
            model_path=args.model,
            test_file=args.test_file,
            benign_file=args.benign_file,
            output_file=args.output,
            max_samples=args.max_samples,
            repe_vectors_path=args.repe_vectors,
            repe_alpha=args.repe_alpha,
            use_llm_judge=not args.no_judge,
        )

    run_evaluation(config)
