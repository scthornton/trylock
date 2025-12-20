"""
TRYLOCK Evaluation Framework

Comprehensive evaluation for LLM security defenses:
- Attack Success Rate (ASR) measurement
- Over-refusal detection
- Quality metrics
- LLM-based and rule-based judges
"""

from .harness import (
    EvalConfig,
    EvalResult,
    AggregateResults,
    TRYLOCKHarness,
    run_evaluation,
)

from .metrics import (
    ConfidenceInterval,
    wilson_score_interval,
    attack_success_rate,
    over_refusal_rate,
    defense_effectiveness_score,
    pareto_optimal,
    asr_by_dimension,
    mcnemar_test,
    bootstrap_confidence_interval,
    MetricsReport,
    compute_asr_targets,
)

from .judges import (
    JudgmentResult,
    BaseJudge,
    LLMJudge,
    RuleBasedJudge,
    EnsembleJudge,
    BatchJudge,
    create_judge,
    quick_judge,
)

__all__ = [
    # Harness
    "EvalConfig",
    "EvalResult",
    "AggregateResults",
    "TRYLOCKHarness",
    "run_evaluation",
    # Metrics
    "ConfidenceInterval",
    "wilson_score_interval",
    "attack_success_rate",
    "over_refusal_rate",
    "defense_effectiveness_score",
    "pareto_optimal",
    "asr_by_dimension",
    "mcnemar_test",
    "bootstrap_confidence_interval",
    "MetricsReport",
    "compute_asr_targets",
    # Judges
    "JudgmentResult",
    "BaseJudge",
    "LLMJudge",
    "RuleBasedJudge",
    "EnsembleJudge",
    "BatchJudge",
    "create_judge",
    "quick_judge",
]
