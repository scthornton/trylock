"""
TRYLOCK Training Pipeline

Multi-stage training for security-aware LLMs:
1. SFT Warmup - Establish baseline security-aware behavior
2. DPO Preference - Teach preference for secure responses
3. RepE Training - Compute control vectors for steering
4. Sidecar Classifier - Train parallel security classifier
"""

from .sft_warmup import (
    SFTConfig,
    TRYLOCKSFTDataset,
    SFTDataCollator,
    run_sft_warmup,
    setup_model_and_tokenizer as setup_sft_model,
    prepare_mixed_dataset,
)

from .dpo_preference import (
    DPOTrainingConfig,
    TRYLOCKDPODataset,
    CurriculumDPODataset,
    run_dpo_training,
    compute_dpo_metrics,
)

from .repe_training import (
    RepEConfig,
    ActivationExtractor,
    ControlVector,
    ControlVectorSet,
    train_repe_vectors,
    RepEInferenceWrapper,
    compute_control_vectors_pca,
    compute_control_vectors_mean_diff,
    compute_control_vectors_logistic,
)

from .sidecar_classifier import (
    SidecarConfig,
    SidecarDataset,
    SidecarClassifier,
    train_sidecar,
    SidecarInference,
    evaluate_sidecar,
)

__all__ = [
    # SFT
    "SFTConfig",
    "TRYLOCKSFTDataset",
    "SFTDataCollator",
    "run_sft_warmup",
    "setup_sft_model",
    "prepare_mixed_dataset",
    # DPO
    "DPOTrainingConfig",
    "TRYLOCKDPODataset",
    "CurriculumDPODataset",
    "run_dpo_training",
    "compute_dpo_metrics",
    # RepE
    "RepEConfig",
    "ActivationExtractor",
    "ControlVector",
    "ControlVectorSet",
    "train_repe_vectors",
    "RepEInferenceWrapper",
    "compute_control_vectors_pca",
    "compute_control_vectors_mean_diff",
    "compute_control_vectors_logistic",
    # Sidecar
    "SidecarConfig",
    "SidecarDataset",
    "SidecarClassifier",
    "train_sidecar",
    "SidecarInference",
    "evaluate_sidecar",
]
