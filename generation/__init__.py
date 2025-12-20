"""
TRYLOCK Data Generation Pipeline

This module provides tools for generating adversarial attack trajectories
for training LLM security defenses.

Components:
- RedBot: Generates attack conversations
- VictimBot: Simulates target models
- JudgeBot: Labels and generates responses
- MutationEngine: Creates attack variants
- ActivationCapture: Extracts pivot activations for RepE
- TRYLOCKPipeline: Orchestrates the full generation flow
"""

from .red_bot import RedBot, AttackConfig, TaxonomyLoader, GeneratedTrajectory
from .victim_bot import VictimBot, MockVictimBot, VictimResponse
from .judge_bot import JudgeBot, MockJudgeBot, JudgmentResult, convert_judgment_to_trajectory
from .mutation_engine import MutationEngine, SimpleMutationEngine, MutationResult
from .activation_capture import (
    ActivationCapture,
    MockActivationCapture,
    CaptureConfig,
    CapturedActivations,
    process_trajectory_with_activations,
    batch_capture_activations,
)
from .pipeline import (
    TRYLOCKPipeline,
    MockPipeline,
    PipelineConfig,
    GenerationStats,
    generate_benign_hard_negatives,
)

__all__ = [
    # Red Bot
    "RedBot",
    "AttackConfig",
    "TaxonomyLoader",
    "GeneratedTrajectory",
    # Victim Bot
    "VictimBot",
    "MockVictimBot",
    "VictimResponse",
    # Judge Bot
    "JudgeBot",
    "MockJudgeBot",
    "JudgmentResult",
    "convert_judgment_to_trajectory",
    # Mutation Engine
    "MutationEngine",
    "SimpleMutationEngine",
    "MutationResult",
    # Activation Capture
    "ActivationCapture",
    "MockActivationCapture",
    "CaptureConfig",
    "CapturedActivations",
    "process_trajectory_with_activations",
    "batch_capture_activations",
    # Pipeline
    "TRYLOCKPipeline",
    "MockPipeline",
    "PipelineConfig",
    "GenerationStats",
    "generate_benign_hard_negatives",
]
