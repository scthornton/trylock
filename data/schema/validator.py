"""
TRYLOCK Trajectory Schema Validator

Validates trajectory records against the JSON schema and business logic rules.
Provides both JSON Schema validation and Pydantic model validation.

Usage:
    # Validate a single record
    validator = TrajectoryValidator()
    is_valid, errors = validator.validate(record)

    # Validate a JSONL file
    valid, invalid, errors = validator.validate_file("trajectories.jsonl")

    # CLI usage
    python validator.py trajectories.jsonl
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft7Validator
from pydantic import BaseModel, Field, field_validator, model_validator


# Enums for type safety
class AttackFamily(str, Enum):
    DIRECT_INJECTION = "direct_injection"
    INDIRECT_INJECTION = "indirect_injection"
    MULTI_TURN_MANIPULATION = "multi_turn_manipulation"
    OBFUSCATION_WRAPPERS = "obfuscation_wrappers"
    TOOL_AGENT_ABUSE = "tool_agent_abuse"
    BENIGN = "benign"


class AttackStage(str, Enum):
    RECONNAISSANCE = "reconnaissance"
    SETUP = "setup"
    PROBE = "probe"
    PIVOT = "pivot"
    PAYLOAD = "payload"
    RECOVERY = "recovery"
    EXTRACTION = "extraction"
    NONE = "none"


class ResponseType(str, Enum):
    ENGAGED = "engaged"
    ENGAGED_WITH_CONTEXT = "engaged_with_context"
    CLARIFICATION_REQUEST = "clarification_request"
    PARTIAL_WITH_SANITIZATION = "partial_with_sanitization"
    SOFT_DECLINE = "soft_decline"
    SOFT_DECLINE_WITH_REDIRECT = "soft_decline_with_redirect"
    HARD_DECLINE = "hard_decline"


class ExpectedAction(str, Enum):
    CONTINUE = "continue"
    WARN = "warn"
    ASK_CLARIFY = "ask_clarify"
    SOFT_DECLINE = "soft_decline"
    HARD_DECLINE = "hard_decline"
    SANDBOX_TOOLS = "sandbox_tools"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class Source(str, Enum):
    SYNTHETIC_REDBLUE = "synthetic_redblue"
    PUBLIC_BENCHMARK = "public_benchmark"
    MANUAL_CRAFT = "manual_craft"
    MUTATION = "mutation"
    ENTERPRISE_DERIVED = "enterprise_derived"


class Wrapper(str, Enum):
    PARAPHRASE_SEMANTIC = "paraphrase_semantic"
    TRANSLATION_ROUNDTRIP = "translation_roundtrip"
    ENCODING_SCHEMES = "encoding_schemes"
    ADVERSARIAL_POETRY = "adversarial_poetry"
    ROLEPLAY_FICTION = "roleplay_fiction"
    CODE_COMMENTS = "code_comments"
    PAYLOAD_SPLITTING = "payload_splitting"


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


# Pydantic models for validation
class TurnLabels(BaseModel):
    """Labels for a single conversation turn."""

    attack_stage: AttackStage = AttackStage.NONE
    attack_intent_score: int = Field(ge=0, le=3, default=0)
    manipulation_tactics: list[str] = Field(default_factory=list)
    expected_action: ExpectedAction = ExpectedAction.CONTINUE


class ToolCall(BaseModel):
    """Tool/function call details."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Tool execution result."""

    output: str
    is_poisoned: bool = False


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""

    turn: int = Field(ge=1)
    role: Role
    content: str = Field(min_length=1)
    turn_labels: TurnLabels | None = None
    is_pivot_point: bool = False
    response_type: ResponseType | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


class ChosenResponse(BaseModel):
    """The ideal response for DPO training."""

    content: str = Field(min_length=1)
    response_type: ResponseType
    security_reasoning: str | None = None


class RejectedResponse(BaseModel):
    """A rejected response for DPO training."""

    content: str = Field(min_length=1)
    failure_mode: str


class Responses(BaseModel):
    """Response options for preference learning."""

    chosen: ChosenResponse
    rejected_unsafe: RejectedResponse | None = None
    rejected_overblock: RejectedResponse | None = None


class AttackMetadata(BaseModel):
    """Metadata about the attack in this trajectory."""

    family: AttackFamily
    type: str = Field(min_length=1)
    wrappers: list[Wrapper] = Field(default_factory=list)
    manipulation_tactics: list[str] = Field(default_factory=list)
    difficulty: Difficulty
    source: Source
    academic_benchmark_alignment: list[str] = Field(default_factory=list)
    notes: str | None = None


class ActivationCapture(BaseModel):
    """Metadata for activation capture (RepE training)."""

    enabled: bool = False
    target_layers: list[int] = Field(default_factory=lambda: [12, 14, 16, 18, 20])
    capture_path: str | None = None
    model_name: str | None = None
    notes: str | None = None


class EvalMetadata(BaseModel):
    """Evaluation benchmark alignment."""

    secalign_category: str | None = None
    mtj_bench_pattern: str | None = None
    poisonedrag_vector: str | None = None
    adversarial_poetry_variant: str | None = None
    custom_tags: list[str] = Field(default_factory=list)


class GenerationMetadata(BaseModel):
    """Metadata about trajectory generation."""

    generator_model: str | None = None
    judge_model: str | None = None
    generation_timestamp: datetime | None = None
    mutation_parent_id: str | None = None
    mutation_type: str | None = None
    quality_score: float | None = Field(ge=0, le=1, default=None)


class TRYLOCKTrajectory(BaseModel):
    """Complete TRYLOCK trajectory record."""

    id: str = Field(pattern=r"^trylock-(mt|st|ind|bn|ta)-[0-9]{5}$")
    version: str = "2.0"
    attack_metadata: AttackMetadata
    conversation: list[ConversationTurn] = Field(min_length=1)
    pivot_turn_index: int | None = Field(ge=1, default=None)
    activation_capture: ActivationCapture = Field(default_factory=ActivationCapture)
    responses: Responses
    eval_metadata: EvalMetadata = Field(default_factory=EvalMetadata)
    generation_metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)

    @field_validator("conversation")
    @classmethod
    def validate_turn_sequence(cls, v: list[ConversationTurn]) -> list[ConversationTurn]:
        """Ensure turns are sequential starting from 1."""
        for i, turn in enumerate(v):
            if turn.turn != i + 1:
                raise ValueError(f"Turn {i + 1} has incorrect turn number {turn.turn}")
        return v

    @model_validator(mode="after")
    def validate_pivot_index(self) -> "TRYLOCKTrajectory":
        """Ensure pivot index is within conversation bounds."""
        if self.pivot_turn_index is not None:
            if self.pivot_turn_index > len(self.conversation):
                raise ValueError(
                    f"pivot_turn_index {self.pivot_turn_index} exceeds "
                    f"conversation length {len(self.conversation)}"
                )
        return self

    @model_validator(mode="after")
    def validate_id_prefix_matches_family(self) -> "TRYLOCKTrajectory":
        """Ensure ID prefix matches attack family."""
        prefix = self.id.split("-")[1]
        family = self.attack_metadata.family

        expected_prefixes = {
            AttackFamily.MULTI_TURN_MANIPULATION: "mt",
            AttackFamily.DIRECT_INJECTION: "st",
            AttackFamily.INDIRECT_INJECTION: "ind",
            AttackFamily.BENIGN: "bn",
            AttackFamily.TOOL_AGENT_ABUSE: "ta",
            AttackFamily.OBFUSCATION_WRAPPERS: "st",  # Can also be mt
        }

        # Allow some flexibility for obfuscation wrappers
        if family == AttackFamily.OBFUSCATION_WRAPPERS:
            if prefix not in ["st", "mt"]:
                raise ValueError(
                    f"ID prefix '{prefix}' doesn't match family '{family}'"
                )
        elif expected_prefixes.get(family) and prefix != expected_prefixes[family]:
            # Warning only - don't fail validation for this
            pass

        return self


class TrajectoryValidator:
    """
    Validate TRYLOCK trajectory records.

    Combines JSON Schema validation with Pydantic validation and
    custom business logic checks.
    """

    def __init__(self, schema_path: str | Path | None = None):
        """
        Initialize validator.

        Args:
            schema_path: Path to JSON schema file. If None, uses Pydantic only.
        """
        self.json_schema: dict | None = None
        self.json_validator: Draft7Validator | None = None

        if schema_path:
            schema_path = Path(schema_path)
            if schema_path.exists():
                with open(schema_path) as f:
                    self.json_schema = json.load(f)
                self.json_validator = Draft7Validator(self.json_schema)
        else:
            # Try to load from default location
            default_path = Path(__file__).parent / "trajectory_schema.json"
            if default_path.exists():
                with open(default_path) as f:
                    self.json_schema = json.load(f)
                self.json_validator = Draft7Validator(self.json_schema)

    def validate(self, record: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate a trajectory record.

        Args:
            record: The trajectory record to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []

        # JSON Schema validation
        if self.json_validator:
            for error in self.json_validator.iter_errors(record):
                path = ".".join(str(p) for p in error.path)
                errors.append(f"Schema: {error.message} at '{path}'")

        # Pydantic validation
        try:
            TRYLOCKTrajectory(**record)
        except Exception as e:
            errors.append(f"Pydantic: {str(e)}")

        # Custom business logic validation
        errors.extend(self._validate_business_logic(record))

        return len(errors) == 0, errors

    def _validate_business_logic(self, record: dict[str, Any]) -> list[str]:
        """Additional business logic validation."""
        errors: list[str] = []

        family = record.get("attack_metadata", {}).get("family")
        conversation = record.get("conversation", [])
        responses = record.get("responses", {})
        pivot_idx = record.get("pivot_turn_index")

        # Rule 1: Non-benign trajectories should have attack turns
        if family and family != "benign":
            has_attack_turn = False
            for turn in conversation:
                labels = turn.get("turn_labels", {})
                if labels.get("attack_intent_score", 0) > 0:
                    has_attack_turn = True
                    break
            if not has_attack_turn:
                errors.append(
                    "Non-benign trajectory has no turns with attack_intent_score > 0"
                )

        # Rule 2: Pivot point consistency
        if pivot_idx:
            if pivot_idx <= len(conversation):
                pivot_turn = conversation[pivot_idx - 1]
                if not pivot_turn.get("is_pivot_point", False):
                    errors.append(
                        f"Turn {pivot_idx} is marked as pivot_turn_index "
                        "but is_pivot_point is False"
                    )
                # Check pivot stage
                turn_labels = pivot_turn.get("turn_labels", {})
                stage = turn_labels.get("attack_stage")
                if stage and stage not in ["pivot", "payload", "probe"]:
                    errors.append(
                        f"Pivot turn has unexpected stage '{stage}' "
                        "(expected: pivot, payload, or probe)"
                    )

        # Rule 3: Only one pivot point should exist
        pivot_count = sum(1 for t in conversation if t.get("is_pivot_point", False))
        if pivot_count > 1:
            errors.append(
                f"Multiple pivot points found ({pivot_count}), expected at most 1"
            )

        # Rule 4: Attack trajectories should have rejected_unsafe
        if family and family != "benign":
            if not responses.get("rejected_unsafe"):
                errors.append(
                    "Attack trajectory should have rejected_unsafe response"
                )

        # Rule 5: Benign trajectories should have rejected_overblock
        if family == "benign":
            if not responses.get("rejected_overblock"):
                errors.append(
                    "Benign trajectory should have rejected_overblock response"
                )
            # Benign should not have pivot
            if pivot_idx is not None:
                errors.append(
                    "Benign trajectory should not have pivot_turn_index"
                )

        # Rule 6: Turn role sequence makes sense
        for i, turn in enumerate(conversation):
            role = turn.get("role")
            if i > 0:
                prev_role = conversation[i - 1].get("role")
                # User shouldn't follow user (usually)
                if role == "user" and prev_role == "user":
                    # Allow but warn
                    pass
                # Assistant shouldn't follow assistant without tool
                if role == "assistant" and prev_role == "assistant":
                    errors.append(
                        f"Turn {i + 1}: assistant turn follows assistant turn"
                    )

        # Rule 7: Intent scores should generally increase (for attacks)
        if family and family != "benign":
            prev_score = 0
            decreases = 0
            for turn in conversation:
                labels = turn.get("turn_labels", {})
                score = labels.get("attack_intent_score", 0)
                if score < prev_score:
                    decreases += 1
                prev_score = score
            # Allow some decreases (recovery turns may reset) but warn if many
            if decreases > len(conversation) / 2:
                errors.append(
                    f"Intent scores decrease frequently ({decreases} times) - "
                    "expected generally increasing pattern for attacks"
                )

        # Rule 8: Validate response type matches expected action
        if pivot_idx and pivot_idx <= len(conversation):
            pivot_turn = conversation[pivot_idx - 1]
            expected_action = pivot_turn.get("turn_labels", {}).get("expected_action")
            chosen_type = responses.get("chosen", {}).get("response_type")

            action_to_response = {
                "continue": ["engaged", "engaged_with_context"],
                "warn": ["engaged_with_context", "clarification_request"],
                "ask_clarify": ["clarification_request"],
                "soft_decline": ["soft_decline", "soft_decline_with_redirect"],
                "hard_decline": ["hard_decline", "soft_decline"],
            }

            if expected_action and chosen_type:
                valid_responses = action_to_response.get(expected_action, [])
                if valid_responses and chosen_type not in valid_responses:
                    # Warning level - don't fail validation
                    pass

        return errors

    def validate_file(
        self, path: str | Path
    ) -> tuple[int, int, list[tuple[int, list[str]]]]:
        """
        Validate a JSONL file of trajectories.

        Args:
            path: Path to the JSONL file

        Returns:
            Tuple of (valid_count, invalid_count, list of (line_number, errors))
        """
        valid = 0
        invalid = 0
        all_errors: list[tuple[int, list[str]]] = []

        with open(path) as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line.strip())
                    is_valid, errors = self.validate(record)
                    if is_valid:
                        valid += 1
                    else:
                        invalid += 1
                        all_errors.append((i, errors))
                except json.JSONDecodeError as e:
                    invalid += 1
                    all_errors.append((i, [f"JSON parse error: {e}"]))

        return valid, invalid, all_errors

    def validate_record_strict(self, record: dict[str, Any]) -> TRYLOCKTrajectory:
        """
        Validate and return a typed trajectory object.

        Raises:
            ValueError: If validation fails
        """
        is_valid, errors = self.validate(record)
        if not is_valid:
            raise ValueError(f"Validation failed: {errors}")
        return TRYLOCKTrajectory(**record)


def validate_trajectory(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Convenience function for single record validation."""
    validator = TrajectoryValidator()
    return validator.validate(record)


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python validator.py <trajectory.jsonl> [--verbose]")
        print("\nValidates TRYLOCK trajectory records against the v2.0 schema.")
        sys.exit(1)

    file_path = sys.argv[1]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    validator = TrajectoryValidator()
    valid, invalid, errors = validator.validate_file(file_path)

    print(f"\nValidation Results for: {file_path}")
    print("=" * 50)
    print(f"Valid records:   {valid}")
    print(f"Invalid records: {invalid}")
    print(f"Total:           {valid + invalid}")

    if invalid > 0:
        print(f"\nSuccess rate: {valid / (valid + invalid) * 100:.1f}%")

        if verbose:
            print("\n" + "=" * 50)
            print("Errors:")
            for line_num, errs in errors:
                print(f"\nLine {line_num}:")
                for e in errs:
                    print(f"  - {e}")
        else:
            # Show first 5 errors
            print(f"\nFirst {min(5, len(errors))} errors:")
            for line_num, errs in errors[:5]:
                print(f"\nLine {line_num}:")
                for e in errs[:3]:  # Max 3 errors per line
                    print(f"  - {e}")
                if len(errs) > 3:
                    print(f"  ... and {len(errs) - 3} more")

            if len(errors) > 5:
                print(f"\n... and {len(errors) - 5} more invalid records")
                print("Use --verbose to see all errors")

        sys.exit(1)
    else:
        print("\nAll records valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
