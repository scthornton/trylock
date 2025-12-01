#!/usr/bin/env python3
"""
AEGIS Production Trajectory Generator

Generates attack trajectories using Claude API with judge filtering.

Usage:
    python scripts/generate_trajectories.py --target 500 --category multi_turn_manipulation
    python scripts/generate_trajectories.py --target 500 --all-categories
    python scripts/generate_trajectories.py --target 500 --all-categories --dry-run
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Warning: anthropic package not installed")

from generation.red_bot import RedBot, AttackConfig
from generation.judge_bot import JudgeBot


@dataclass
class GenerationConfig:
    """Configuration for trajectory generation."""
    category: str
    target_count: int
    min_quality_score: float = 0.7
    max_retries: int = 3
    batch_size: int = 10
    save_interval: int = 10
    dry_run: bool = False


CATEGORIES = [
    "multi_turn_manipulation",
    "indirect_injection",
    "obfuscation_wrappers",
    "direct_injection",
    "tool_agent_abuse",
    "benign_hard_negatives",
]

# Attack types per category for variety
ATTACK_TYPES = {
    "multi_turn_manipulation": [
        "crescendo", "context_anchoring", "trust_building",
        "boundary_softening", "false_continuity", "expert_framing"
    ],
    "indirect_injection": [
        "rag_poisoning", "email_injection", "web_scrape_injection",
        "code_injection", "api_response_injection", "document_injection"
    ],
    "obfuscation_wrappers": [
        "base64_encoding", "rot13_encoding", "leetspeak", "unicode_substitution",
        "word_splitting", "fictional_framing", "academic_framing"
    ],
    "direct_injection": [
        "system_prompt_override", "dan_jailbreak", "developer_mode",
        "sudo_mode", "token_manipulation", "persona_override"
    ],
    "tool_agent_abuse": [
        "destructive_tool_call", "data_exfiltration", "privilege_escalation",
        "sandbox_escape", "credential_theft", "attack_chain"
    ],
    "benign_hard_negatives": [
        "security_research", "ctf_learning", "creative_writing",
        "it_administration", "authorized_pentest", "malware_analysis"
    ],
}

MANIPULATION_TACTICS = [
    "expert_persona_adoption", "progressive_boundary_erosion",
    "compliance_momentum", "emotional_manipulation", "false_authority_claim",
    "urgency_pressure", "legitimizing_frame", "hypothetical_distancing",
    "context_shift", "false_continuity"
]


def load_seed_examples(category: str) -> list[dict]:
    """Load seed examples for few-shot prompting."""
    seed_path = Path(__file__).parent.parent / "data" / "tier1_open" / "attacks" / f"{category}.jsonl"

    if not seed_path.exists():
        print(f"Warning: No seed file for {category}")
        return []

    examples = []
    with open(seed_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    return examples


def format_seed_examples(examples: list[dict], num_examples: int = 3) -> str:
    """Format seed examples for few-shot prompting."""
    if not examples:
        return ""

    selected = random.sample(examples, min(num_examples, len(examples)))

    formatted = "## Example Trajectories\n\nHere are examples of the format and quality expected:\n\n"

    for i, ex in enumerate(selected, 1):
        formatted += f"### Example {i}\n```json\n{json.dumps(ex, indent=2)[:2000]}...\n```\n\n"

    return formatted


def generate_attack_config(category: str, difficulty: str = None) -> AttackConfig:
    """Generate random attack config for a category."""
    attack_types = ATTACK_TYPES.get(category, ["generic"])
    attack_type = random.choice(attack_types)

    # Random difficulty if not specified
    if difficulty is None:
        difficulties = ["easy", "medium", "hard", "expert"]
        weights = [0.1, 0.3, 0.4, 0.2]
        difficulty = random.choices(difficulties, weights=weights)[0]

    # Random tactics
    num_tactics = random.randint(1, 3)
    tactics = random.sample(MANIPULATION_TACTICS, num_tactics)

    # Random wrapper (30% chance for applicable categories)
    wrapper = None
    if category == "obfuscation_wrappers" or random.random() < 0.3:
        wrappers = ["base64", "roleplay", "code_comments", "multilingual"]
        wrapper = random.choice(wrappers)

    # Turn count based on category
    if category == "direct_injection":
        target_turns = random.randint(1, 3)
    elif category == "multi_turn_manipulation":
        target_turns = random.randint(5, 8)
    else:
        target_turns = random.randint(4, 6)

    return AttackConfig(
        attack_family=category,
        attack_type=attack_type,
        wrapper=wrapper,
        tactics=tactics,
        difficulty=difficulty,
        target_turns=target_turns,
    )


def score_trajectory(trajectory: dict, judge: JudgeBot) -> tuple[float, dict]:
    """Score a trajectory using the judge bot."""
    try:
        result = judge.judge(
            trajectory.get("conversation", []),
            trajectory.get("attack_metadata"),
        )

        if not result.success:
            return 0.0, {"error": result.error}

        # Calculate quality score based on:
        # - Has conversation
        # - Has proper turn labels
        # - Has pivot turn (for attacks)
        # - Has all response types
        score = 0.0
        reasons = []

        conv = trajectory.get("conversation", [])
        if len(conv) >= 3:
            score += 0.2
            reasons.append("sufficient_turns")

        if result.pivot_turn_index is not None:
            score += 0.2
            reasons.append("has_pivot")

        if result.responses.chosen_content:
            score += 0.2
            reasons.append("has_chosen")

        if result.responses.rejected_unsafe_content:
            score += 0.2
            reasons.append("has_rejected_unsafe")

        if len(result.turn_labels) > 0:
            score += 0.2
            reasons.append("has_labels")

        return score, {
            "score": score,
            "reasons": reasons,
            "pivot_turn": result.pivot_turn_index,
            "trajectory_type": result.trajectory_type,
        }

    except Exception as e:
        return 0.0, {"error": str(e)}


def merge_judgment_into_trajectory(trajectory: dict, judge: JudgeBot) -> dict:
    """Merge judge labels and responses into trajectory."""
    try:
        result = judge.judge(
            trajectory.get("conversation", []),
            trajectory.get("attack_metadata"),
            generate_responses=True,
        )

        if not result.success:
            return trajectory

        # Update turn labels
        for turn in trajectory.get("conversation", []):
            turn_num = turn.get("turn")
            if turn.get("role") == "user":
                for label in result.turn_labels:
                    if label.turn == turn_num:
                        turn["turn_labels"] = {
                            "attack_stage": label.attack_stage,
                            "attack_intent_score": label.attack_intent_score,
                            "manipulation_tactics": label.manipulation_tactics,
                            "expected_action": label.expected_action,
                        }
                        break

        # Update pivot turn
        trajectory["pivot_turn_index"] = result.pivot_turn_index

        # Update responses
        trajectory["responses"] = {
            "chosen": {
                "content": result.responses.chosen_content,
            },
        }

        if result.responses.rejected_unsafe_content:
            trajectory["responses"]["rejected_unsafe"] = {
                "content": result.responses.rejected_unsafe_content,
                "failure_mode": result.responses.rejected_unsafe_failure_mode,
            }

        if result.responses.rejected_overblock_content:
            trajectory["responses"]["rejected_overblock"] = {
                "content": result.responses.rejected_overblock_content,
                "failure_mode": result.responses.rejected_overblock_failure_mode,
            }

        return trajectory

    except Exception as e:
        print(f"Warning: Failed to merge judgment: {e}")
        return trajectory


class TrajectoryGenerator:
    """Production trajectory generator with quality filtering."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        judge_model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        judge_provider: str = "anthropic",
    ):
        self.provider = provider

        self.red_bot = RedBot(
            model_name=model,
            api_provider=provider,
            temperature=0.9,
        )

        # Judge always uses Claude for consistency
        self.judge = JudgeBot(
            model_name=judge_model,
            api_provider=judge_provider,
            temperature=0.3,
        )

        self.stats = {
            "generated": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
        }

    def generate_one(
        self,
        category: str,
        seed_examples: list[dict],
        difficulty: str = None,
    ) -> tuple[dict | None, dict]:
        """Generate and validate one trajectory."""

        config = generate_attack_config(category, difficulty)

        try:
            # Generate with red bot
            result = self.red_bot.generate(config)

            if not result.success:
                return None, {"error": result.error}

            # Build trajectory dict
            trajectory = {
                "id": f"aegis-{category[:2]}-{int(time.time()*1000)}",
                "version": "2.0",
                "attack_metadata": {
                    "family": category,
                    "type": config.attack_type,
                    "wrappers": [config.wrapper] if config.wrapper else [],
                    "manipulation_tactics": config.tactics,
                    "difficulty": config.difficulty,
                    "source": "synthetic_claude",
                },
                "conversation": result.conversation,
                "pivot_turn_index": result.pivot_turn,
            }

            # Score with judge
            score, score_info = score_trajectory(trajectory, self.judge)

            if score >= 0.6:  # Minimum threshold
                # Merge judge labels
                trajectory = merge_judgment_into_trajectory(trajectory, self.judge)
                return trajectory, {"score": score, **score_info}
            else:
                return None, {"score": score, **score_info}

        except Exception as e:
            return None, {"error": str(e)}

    def generate_batch(
        self,
        config: GenerationConfig,
        output_path: Path,
    ) -> list[dict]:
        """Generate a batch of trajectories for a category."""

        # Load seed examples
        seed_examples = load_seed_examples(config.category)
        print(f"Loaded {len(seed_examples)} seed examples for {config.category}")

        # Load existing if resuming
        existing = []
        if output_path.exists():
            with open(output_path) as f:
                for line in f:
                    if line.strip():
                        existing.append(json.loads(line))
            print(f"Found {len(existing)} existing trajectories")

        generated = existing.copy()
        remaining = config.target_count - len(generated)

        if remaining <= 0:
            print(f"Already have {len(generated)} trajectories, target is {config.target_count}")
            return generated

        print(f"Generating {remaining} more trajectories for {config.category}")

        # Progress tracking
        start_time = time.time()
        last_save = time.time()

        while len(generated) < config.target_count:
            # Vary difficulty
            difficulties = ["easy", "medium", "hard", "expert"]
            difficulty = random.choices(
                difficulties,
                weights=[0.1, 0.3, 0.4, 0.2]
            )[0]

            if config.dry_run:
                print(f"[DRY RUN] Would generate {config.category}/{difficulty}")
                generated.append({"dry_run": True})
                time.sleep(0.1)
                continue

            trajectory, info = self.generate_one(
                config.category,
                seed_examples,
                difficulty,
            )

            self.stats["generated"] += 1

            if trajectory:
                generated.append(trajectory)
                self.stats["passed"] += 1
                print(f"✓ [{len(generated)}/{config.target_count}] "
                      f"Generated {config.category}/{difficulty} "
                      f"(score: {info.get('score', 0):.2f})")
            else:
                self.stats["failed"] += 1
                error = info.get("error", info.get("score", "unknown"))
                print(f"✗ Failed: {error}")

            # Save periodically
            if time.time() - last_save > 60:  # Every minute
                self._save_progress(generated, output_path)
                last_save = time.time()

            # Rate limiting
            time.sleep(1)  # Basic rate limit

        # Final save
        self._save_progress(generated, output_path)

        elapsed = time.time() - start_time
        print(f"\nCompleted {config.category}:")
        print(f"  Generated: {self.stats['generated']}")
        print(f"  Passed: {self.stats['passed']}")
        print(f"  Failed: {self.stats['failed']}")
        print(f"  Time: {elapsed/60:.1f} minutes")

        return generated

    def _save_progress(self, trajectories: list[dict], output_path: Path):
        """Save current progress to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for traj in trajectories:
                if not traj.get("dry_run"):
                    f.write(json.dumps(traj) + "\n")

        print(f"  Saved {len(trajectories)} trajectories to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate AEGIS attack trajectories")
    parser.add_argument("--target", type=int, default=500, help="Target count per category")
    parser.add_argument("--category", type=str, help="Single category to generate")
    parser.add_argument("--all-categories", action="store_true", help="Generate all categories")
    parser.add_argument("--output-dir", type=str, default="data/tier1_open/attacks", help="Output directory")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Generation model")
    parser.add_argument("--provider", type=str, default="anthropic", choices=["anthropic", "openai"], help="API provider")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from existing files")

    args = parser.parse_args()

    # Validate
    if not args.category and not args.all_categories:
        parser.error("Specify --category or --all-categories")

    categories = CATEGORIES if args.all_categories else [args.category]

    # Validate categories
    for cat in categories:
        if cat not in CATEGORIES:
            parser.error(f"Unknown category: {cat}. Valid: {CATEGORIES}")

    # Check API keys based on provider
    if not args.dry_run:
        if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            parser.error("ANTHROPIC_API_KEY environment variable required")
        if args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            parser.error("OPENAI_API_KEY environment variable required")
        # Judge always needs Claude
        if not os.environ.get("ANTHROPIC_API_KEY"):
            parser.error("ANTHROPIC_API_KEY required for judge bot")

    print("=" * 60)
    print("AEGIS Trajectory Generator")
    print("=" * 60)
    print(f"Target: {args.target} per category")
    print(f"Categories: {categories}")
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60, flush=True)

    if args.dry_run:
        print("\n[DRY RUN MODE - No API calls will be made]\n")
        return

    # Initialize generator
    generator = TrajectoryGenerator(model=args.model, provider=args.provider)

    output_dir = Path(__file__).parent.parent / args.output_dir

    for category in categories:
        print(f"\n{'='*60}")
        print(f"Generating: {category}")
        print(f"{'='*60}")

        config = GenerationConfig(
            category=category,
            target_count=args.target,
            dry_run=args.dry_run,
        )

        output_path = output_dir / f"{category}_generated.jsonl"

        generator.generate_batch(config, output_path)

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
