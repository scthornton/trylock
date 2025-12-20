#!/usr/bin/env python3
"""
TRYLOCK Data Validation Script

Validates generated trajectories for:
- JSON parsing
- Schema compliance
- Quality metrics
- Distribution analysis
- Deduplication candidates

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --fix  # Auto-fix minor issues
"""

import argparse
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# Required fields in each trajectory
REQUIRED_FIELDS = ["id", "conversation", "attack_metadata"]
REQUIRED_METADATA = ["family", "type", "difficulty"]
REQUIRED_TURN_FIELDS = ["turn", "role", "content"]

CATEGORIES = [
    "multi_turn_manipulation",
    "indirect_injection",
    "obfuscation_wrappers",
    "direct_injection",
    "tool_agent_abuse",
    "benign_hard_negatives",
]

DIFFICULTIES = ["easy", "medium", "hard", "expert"]


class ValidationResult:
    def __init__(self):
        self.total = 0
        self.valid = 0
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(Counter)
        self.duplicates = []
        self.content_hashes = {}

    def add_error(self, file: str, line: int, msg: str):
        self.errors.append(f"{file}:{line}: ERROR - {msg}")

    def add_warning(self, file: str, line: int, msg: str):
        self.warnings.append(f"{file}:{line}: WARN - {msg}")


def hash_conversation(conv: list) -> str:
    """Create hash of conversation content for dedup."""
    content = "".join(
        t.get("content", "")[:100] for t in conv if t.get("role") == "user"
    )
    return hashlib.md5(content.encode()).hexdigest()[:12]


def validate_trajectory(traj: dict, file: str, line: int, result: ValidationResult) -> bool:
    """Validate a single trajectory."""
    valid = True

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in traj:
            result.add_error(file, line, f"Missing required field: {field}")
            valid = False

    if not valid:
        return False

    # Check attack_metadata
    metadata = traj.get("attack_metadata", {})
    for field in REQUIRED_METADATA:
        if field not in metadata:
            result.add_error(file, line, f"Missing metadata field: {field}")
            valid = False

    # Track stats
    family = metadata.get("family", "unknown")
    difficulty = metadata.get("difficulty", "unknown")
    attack_type = metadata.get("type", "unknown")

    result.stats["families"][family] += 1
    result.stats["difficulties"][difficulty] += 1
    result.stats["types"][attack_type] += 1

    # Validate difficulty
    if difficulty not in DIFFICULTIES and difficulty != "unknown":
        result.add_warning(file, line, f"Non-standard difficulty: {difficulty}")

    # Check conversation
    conv = traj.get("conversation", [])
    if not conv:
        result.add_error(file, line, "Empty conversation")
        valid = False
    elif len(conv) < 2:
        result.add_warning(file, line, f"Very short conversation: {len(conv)} turns")

    # Check turn structure
    for i, turn in enumerate(conv):
        for field in REQUIRED_TURN_FIELDS:
            if field not in turn:
                result.add_error(file, line, f"Turn {i+1} missing field: {field}")
                valid = False

        # Check role
        role = turn.get("role", "")
        if role not in ["user", "assistant", "system"]:
            result.add_warning(file, line, f"Turn {i+1} unusual role: {role}")

        # Check content
        content = turn.get("content", "")
        if not content or len(content.strip()) < 5:
            result.add_warning(file, line, f"Turn {i+1} very short content")

    # Check for pivot turn
    pivot = traj.get("pivot_turn_index") or traj.get("pivot_turn")
    if pivot is not None:
        result.stats["has_pivot"]["yes"] += 1
        if pivot > len(conv):
            result.add_warning(file, line, f"Pivot turn {pivot} > conversation length {len(conv)}")
    else:
        result.stats["has_pivot"]["no"] += 1
        if family != "benign_hard_negatives":
            result.add_warning(file, line, "Attack trajectory missing pivot_turn")

    # Check for responses
    responses = traj.get("responses", {})
    if responses:
        result.stats["has_responses"]["yes"] += 1
        if responses.get("chosen"):
            result.stats["response_types"]["chosen"] += 1
        if responses.get("rejected_unsafe"):
            result.stats["response_types"]["rejected_unsafe"] += 1
        if responses.get("rejected_overblock"):
            result.stats["response_types"]["rejected_overblock"] += 1
    else:
        result.stats["has_responses"]["no"] += 1

    # Check for duplicates via content hash
    conv_hash = hash_conversation(conv)
    traj_id = traj.get("id", f"{file}:{line}")

    if conv_hash in result.content_hashes:
        existing_id = result.content_hashes[conv_hash]
        result.duplicates.append((traj_id, existing_id))
        result.add_warning(file, line, f"Possible duplicate of {existing_id}")
    else:
        result.content_hashes[conv_hash] = traj_id

    # Track turn count stats
    result.stats["turn_counts"][len(conv)] += 1

    return valid


def validate_file(filepath: Path, result: ValidationResult) -> int:
    """Validate all trajectories in a file."""
    valid_count = 0
    filename = filepath.name

    try:
        with open(filepath) as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                result.total += 1

                try:
                    traj = json.loads(line)
                    if validate_trajectory(traj, filename, line_num, result):
                        valid_count += 1
                        result.valid += 1
                except json.JSONDecodeError as e:
                    result.add_error(filename, line_num, f"JSON parse error: {e}")

    except Exception as e:
        result.add_error(filename, 0, f"File read error: {e}")

    return valid_count


def print_report(result: ValidationResult):
    """Print validation report."""
    print("\n" + "=" * 60)
    print("TRYLOCK DATA VALIDATION REPORT")
    print("=" * 60)

    # Summary
    print(f"\n## Summary")
    print(f"Total trajectories: {result.total}")
    print(f"Valid trajectories: {result.valid} ({100*result.valid/max(result.total,1):.1f}%)")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Potential duplicates: {len(result.duplicates)}")

    # Distribution by family
    print(f"\n## Distribution by Family")
    for family, count in sorted(result.stats["families"].items()):
        pct = 100 * count / max(result.total, 1)
        bar = "#" * int(pct / 2)
        print(f"  {family:30} {count:4} ({pct:5.1f}%) {bar}")

    # Distribution by difficulty
    print(f"\n## Distribution by Difficulty")
    for diff in DIFFICULTIES:
        count = result.stats["difficulties"].get(diff, 0)
        pct = 100 * count / max(result.total, 1)
        bar = "#" * int(pct / 2)
        print(f"  {diff:10} {count:4} ({pct:5.1f}%) {bar}")

    # Turn count distribution
    print(f"\n## Turn Count Distribution")
    for turns, count in sorted(result.stats["turn_counts"].items()):
        pct = 100 * count / max(result.total, 1)
        bar = "#" * int(pct / 2)
        print(f"  {turns:2} turns: {count:4} ({pct:5.1f}%) {bar}")

    # Response coverage
    print(f"\n## Response Coverage")
    has_resp = result.stats["has_responses"].get("yes", 0)
    no_resp = result.stats["has_responses"].get("no", 0)
    print(f"  Has responses: {has_resp} ({100*has_resp/max(result.total,1):.1f}%)")
    print(f"  No responses:  {no_resp} ({100*no_resp/max(result.total,1):.1f}%)")

    for rtype, count in result.stats["response_types"].items():
        print(f"    - {rtype}: {count}")

    # Pivot turn coverage
    print(f"\n## Pivot Turn Coverage")
    has_pivot = result.stats["has_pivot"].get("yes", 0)
    no_pivot = result.stats["has_pivot"].get("no", 0)
    print(f"  Has pivot: {has_pivot} ({100*has_pivot/max(result.total,1):.1f}%)")
    print(f"  No pivot:  {no_pivot} ({100*no_pivot/max(result.total,1):.1f}%)")

    # Errors (first 20)
    if result.errors:
        print(f"\n## Errors (showing first 20 of {len(result.errors)})")
        for err in result.errors[:20]:
            print(f"  {err}")

    # Warnings (first 20)
    if result.warnings:
        print(f"\n## Warnings (showing first 20 of {len(result.warnings)})")
        for warn in result.warnings[:20]:
            print(f"  {warn}")

    # Duplicates (first 10)
    if result.duplicates:
        print(f"\n## Potential Duplicates (showing first 10 of {len(result.duplicates)})")
        for dup1, dup2 in result.duplicates[:10]:
            print(f"  {dup1} <-> {dup2}")

    print("\n" + "=" * 60)

    # Final verdict
    if len(result.errors) == 0:
        print("RESULT: PASS - All trajectories valid")
    else:
        print(f"RESULT: FAIL - {len(result.errors)} errors found")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate TRYLOCK trajectories")
    parser.add_argument("--data-dir", type=str, default="data/tier1_open/attacks",
                        help="Directory containing trajectory files")
    parser.add_argument("--fix", action="store_true", help="Auto-fix minor issues")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / args.data_dir

    print(f"Validating trajectories in: {data_dir}")

    result = ValidationResult()

    # Find all generated files
    files = sorted(data_dir.glob("*_generated.jsonl"))

    if not files:
        print("No generated files found!")
        return 1

    print(f"Found {len(files)} files to validate")

    for filepath in files:
        print(f"  Validating {filepath.name}...", end=" ", flush=True)
        valid = validate_file(filepath, result)
        total_in_file = sum(1 for _ in open(filepath) if _.strip())
        print(f"{valid}/{total_in_file} valid")

    print_report(result)

    return 0 if len(result.errors) == 0 else 1


if __name__ == "__main__":
    exit(main())
