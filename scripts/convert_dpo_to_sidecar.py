#!/usr/bin/env python3
"""
Convert DPO training data to sidecar classifier format.

The DPO format has:
- prompt: conversation turns
- chosen: ideal response
- rejected: unsafe response
- metadata: family, type, difficulty, pivot_turn

The sidecar format needs:
- attack_metadata: family, type, difficulty
- conversation: list of turns with turn_labels
- pivot_turn_index: where the attack happens

Usage:
    python scripts/convert_dpo_to_sidecar.py \
        --input data/dpo/train.jsonl \
        --output data/sidecar/train.jsonl
"""

import argparse
import json
from pathlib import Path


def convert_record(dpo_record: dict) -> dict:
    """Convert a single DPO record to sidecar format."""

    conversation = dpo_record.get("prompt", [])
    metadata = dpo_record.get("metadata", {})

    family = metadata.get("family", "unknown")
    attack_type = metadata.get("type", "unknown")
    difficulty = metadata.get("difficulty", "medium")
    pivot_turn = metadata.get("pivot_turn")

    # Determine if this is an attack or benign
    is_attack = family not in ["benign", "benign_hard_negatives", "legitimate"]

    # Build conversation with turn labels
    labeled_conversation = []
    for i, turn in enumerate(conversation):
        turn_num = i + 1

        # Determine turn label based on position
        if not is_attack:
            # Benign - all turns are safe
            turn_labels = {
                "attack_stage": "none",
                "attack_intent_score": 0,
                "manipulation_tactics": [],
                "expected_action": "continue"
            }
        elif pivot_turn is None:
            # Attack but no pivot marked - assume last user turn is payload
            if turn.get("role") == "user" and i == len(conversation) - 1:
                turn_labels = {
                    "attack_stage": "payload",
                    "attack_intent_score": 3,
                    "manipulation_tactics": [],
                    "expected_action": "hard_decline"
                }
            else:
                turn_labels = {
                    "attack_stage": "setup",
                    "attack_intent_score": 1,
                    "manipulation_tactics": [],
                    "expected_action": "continue"
                }
        else:
            # Attack with pivot marked
            if turn_num < pivot_turn:
                # Before pivot - setup/probe phase
                turn_labels = {
                    "attack_stage": "setup" if turn_num < pivot_turn - 1 else "probe",
                    "attack_intent_score": 1 if turn_num < pivot_turn - 1 else 2,
                    "manipulation_tactics": [],
                    "expected_action": "continue"
                }
            elif turn_num == pivot_turn:
                # At pivot - payload delivery
                turn_labels = {
                    "attack_stage": "payload",
                    "attack_intent_score": 3,
                    "manipulation_tactics": [],
                    "expected_action": "hard_decline"
                }
            else:
                # After pivot - extraction/recovery
                turn_labels = {
                    "attack_stage": "extraction",
                    "attack_intent_score": 3,
                    "manipulation_tactics": [],
                    "expected_action": "hard_decline"
                }

        labeled_turn = {
            "turn": turn_num,
            "role": turn.get("role", "user"),
            "content": turn.get("content", ""),
            "turn_labels": turn_labels
        }

        if turn_num == pivot_turn:
            labeled_turn["is_pivot_point"] = True

        labeled_conversation.append(labeled_turn)

    # Build sidecar record
    sidecar_record = {
        "id": dpo_record.get("id", "unknown"),
        "version": "2.0",
        "attack_metadata": {
            "family": family,
            "type": attack_type,
            "difficulty": difficulty,
            "source": "dpo_conversion"
        },
        "conversation": labeled_conversation,
        "pivot_turn_index": pivot_turn,
        "is_attack": is_attack,
        "responses": {
            "chosen": {
                "content": dpo_record.get("chosen", ""),
                "response_type": "refusal" if is_attack else "engaged"
            },
            "rejected": {
                "content": dpo_record.get("rejected", ""),
                "failure_mode": "compliance" if is_attack else "over_refusal"
            }
        }
    }

    return sidecar_record


def convert_file(input_path: Path, output_path: Path) -> dict:
    """Convert an entire file from DPO to sidecar format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": 0,
        "attacks": 0,
        "benign": 0,
        "families": {}
    }

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue

            try:
                dpo_record = json.loads(line)
                sidecar_record = convert_record(dpo_record)

                f_out.write(json.dumps(sidecar_record) + "\n")

                stats["total"] += 1
                if sidecar_record["is_attack"]:
                    stats["attacks"] += 1
                else:
                    stats["benign"] += 1

                family = sidecar_record["attack_metadata"]["family"]
                stats["families"][family] = stats["families"].get(family, 0) + 1

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert DPO to sidecar format")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input DPO JSONL file")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output sidecar JSONL file")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Converting {input_path} -> {output_path}")

    stats = convert_file(input_path, output_path)

    print(f"\nConversion complete:")
    print(f"  Total records: {stats['total']}")
    print(f"  Attacks: {stats['attacks']}")
    print(f"  Benign: {stats['benign']}")
    print(f"\nFamilies:")
    for family, count in sorted(stats["families"].items()):
        print(f"  {family}: {count}")


if __name__ == "__main__":
    main()
