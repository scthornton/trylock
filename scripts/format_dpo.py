#!/usr/bin/env python3
"""
Convert TRYLOCK trajectories to DPO training format.

Creates:
- data/dpo/train.jsonl (80%)
- data/dpo/val.jsonl (10%)
- data/dpo/test.jsonl (10%)

DPO format:
{
    "prompt": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...],
    "chosen": "The safe/correct response",
    "rejected": "The unsafe/incorrect response",
    "metadata": {...}
}
"""

import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)  # Reproducibility


def build_prompt_from_conversation(conversation, pivot_turn=None):
    """
    Build the prompt (conversation history) up to the point where response is needed.

    For attacks: conversation up to pivot turn
    For benign: full conversation minus last assistant turn
    """
    messages = []

    for turn in conversation:
        role = turn.get('role')
        content = turn.get('content', '')

        if role in ['user', 'assistant']:
            messages.append({
                "role": role,
                "content": content
            })

    # If we have a pivot turn, truncate there
    # Otherwise, remove the last assistant message (that's what we're predicting)
    if messages and messages[-1]['role'] == 'assistant':
        messages = messages[:-1]

    return messages


def convert_trajectory_to_dpo(traj):
    """Convert a single trajectory to DPO format."""

    responses = traj.get('responses', {})
    chosen_resp = responses.get('chosen', {})
    rejected_resp = responses.get('rejected_unsafe', {})

    # Need both chosen and rejected
    chosen_content = chosen_resp.get('content', '')
    rejected_content = rejected_resp.get('content', '')

    if not chosen_content or not rejected_content:
        return None

    # Build prompt
    conversation = traj.get('conversation', [])
    pivot = traj.get('pivot_turn_index') or traj.get('pivot_turn')

    prompt = build_prompt_from_conversation(conversation, pivot)

    if not prompt:
        return None

    # Build metadata
    attack_meta = traj.get('attack_metadata', {})

    return {
        "id": traj.get('id', 'unknown'),
        "prompt": prompt,
        "chosen": chosen_content,
        "rejected": rejected_content,
        "metadata": {
            "family": attack_meta.get('family', 'unknown'),
            "type": attack_meta.get('type', 'unknown'),
            "difficulty": attack_meta.get('difficulty', 'unknown'),
            "pivot_turn": pivot,
            "num_turns": len(conversation),
        }
    }


def load_all_trajectories(data_dir):
    """Load all generated trajectories."""
    trajectories = []

    for filepath in data_dir.glob("*_generated.jsonl"):
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    traj = json.loads(line)
                    traj['_source'] = filepath.name
                    trajectories.append(traj)

    return trajectories


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """Split data into train/val/test sets, stratified by family."""

    # Group by family
    by_family = defaultdict(list)
    for item in data:
        family = item.get('metadata', {}).get('family', 'unknown')
        by_family[family].append(item)

    train, val, test = [], [], []

    for family, items in by_family.items():
        random.shuffle(items)
        n = len(items)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def main():
    data_dir = Path(__file__).parent.parent / "data" / "tier1_open" / "attacks"
    output_dir = Path(__file__).parent.parent / "data" / "dpo"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRYLOCK DPO FORMAT CONVERSION")
    print("=" * 60)

    # Load trajectories
    print("\n1. Loading trajectories...")
    trajectories = load_all_trajectories(data_dir)
    print(f"   Loaded {len(trajectories)} trajectories")

    # Convert to DPO format
    print("\n2. Converting to DPO format...")
    dpo_data = []
    skipped = 0

    for traj in trajectories:
        dpo_item = convert_trajectory_to_dpo(traj)
        if dpo_item:
            dpo_data.append(dpo_item)
        else:
            skipped += 1

    print(f"   Converted: {len(dpo_data)}")
    print(f"   Skipped (missing responses): {skipped}")

    # Split data
    print("\n3. Splitting into train/val/test...")
    train, val, test = split_data(dpo_data)
    print(f"   Train: {len(train)} (80%)")
    print(f"   Val:   {len(val)} (10%)")
    print(f"   Test:  {len(test)} (10%)")

    # Save
    print("\n4. Saving...")

    for name, data in [("train", train), ("val", val), ("test", test)]:
        filepath = output_dir / f"{name}.jsonl"
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        print(f"   Saved {filepath}")

    # Also save a combined version for HuggingFace
    combined_path = output_dir / "trylock_dpo_full.jsonl"
    with open(combined_path, 'w') as f:
        for item in dpo_data:
            f.write(json.dumps(item) + '\n')
    print(f"   Saved {combined_path}")

    # Stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Distribution by family
    print("\n## Distribution by Family (Train)")
    family_counts = defaultdict(int)
    for item in train:
        family_counts[item['metadata']['family']] += 1

    for family, count in sorted(family_counts.items()):
        print(f"   {family:30} {count:4}")

    # Avg prompt length
    prompt_lens = [len(item['prompt']) for item in dpo_data]
    print(f"\n## Prompt Statistics")
    print(f"   Avg turns in prompt: {sum(prompt_lens) / len(prompt_lens):.1f}")
    print(f"   Min: {min(prompt_lens)}, Max: {max(prompt_lens)}")

    # Response lengths
    chosen_lens = [len(item['chosen']) for item in dpo_data]
    rejected_lens = [len(item['rejected']) for item in dpo_data]
    print(f"\n## Response Statistics")
    print(f"   Chosen avg length: {sum(chosen_lens) // len(chosen_lens)} chars")
    print(f"   Rejected avg length: {sum(rejected_lens) // len(rejected_lens)} chars")

    print("\n" + "=" * 60)
    print("DPO CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Review samples: head -1 data/dpo/train.jsonl | python -m json.tool")
    print("  2. Upload to HuggingFace or use for local training")


if __name__ == "__main__":
    main()
