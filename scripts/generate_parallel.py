#!/usr/bin/env python3
"""
Parallel AEGIS Trajectory Generator

Runs multiple generation processes in parallel, splitting work between Claude and OpenAI.

Usage:
    export ANTHROPIC_API_KEY=your-key
    export OPENAI_API_KEY=your-key
    python scripts/generate_parallel.py --target 500
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Categories and their assigned providers
# 80% Claude (5 categories), 20% OpenAI (1 category)
CATEGORY_ASSIGNMENTS = {
    # Claude categories (80%)
    "multi_turn_manipulation": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    "indirect_injection": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    "obfuscation_wrappers": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    "direct_injection": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    "tool_agent_abuse": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    # OpenAI category (20%)
    "benign_hard_negatives": {"provider": "openai", "model": "gpt-4o"},
}


def check_api_keys():
    """Check that required API keys are set."""
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")

    if missing:
        print(f"ERROR: Missing API keys: {', '.join(missing)}")
        print("Set them with:")
        for key in missing:
            print(f"  export {key}=your-key")
        return False
    return True


def run_category(category: str, target: int, provider: str, model: str, log_dir: Path):
    """Launch a subprocess to generate trajectories for one category."""
    log_file = log_dir / f"{category}.log"

    cmd = [
        sys.executable,
        "scripts/generate_trajectories.py",
        "--target", str(target),
        "--category", category,
        "--model", model,
        "--provider", provider,
    ]

    print(f"Starting {category} with {provider}/{model}...")

    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    return process, log_file


def monitor_progress(processes: dict, log_dir: Path, data_dir: Path):
    """Monitor progress of all running processes."""
    while any(p.poll() is None for p, _ in processes.values()):
        print("\n" + "=" * 60)
        print(f"Progress at {time.strftime('%H:%M:%S')}:")
        print("-" * 60)

        for category, (process, log_file) in processes.items():
            # Check generated file
            gen_file = data_dir / f"{category}_generated.jsonl"
            count = 0
            if gen_file.exists():
                with open(gen_file) as f:
                    count = sum(1 for _ in f)

            status = "✓ Done" if process.poll() is not None else "Running"
            provider = CATEGORY_ASSIGNMENTS[category]["provider"]
            print(f"  {category}: {count}/500 [{provider}] {status}")

        print("=" * 60)
        time.sleep(60)  # Update every minute


def main():
    parser = argparse.ArgumentParser(description="Parallel AEGIS trajectory generation")
    parser.add_argument("--target", type=int, default=500, help="Target per category")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    args = parser.parse_args()

    print("=" * 60)
    print("AEGIS Parallel Generator")
    print("=" * 60)
    print(f"\nTarget: {args.target} per category")
    print(f"Total: {args.target * 6} trajectories")
    print("\nCategory assignments:")

    claude_count = 0
    openai_count = 0
    for cat, config in CATEGORY_ASSIGNMENTS.items():
        print(f"  {cat}: {config['provider']}/{config['model']}")
        if config['provider'] == 'anthropic':
            claude_count += 1
        else:
            openai_count += 1

    print(f"\nSplit: Claude {claude_count}/6 ({claude_count*100//6}%), OpenAI {openai_count}/6 ({openai_count*100//6}%)")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN - No processes started]")
        return

    if not check_api_keys():
        sys.exit(1)

    # Create log directory
    log_dir = Path(__file__).parent.parent / "logs" / f"gen_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nLogs: {log_dir}")

    data_dir = Path(__file__).parent.parent / "data" / "tier1_open" / "attacks"

    # Launch all processes
    processes = {}
    for category, config in CATEGORY_ASSIGNMENTS.items():
        process, log_file = run_category(
            category,
            args.target,
            config["provider"],
            config["model"],
            log_dir,
        )
        processes[category] = (process, log_file)
        time.sleep(2)  # Stagger starts slightly

    print(f"\nLaunched {len(processes)} parallel generators")
    print("Monitor with: tail -f logs/gen_*/multi_turn_manipulation.log")

    # Monitor until complete
    try:
        monitor_progress(processes, log_dir, data_dir)
    except KeyboardInterrupt:
        print("\nInterrupted! Killing processes...")
        for category, (process, _) in processes.items():
            if process.poll() is None:
                process.terminate()
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)

    total = 0
    for category in CATEGORY_ASSIGNMENTS:
        gen_file = data_dir / f"{category}_generated.jsonl"
        count = 0
        if gen_file.exists():
            with open(gen_file) as f:
                count = sum(1 for _ in f)
        total += count
        print(f"  {category}: {count}")

    print(f"\nTotal generated: {total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
