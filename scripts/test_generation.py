#!/usr/bin/env python3
"""
Quick test script to verify trajectory generation works.

Usage:
    export ANTHROPIC_API_KEY=your-key
    python scripts/test_generation.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_generation():
    """Test generating a single trajectory."""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        return False

    print("Testing TRYLOCK trajectory generation...")
    print("=" * 50)

    try:
        from generation.red_bot import RedBot, AttackConfig
        from generation.judge_bot import JudgeBot

        # Initialize bots
        print("\n1. Initializing Red Bot...")
        red_bot = RedBot(
            model_name="claude-sonnet-4-20250514",
            api_provider="anthropic",
            temperature=0.9,
        )
        print("   ✓ Red Bot initialized")

        print("\n2. Initializing Judge Bot...")
        judge = JudgeBot(
            model_name="claude-sonnet-4-20250514",
            api_provider="anthropic",
            temperature=0.3,
        )
        print("   ✓ Judge Bot initialized")

        # Generate one trajectory
        print("\n3. Generating test trajectory...")
        config = AttackConfig(
            attack_family="multi_turn_manipulation",
            attack_type="crescendo",
            wrapper=None,
            tactics=["expert_persona_adoption", "progressive_boundary_erosion"],
            difficulty="medium",
            target_turns=5,
        )

        result = red_bot.generate(config)

        if result.success:
            print(f"   ✓ Generated trajectory with {len(result.conversation)} turns")
            print(f"   Pivot turn: {result.pivot_turn}")
        else:
            print(f"   ✗ Generation failed: {result.error}")
            return False

        # Judge the trajectory
        print("\n4. Judging trajectory...")
        judgment = judge.judge(result.conversation, {"family": "multi_turn_manipulation"})

        if judgment.success:
            print(f"   ✓ Judgment successful")
            print(f"   Type: {judgment.trajectory_type}")
            print(f"   Pivot: {judgment.pivot_turn_index}")
            print(f"   Turn labels: {len(judgment.turn_labels)}")
        else:
            print(f"   ✗ Judgment failed: {judgment.error}")
            return False

        # Show sample output
        print("\n5. Sample Output:")
        print("-" * 50)

        sample = {
            "id": "trylock-test-001",
            "attack_metadata": {
                "family": config.attack_family,
                "type": config.attack_type,
                "difficulty": config.difficulty,
            },
            "conversation": result.conversation[:2],  # Just first 2 turns
            "pivot_turn": result.pivot_turn,
        }
        print(json.dumps(sample, indent=2))

        print("\n" + "=" * 50)
        print("✓ TEST PASSED - Generation pipeline working!")
        print("=" * 50)
        print("\nNext steps:")
        print("  python scripts/generate_trajectories.py --target 10 --category multi_turn_manipulation")
        print("  python scripts/generate_trajectories.py --target 500 --all-categories")

        return True

    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("  Make sure you're in the trylock directory and dependencies are installed")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_generation()
    sys.exit(0 if success else 1)
