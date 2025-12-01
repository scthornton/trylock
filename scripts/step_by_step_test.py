#!/usr/bin/env python3
"""
Step-by-step diagnostic to find exactly what causes segfault.
Run this and tell me at which step it crashes.
"""

import sys

def step(num, desc):
    print(f"\n{'='*60}")
    print(f"STEP {num}: {desc}")
    print('='*60)

try:
    step(1, "Import torch")
    import torch
    print(f"✓ torch {torch.__version__}")

    step(2, "Import transformers")
    import transformers
    print(f"✓ transformers {transformers.__version__}")

    step(3, "Import AutoTokenizer")
    from transformers import AutoTokenizer
    print("✓ AutoTokenizer imported")

    step(4, "Import AutoModelForCausalLM")
    from transformers import AutoModelForCausalLM
    print("✓ AutoModelForCausalLM imported")

    step(5, "Import AutoModelForSequenceClassification")
    from transformers import AutoModelForSequenceClassification
    print("✓ AutoModelForSequenceClassification imported")

    step(6, "Import peft")
    import peft
    print(f"✓ peft {peft.__version__}")

    step(7, "Import PeftModel")
    from peft import PeftModel
    print("✓ PeftModel imported")

    step(8, "Import safetensors")
    import safetensors
    print(f"✓ safetensors")

    step(9, "Import load_file from safetensors")
    from safetensors.torch import load_file
    print("✓ load_file imported")

    step(10, "Import tqdm")
    from tqdm import tqdm
    print("✓ tqdm imported")

    step(11, "Import json, time, argparse")
    import json, time, argparse
    print("✓ Standard library imports OK")

    step(12, "Import dataclasses")
    from dataclasses import dataclass, field
    print("✓ dataclasses imported")

    step(13, "Import typing")
    from typing import Optional
    print("✓ typing imported")

    step(14, "Import collections")
    from collections import defaultdict
    print("✓ collections imported")

    step(15, "Import re")
    import re
    print("✓ re imported")

    step(16, "Import pathlib")
    from pathlib import Path
    print("✓ pathlib imported")

    print("\n" + "="*60)
    print("ALL IMPORTS SUCCESSFUL!")
    print("="*60)
    print("\nThe segfault is NOT in imports.")
    print("It must be in the script logic or class definitions.")

except Exception as e:
    print(f"\n❌ FAILED at current step!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
