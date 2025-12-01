#!/usr/bin/env python3
"""
Debug script to isolate what's causing the segfault.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

print("\n--- Testing basic imports ---")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except Exception as e:
    print(f"✗ torch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ transformers {transformers.__version__}")
except Exception as e:
    print(f"✗ transformers: {e}")
    sys.exit(1)

try:
    from peft import PeftModel
    print(f"✓ peft")
except Exception as e:
    print(f"✗ peft: {e}")
    sys.exit(1)

try:
    from safetensors.torch import load_file
    print(f"✓ safetensors")
except Exception as e:
    print(f"✗ safetensors: {e}")
    sys.exit(1)

print("\n--- Testing CUDA ---")
try:
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ CUDA check failed: {e}")

print("\n--- Testing local imports ---")
try:
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from scripts.run_eval import RuleBasedJudge
    print("✓ RuleBasedJudge import works")
except Exception as e:
    print(f"✗ RuleBasedJudge import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n--- All basic imports successful! ---")
print("Segfault is likely in the script logic, not imports.")
