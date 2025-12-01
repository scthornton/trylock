#!/usr/bin/env python3
"""
Test imports one by one to find the segfault cause.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys

def test_import(num, desc, code):
    print(f"\n{'='*50}")
    print(f"TEST {num}: {desc}")
    print('='*50)
    exec(code, globals())
    print("✓ Success")

try:
    test_import(1, "Import torch", "import torch")
    test_import(2, "Import transformers", "import transformers")
    test_import(3, "Import AutoTokenizer", "from transformers import AutoTokenizer")
    test_import(4, "Import AutoModelForCausalLM", "from transformers import AutoModelForCausalLM")
    test_import(5, "Import AutoModelForSequenceClassification", "from transformers import AutoModelForSequenceClassification")
    test_import(6, "Import peft", "import peft")
    test_import(7, "Import PeftModel", "from peft import PeftModel")
    test_import(8, "Import safetensors", "import safetensors")
    test_import(9, "Import load_file", "from safetensors.torch import load_file")
    test_import(10, "Import huggingface_hub", "from huggingface_hub import hf_hub_download")

    print(f"\n{'='*50}")
    print("ALL IMPORTS SUCCESSFUL!")
    print('='*50)

except Exception as e:
    print(f"\n❌ FAILED!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
