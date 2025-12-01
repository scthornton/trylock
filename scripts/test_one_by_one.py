#!/usr/bin/env python3
"""
Import one thing at a time to find the culprit
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("TEST 1: Import torch")
import torch
print("✓ torch imported")

print("\nTEST 2: Import transformers")
import transformers
print("✓ transformers imported")

print("\nTEST 3: Import AutoTokenizer")
from transformers import AutoTokenizer
print("✓ AutoTokenizer imported")

print("\nTEST 4: Import AutoModelForCausalLM")
from transformers import AutoModelForCausalLM
print("✓ AutoModelForCausalLM imported")

print("\nTEST 5: Import AutoModelForSequenceClassification")
from transformers import AutoModelForSequenceClassification
print("✓ AutoModelForSequenceClassification imported")

print("\nTEST 6: Import peft")
import peft
print(f"✓ peft imported (version {peft.__version__})")

print("\nSUCCESS - all tests passed!")
