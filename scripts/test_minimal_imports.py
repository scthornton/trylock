#!/usr/bin/env python3
"""
Minimal import test matching what worked in test_model_loading.py
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("Starting import tests...")

import torch
print("✓ torch imported")

from transformers import AutoTokenizer
print("✓ AutoTokenizer imported")

from transformers import AutoModelForCausalLM
print("✓ AutoModelForCausalLM imported")

from transformers import AutoModelForSequenceClassification
print("✓ AutoModelForSequenceClassification imported")

import peft
print(f"✓ peft imported (version {peft.__version__})")

from peft import PeftModel
print("✓ PeftModel imported")

from safetensors.torch import load_file
print("✓ load_file imported")

from huggingface_hub import hf_hub_download
print("✓ hf_hub_download imported")

print("\n" + "="*60)
print("ALL IMPORTS SUCCESSFUL!")
print("="*60)
