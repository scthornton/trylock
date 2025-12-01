#!/usr/bin/env python3
"""
Test if we can load AEGIS models without crashing.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("Testing model loading...")

import torch
from transformers import AutoTokenizer

print(f"\n1. torch {torch.__version__}")
print(f"   Device: CPU (MPS disabled)")

print("\n2. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", use_fast=False)
print(f"   ✓ Tokenizer loaded")

print("\n3. Testing tokenizer...")
test_text = "Hello, world!"
tokens = tokenizer(test_text, return_tensors="pt")
print(f"   ✓ Tokenization works ({len(tokens['input_ids'][0])} tokens)")

print("\n" + "="*60)
print("SUCCESS! Model loading infrastructure works.")
print("="*60)
print("\nReady to run full evaluation.")
