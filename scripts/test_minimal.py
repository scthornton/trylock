#!/usr/bin/env python3
"""Minimal test - import one thing at a time."""

print("1. Starting Python...")

print("2. Importing sys...")
import sys
print(f"   Python: {sys.version}")

print("3. Importing torch...")
import torch
print(f"   ✓ torch {torch.__version__}")

print("4. Testing CUDA...")
print(f"   CUDA: {torch.cuda.is_available()}")

print("5. All done!")
