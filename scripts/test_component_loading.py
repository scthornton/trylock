#!/usr/bin/env python3
"""
Test each TRYLOCK component individually to find what causes segfault.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys

def test_step(num, desc):
    print(f"\n{'='*60}")
    print(f"TEST {num}: {desc}")
    print('='*60)

try:
    test_step(1, "Import libraries")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    from peft import PeftModel
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    print("✓ All imports successful")

    test_step(2, "Load base tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", use_fast=False)
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

    test_step(3, "Load base model (this will take 5-10 minutes)")
    print("   Downloading ~14GB model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    print(f"✓ Base model loaded")

    test_step(4, "Load DPO adapter")
    dpo_model = PeftModel.from_pretrained(
        base_model,
        "scthornton/trylock-mistral-7b-dpo",
        device_map={"": "cpu"},
    )
    print(f"✓ DPO adapter loaded")

    test_step(5, "Load RepE vectors")
    repe_path = hf_hub_download(
        repo_id="scthornton/trylock-repe-vectors",
        filename="steering_vectors.safetensors",
        repo_type="model"
    )
    vectors = load_file(repe_path)
    print(f"✓ RepE vectors loaded ({len(vectors)} vectors)")

    test_step(6, "Load sidecar classifier")
    sidecar_tokenizer = AutoTokenizer.from_pretrained("scthornton/trylock-sidecar-classifier")
    sidecar_model = AutoModelForSequenceClassification.from_pretrained(
        "scthornton/trylock-sidecar-classifier",
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
    )
    print(f"✓ Sidecar loaded")

    print(f"\n{'='*60}")
    print("ALL COMPONENTS LOADED SUCCESSFULLY!")
    print('='*60)

except Exception as e:
    print(f"\n❌ FAILED at current test!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
