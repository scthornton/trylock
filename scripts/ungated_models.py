#!/usr/bin/env python3
"""
Remove gating from TRYLOCK models.
"""

from huggingface_hub import HfApi
import sys

def ungated_models():
    api = HfApi()

    models = [
        'scthornton/trylock-mistral-7b-dpo',
        'scthornton/trylock-repe-vectors',
        'scthornton/trylock-sidecar-classifier'
    ]

    print("This script will guide you to ungated the models manually.")
    print("\nFor each model, you need to:")
    print("1. Go to the settings page")
    print("2. Find 'Access control' section")
    print("3. Change from 'Gated' to 'Public'")
    print("4. Click 'Save changes'")
    print("\n" + "="*60)

    for model_id in models:
        settings_url = f"https://huggingface.co/{model_id}/settings"
        print(f"\n📝 {model_id}")
        print(f"   Settings: {settings_url}")
        input("   Press Enter after you've ungated this model...")

    print("\n" + "="*60)
    print("✅ All done! Models should now be accessible.")
    print("\nYou can verify at:")
    for model_id in models:
        print(f"  - https://huggingface.co/{model_id}")

if __name__ == "__main__":
    ungated_models()
