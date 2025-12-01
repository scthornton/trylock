#!/usr/bin/env python3
"""
Make the AEGIS dataset private to protect intellectual property.
"""

from huggingface_hub import HfApi

def make_private():
    api = HfApi()

    print("Making aegis-dataset private...")

    try:
        api.update_repo_visibility(
            repo_id='scthornton/aegis-dataset',
            private=True,
            repo_type='dataset'
        )
        print("✅ Dataset is now PRIVATE")
        print("\nVerifying...")

        # Verify it's private
        info = api.dataset_info('scthornton/aegis-dataset')
        if info.private:
            print("✅ Confirmed: Dataset is private")
        else:
            print("⚠️ Warning: Dataset may still be public")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease make it private manually:")
        print("1. Go to: https://huggingface.co/datasets/scthornton/aegis-dataset/settings")
        print("2. Scroll to 'Danger Zone'")
        print("3. Click 'Change visibility'")
        print("4. Select 'Private'")

if __name__ == "__main__":
    make_private()
