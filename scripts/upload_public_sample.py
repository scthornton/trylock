#!/usr/bin/env python3
"""
Upload the public sample dataset to HuggingFace.
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_sample():
    api = HfApi()
    repo_id = "scthornton/aegis-demo-dataset"

    print(f"Creating repository: {repo_id}")

    # Create repo (public)
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print(f"✅ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"Note: {e}")

    # Upload files
    print("\nUploading files...")

    sample_dir = Path("data/public_sample")

    # Upload the sample data
    api.upload_file(
        path_or_fileobj=str(sample_dir / "aegis_sample.jsonl"),
        path_in_repo="aegis_sample.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  ✅ aegis_sample.jsonl")

    # Upload README
    api.upload_file(
        path_or_fileobj=str(sample_dir / "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  ✅ README.md")

    # Upload LICENSE (copy from main repo)
    if Path("LICENSE").exists():
        api.upload_file(
            path_or_fileobj="LICENSE",
            path_in_repo="LICENSE",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("  ✅ LICENSE")

    print(f"\n🎉 Public sample dataset uploaded!")
    print(f"📁 View at: https://huggingface.co/datasets/{repo_id}")
    print(f"\nThis dataset is PUBLIC (48 examples)")
    print(f"Full dataset remains PRIVATE (2,939 examples)")

if __name__ == "__main__":
    upload_sample()
