#!/usr/bin/env python3
"""
Create a public sample dataset (50 examples) for demonstration purposes.
Keeps the full dataset private while showing format and variety.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

def create_sample_dataset():
    """Extract diverse samples across attack families."""

    # Load full training data
    train_data = []
    with open("data/dpo/train.jsonl") as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line))

    print(f"Loaded {len(train_data)} training examples")

    # Group by attack family
    by_family = defaultdict(list)
    for item in train_data:
        family = item.get("metadata", {}).get("family", "unknown")
        by_family[family].append(item)

    print(f"\nAttack families found: {len(by_family)}")
    for family, items in sorted(by_family.items()):
        print(f"  {family}: {len(items)} examples")

    # Sample ~8 from each family (total ~50)
    samples = []
    for family, items in sorted(by_family.items()):
        # Take 8 examples from each family
        n = min(8, len(items))
        family_samples = random.sample(items, n)
        samples.extend(family_samples)

    print(f"\nCreated sample dataset: {len(samples)} examples")

    # Shuffle to mix families
    random.shuffle(samples)

    # Save public sample
    output_dir = Path("data/public_sample")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "trylock_sample.jsonl"
    with open(output_file, "w") as f:
        for item in samples:
            f.write(json.dumps(item) + "\n")

    print(f"Saved to: {output_file}")

    # Create README
    readme = output_dir / "README.md"
    with open(readme, "w") as f:
        f.write("""# TRYLOCK Sample Dataset

This is a **sample dataset** containing 50 diverse examples from the TRYLOCK training corpus.

## Purpose

This sample demonstrates:
- Data format and structure
- Attack family diversity
- Preference pair quality
- Multi-turn conversation patterns

## Full Dataset

The complete TRYLOCK dataset (2,939 preference pairs) is **private** to protect intellectual property.

## Data Format

Each example contains:
- `id`: Unique identifier
- `prompt`: Multi-turn conversation leading to jailbreak attempt
- `chosen`: Correct refusal response
- `rejected`: Compliant (unsafe) response
- `metadata`: Attack family, difficulty, turn information

## Attack Families Included

This sample includes examples from:
- Direct injection (sudo mode, developer mode, etc.)
- Multi-turn manipulation
- Obfuscation wrappers
- Tool/agent abuse
- Indirect injection
- Benign hard negatives
- And more...

## Usage

```python
import json

# Load samples
samples = []
with open("trylock_sample.jsonl") as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} examples")

# Inspect structure
example = samples[0]
print(f"Prompt turns: {len(example['prompt'])}")
print(f"Attack family: {example['metadata']['family']}")
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{thornton2025trylock,
  title={TRYLOCK: Adaptive LLM Jailbreak Defense via Layered Security Architecture},
  author={Thornton, Scott},
  year={2025}
}
```

## License

This sample dataset is released under the same license as the TRYLOCK models.

## Full Models

The trained TRYLOCK models are publicly available:
- DPO Adapter: `scthornton/trylock-mistral-7b-dpo`
- RepE Vectors: `scthornton/trylock-repe-vectors`
- Sidecar Classifier: `scthornton/trylock-sidecar-classifier`

## Contact

For questions or access to the full dataset for academic research, please contact Scott Thornton.
""")

    print(f"Created README: {readme}")

    return output_file, len(samples)

if __name__ == "__main__":
    random.seed(42)  # Reproducible sampling
    output_file, n = create_sample_dataset()
    print(f"\n✅ Public sample dataset created: {n} examples")
    print(f"📁 Location: {output_file}")
    print("\nNext: Upload to HuggingFace as scthornton/trylock-demo-dataset")
