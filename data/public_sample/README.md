---
language:
- en
license: apache-2.0
task_categories:
- text-generation
- text-classification
pretty_name: TRYLOCK Sample Dataset
size_categories:
- n<1K
tags:
- jailbreak
- defense
- preference-learning
- llm-security
- adversarial-robustness
---

# TRYLOCK Sample Dataset

This is a **sample dataset** containing 48 diverse examples from the TRYLOCK training corpus.

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
