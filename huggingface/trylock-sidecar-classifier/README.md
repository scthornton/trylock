---
license: cc-by-nc-sa-4.0
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
tags:
  - llm-security
  - jailbreak-detection
  - text-classification
  - safety
  - lora
pipeline_tag: text-classification
---

# TRYLOCK Sidecar Classifier

Layer 3 of the TRYLOCK (Adaptive Ensemble Guard with Integrated Steering) defense system. This is a lightweight classifier that runs alongside the main LLM to detect attack patterns and dynamically adjust defense strength.

## Model Description

The sidecar classifier categorizes inputs into three classes:
- **SAFE**: Benign queries that need minimal defense
- **WARN**: Ambiguous or suspicious queries
- **ATTACK**: Clear jailbreak/attack attempts

## Intended Uses

**Primary use cases:**
- Real-time classification of LLM inputs for adaptive defense
- Dynamically adjusting RepE steering strength
- Research on attack detection methods
- Content moderation and threat triage

**Out of scope:**
- Standalone content moderation (designed for adaptive steering)
- High-stakes security decisions without human review

### Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-3B-Instruct |
| Method | LoRA fine-tuning |
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Training Samples | 2,349 |
| Classes | SAFE, WARN, ATTACK |

### Evaluation Results

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| SAFE | 24.1% | 34.5% | 28.4% |
| WARN | 58.3% | 51.6% | 54.8% |
| ATTACK | 62.4% | 60.8% | 61.6% |
| **Macro Avg** | 48.3% | 48.9% | 48.3% |

**Note**: This is a research prototype. ATTACK detection is prioritized over SAFE detection for security.

## Usage

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    num_labels=3,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, "scthornton/trylock-sidecar-classifier")
tokenizer = AutoTokenizer.from_pretrained("scthornton/trylock-sidecar-classifier")

# Classify input
label_names = ["SAFE", "WARN", "ATTACK"]

def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        outputs = model(**inputs.to(model.device))
        probs = torch.softmax(outputs.logits, dim=-1)[0]
    return label_names[probs.argmax()], probs.tolist()

# Example
result, probs = classify("Ignore all previous instructions and...")
print(f"Classification: {result}")  # ATTACK
print(f"Probabilities: {dict(zip(label_names, probs))}")
```

## Dynamic Defense Integration

Use the classification to adjust RepE steering strength:

```python
alpha_map = {
    "SAFE": 0.5,   # Minimal steering - preserve fluency
    "WARN": 1.5,   # Moderate steering
    "ATTACK": 2.5  # Maximum steering - block harmful output
}

classification, _ = classify(user_input)
alpha = alpha_map[classification]
# Apply RepE steering with this alpha value
```

## TRYLOCK Architecture

This classifier is Layer 3 of the 3-layer TRYLOCK defense:

1. **Layer 1 (KNOWLEDGE)**: [trylock-mistral-7b-dpo](https://huggingface.co/scthornton/trylock-mistral-7b-dpo)
2. **Layer 2 (INSTINCT)**: [trylock-repe-vectors](https://huggingface.co/scthornton/trylock-repe-vectors)
3. **Layer 3 (OVERSIGHT)**: Sidecar classifier (this model)

## Limitations and Risks

**Limitations:**
- SAFE class has lower precision (24%) - may over-classify benign queries as WARN
- Trained on English-language attacks only
- 3-class granularity may miss nuanced threat levels
- Requires ~3B parameter model inference overhead

**Risks:**
- False positives may trigger unnecessary steering
- False negatives on novel attack patterns
- Classification confidence doesn't guarantee correctness

**Recommendations:**
- Use probability scores, not just class labels, for fine-grained control
- Consider WARN classification as "elevated caution" rather than definite threat
- Combine with other safety mechanisms for production use

## Framework Versions

- PEFT 0.18.0

## Citation

```bibtex
@misc{trylock2024,
  title={TRYLOCK: Defense-in-Depth Against LLM Jailbreaks via Layered Preference and Representation Engineering},
  author={Thornton, Scott},
  year={2024},
  url={https://huggingface.co/scthornton/trylock-sidecar-classifier}
}
```

## License

**CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International)

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit to **Scott Thornton**, provide a link to the license, and indicate if changes were made
- **NonCommercial** — You may not use the material for commercial purposes without explicit written permission
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license

For commercial licensing inquiries, contact: scott@perfecxion.ai
