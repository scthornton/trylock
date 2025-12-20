---
license: cc-by-nc-sa-4.0
base_model: mistralai/Mistral-7B-Instruct-v0.3
library_name: peft
tags:
  - llm-security
  - jailbreak-defense
  - dpo
  - safety
  - alignment
  - lora
pipeline_tag: text-generation
---

# TRYLOCK Mistral-7B DPO Adapter

Layer 1 of the TRYLOCK (Adaptive Ensemble Guard with Integrated Steering) defense system. This is a LoRA adapter trained with Direct Preference Optimization (DPO) to improve jailbreak resistance.

## Model Description

This adapter was trained on the TRYLOCK dataset to:
- Recognize and refuse jailbreak attempts
- Maintain helpful responses to legitimate queries
- Resist prompt injection and roleplay attacks

## Intended Uses

**Primary use cases:**
- Enhancing LLM safety against jailbreak attacks
- Research on preference-based alignment techniques
- Combining with RepE steering for defense-in-depth
- Evaluating robustness of safety training methods

**Out of scope:**
- Production deployment without additional safety testing
- Standalone safety solution (designed for layered defense)

### Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | Mistral-7B-Instruct-v0.3 |
| Method | DPO (Direct Preference Optimization) |
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Target Modules | q_proj, k_proj, v_proj, o_proj |
| Training Samples | 2,349 |
| Epochs | 3 |

### Evaluation Results

| Metric | Score |
|--------|-------|
| Baseline ASR | 46.5% |
| Post-DPO ASR | 39.8% |
| Relative ASR Reduction | 14.4% |

*Combined with Layer 2 (RepE α=2.0), full TRYLOCK achieves 8.0% ASR (82.8% relative reduction).*

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Load DPO adapter
model = PeftModel.from_pretrained(base_model, "scthornton/trylock-mistral-7b-dpo")

# Generate response
tokenizer = AutoTokenizer.from_pretrained("scthornton/trylock-mistral-7b-dpo")
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## TRYLOCK Architecture

This adapter is Layer 1 of the 3-layer TRYLOCK defense:

1. **Layer 1 (KNOWLEDGE)**: DPO fine-tuning for inherent safety (this model)
2. **Layer 2 (INSTINCT)**: RepE steering for activation-space defense
3. **Layer 3 (OVERSIGHT)**: Sidecar classifier for adaptive defense

For full defense, combine with:
- [trylock-repe-vectors](https://huggingface.co/scthornton/trylock-repe-vectors)
- [trylock-sidecar-classifier](https://huggingface.co/scthornton/trylock-sidecar-classifier)

## Limitations and Risks

**Limitations:**
- Trained on English-language attacks only
- 14.4% ASR reduction alone; requires RepE (Layer 2) for full 82.8% protection
- May reduce model fluency on edge cases
- Requires Mistral-7B-Instruct-v0.3 base model

**Risks:**
- Novel attack patterns not in training data may bypass defenses
- Over-refusal possible on legitimate but sensitive queries
- Should not be relied upon as sole safety mechanism

**Recommendations:**
- Combine with Layer 2 (RepE) and Layer 3 (Sidecar) for full protection
- Monitor for false positives in production
- Regularly update with new attack patterns

## Framework Versions

- PEFT 0.18.0
- TRL: 0.25.1
- Transformers: 4.57.3
- Pytorch: 2.7.1+cu128

## Citation

```bibtex
@misc{trylock2024,
  title={TRYLOCK: Defense-in-Depth Against LLM Jailbreaks via Layered Preference and Representation Engineering},
  author={Thornton, Scott},
  year={2024},
  url={https://huggingface.co/scthornton/trylock-mistral-7b-dpo}
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
