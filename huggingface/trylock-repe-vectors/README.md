---
license: cc-by-nc-sa-4.0
tags:
  - llm-security
  - jailbreak-defense
  - representation-engineering
  - repe
  - steering-vectors
  - safety
---

# TRYLOCK RepE Steering Vectors

Layer 2 of the TRYLOCK (Adaptive Ensemble Guard with Integrated Steering) defense system. These are Representation Engineering (RepE) steering vectors that operate in the model's activation space to enforce safety.

## Description

Unlike fine-tuning which modifies model weights, RepE steering works by adding safety-inducing directions to the model's hidden states during inference. This provides:

- **Complementary defense** to weight-based methods
- **Dynamic control** via adjustable steering strength (alpha)
- **No training required** - works at inference time

## Intended Uses

**Primary use cases:**
- Runtime safety enforcement for LLM inference
- Research on representation engineering techniques
- Combining with DPO training for defense-in-depth
- Dynamic defense adjustment based on threat level

**Out of scope:**
- Models other than Mistral-7B-Instruct-v0.3 (vectors are model-specific)
- Standalone safety solution (best combined with other layers)

### Vector Details

| Parameter | Value |
|-----------|-------|
| Base Model | Mistral-7B-Instruct-v0.3 |
| Method | Contrastive Activation Addition (CAA) |
| Layers | 12, 14, 16, 18, 20, 22, 24, 26 |
| Vector Dimension | 4096 |
| Optimal Alpha | 2.0 |

### Evaluation Results

Results with DPO (Layer 1) active:

| Alpha | ASR | Analysis |
|-------|-----|----------|
| 0.0 | 39.8% | DPO only (no steering) |
| 1.0 | 32.1% | Moderate steering |
| 2.0 | 8.0% | **Optimal trade-off** |
| 2.5 | 0.0% | Maximum protection (high over-refusal) |

*Full TRYLOCK (DPO + RepE α=2.0) achieves 82.8% relative ASR reduction from 46.5% baseline.*

## Usage

```python
import torch
from safetensors.torch import load_file
import json

# Load steering vectors
vectors = load_file("steering_vectors.safetensors")

# Load config
with open("repe_config.json") as f:
    config = json.load(f)

# Apply steering during forward pass
def steering_hook(layer_idx, alpha=2.0):
    vector = vectors[f"layer_{layer_idx}"]
    def hook(module, input, output):
        # Add steering vector to hidden states
        hidden_states = output[0]
        hidden_states = hidden_states + alpha * vector.to(hidden_states.device)
        return (hidden_states,) + output[1:]
    return hook

# Register hooks on model
for layer_idx in config["steering_layers"]:
    model.model.layers[layer_idx].register_forward_hook(
        steering_hook(layer_idx, alpha=2.0)
    )
```

## Dynamic Alpha with Sidecar

For optimal defense, use with the sidecar classifier to dynamically adjust alpha:

```python
# Sidecar classifies input as SAFE/WARN/ATTACK
alpha_map = {"SAFE": 0.5, "WARN": 1.5, "ATTACK": 2.5}
classification = sidecar.classify(prompt)
alpha = alpha_map[classification]
```

## TRYLOCK Architecture

These vectors are Layer 2 of the 3-layer TRYLOCK defense:

1. **Layer 1 (KNOWLEDGE)**: [trylock-mistral-7b-dpo](https://huggingface.co/scthornton/trylock-mistral-7b-dpo)
2. **Layer 2 (INSTINCT)**: RepE steering vectors (this model)
3. **Layer 3 (OVERSIGHT)**: [trylock-sidecar-classifier](https://huggingface.co/scthornton/trylock-sidecar-classifier)

## Limitations and Risks

**Limitations:**
- Vectors are specific to Mistral-7B-Instruct-v0.3 architecture
- High alpha values (>2.5) may degrade output quality
- Steering affects all outputs, not just harmful ones
- Requires careful alpha tuning for optimal balance

**Risks:**
- Over-steering can cause repetitive or incoherent outputs
- May interfere with legitimate use cases at high alpha
- Novel attacks may find directions orthogonal to steering vectors

**Recommendations:**
- Use dynamic alpha based on sidecar classification
- Start with alpha=1.0 and adjust based on use case
- Combine with DPO training for robust defense

## Citation

```bibtex
@misc{trylock2024,
  title={TRYLOCK: Defense-in-Depth Against LLM Jailbreaks via Layered Preference and Representation Engineering},
  author={Thornton, Scott},
  year={2024},
  url={https://huggingface.co/scthornton/trylock-repe-vectors}
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
