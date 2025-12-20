---
title: TRYLOCK Defense Demo
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: cc-by-nc-sa-4.0
tags:
  - llm-security
  - jailbreak-defense
  - ai-safety
  - representation-engineering
  - dpo
---

# 🛡️ TRYLOCK: Defense-in-Depth Against LLM Jailbreaks

**Adaptive Ensemble Guard with Integrated Steering**

## Overview

TRYLOCK is a three-layer defense system that achieves **82.8% reduction** in Attack Success Rate against LLM jailbreak attacks. This demo allows you to interactively test each defense layer.

## Defense Layers

| Layer | Name | Mechanism | Protection |
|-------|------|-----------|------------|
| 1 | **KNOWLEDGE** | DPO fine-tuning | Embeds safety preferences in weights |
| 2 | **INSTINCT** | RepE steering | Runtime activation-space protection |
| 3 | **OVERSIGHT** | Sidecar classifier | Adaptive defense strength |

## How to Use

1. **Load Models**: Click "Load TRYLOCK Models" to initialize all three layers
2. **Configure Layers**: Toggle individual layers on/off to see their effects
3. **Select Example**: Choose from preset attack prompts or write your own
4. **Generate**: Click "Generate Response" to see TRYLOCK in action

## Attack Categories Covered

- **Direct Attacks**: Explicit harmful requests
- **Roleplay/Persona**: DAN, UCAR, character jailbreaks
- **Prompt Injection**: System prompt manipulation
- **Encoding Tricks**: Base64, ROT13, leetspeak
- **Obfuscation**: Typos, word substitution

## Threat Classification

The sidecar classifier (Layer 3) categorizes inputs into three threat levels:

- 🟢 **SAFE** (α=0.5): Benign queries - light steering
- 🟡 **WARN** (α=1.5): Ambiguous content - moderate steering
- 🔴 **ATTACK** (α=2.5): Clear jailbreak - strong steering

## Results

| Configuration | ASR | Reduction |
|---------------|-----|-----------|
| Baseline | 46.5% | - |
| + Layer 1 (DPO) | 39.8% | 14.4% |
| + Layer 2 (RepE α=2.0) | 8.0% | 82.8% |
| Full TRYLOCK (adaptive) | 8.0% | **82.8%** |

## Resources

- **Paper**: [TRYLOCK: Defense-in-Depth Against LLM Jailbreaks](https://arxiv.org/abs/XXXX.XXXXX)
- **Dataset**: [scthornton/trylock-dataset](https://huggingface.co/datasets/scthornton/trylock-dataset)
- **DPO Adapter**: [scthornton/trylock-mistral-7b-dpo](https://huggingface.co/scthornton/trylock-mistral-7b-dpo)
- **RepE Vectors**: [scthornton/trylock-repe-vectors](https://huggingface.co/scthornton/trylock-repe-vectors)
- **Sidecar**: [scthornton/trylock-sidecar-classifier](https://huggingface.co/scthornton/trylock-sidecar-classifier)

## Citation

```bibtex
@misc{trylock2024,
  title={TRYLOCK: Defense-in-Depth Against LLM Jailbreaks via Layered Preference and Representation Engineering},
  author={Thornton, Scott},
  year={2024},
  organization={perfecXion.ai}
}
```

## License

CC BY-NC-SA 4.0 - Attribution required, non-commercial use only.

**Contact**: scott@perfecxion.ai

---

*Built by [perfecXion.ai](https://perfecxion.ai)*
