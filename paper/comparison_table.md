# TRYLOCK Comparison to Prior Art

## Defense Systems Comparison

| System | Organization | Method | Layers | Dynamic | Open Source | ASR Reduction |
|--------|--------------|--------|--------|---------|-------------|---------------|
| **TRYLOCK** | perfecXion.ai | DPO + RepE + Classifier | 3 | ✅ Yes | ✅ Yes | **86%** |
| Llama Guard | Meta | External Classifier | 1 | ❌ No | ✅ Yes | ~40-50%* |
| NeMo Guardrails | NVIDIA | Rule-based + LLM | 1 | Partial | ✅ Yes | ~30-40%* |
| Constitutional AI | Anthropic | RLHF Training | 1 | ❌ No | ❌ No | ~50-60%* |
| OpenAI Moderation | OpenAI | External Classifier | 1 | ❌ No | ❌ No | ~40-50%* |
| Rebuff | LangChain | Prompt Analysis | 1 | ❌ No | ✅ Yes | ~20-30%* |

*Approximate values from published papers and community benchmarks. Direct comparison requires evaluation on identical datasets.

## Architecture Comparison

### Single-Layer Defenses

| Defense | Mechanism | Strengths | Weaknesses |
|---------|-----------|-----------|------------|
| **Safety Fine-tuning** | Weight modification | Deep integration | Bypassed by novel attacks |
| **External Classifiers** | Input/output filtering | No model changes needed | Latency overhead, false positives |
| **Rule-based Systems** | Pattern matching | Interpretable | Easy to circumvent |

### TRYLOCK Multi-Layer Approach

| Layer | Name | Mechanism | What It Catches |
|-------|------|-----------|-----------------|
| 1 | KNOWLEDGE | DPO fine-tuning | Attacks seen during training |
| 2 | INSTINCT | RepE activation steering | Novel attack patterns |
| 3 | OVERSIGHT | Sidecar classification | Adaptive response strength |

## Technical Specifications

### TRYLOCK Components

| Component | Base Model | Size | Method | Inference Overhead |
|-----------|------------|------|--------|-------------------|
| DPO Adapter | Mistral-7B | 164 MB | LoRA r=32 | ~0% (merged) |
| RepE Vectors | - | 76 KB | 8 layers × 4096 | ~5% per forward |
| Sidecar | Qwen2.5-3B | 71 MB | LoRA r=32 | +3B inference |

### Llama Guard Specifications

| Version | Base Model | Size | Classes | Languages |
|---------|------------|------|---------|-----------|
| Llama Guard 1 | Llama-2-7B | 13 GB | 6 harmful categories | English |
| Llama Guard 2 | Llama-3-8B | 16 GB | 11 categories | English |
| Llama Guard 3 | Llama-3.1-8B | 16 GB | 13 categories | 8 languages |

## Feature Comparison

| Feature | TRYLOCK | Llama Guard | NeMo Guardrails |
|---------|-------|-------------|-----------------|
| Weight-level protection | ✅ | ❌ | ❌ |
| Activation-space defense | ✅ | ❌ | ❌ |
| Input classification | ✅ | ✅ | ✅ |
| Output classification | ❌ | ✅ | ✅ |
| Dynamic strength adjustment | ✅ | ❌ | Partial |
| Works offline | ✅ | ✅ | Partial |
| Multi-turn context | ✅ | ✅ | ✅ |
| Custom categories | ✅ | ❌ | ✅ |
| Base model agnostic | ❌* | ✅ | ✅ |

*TRYLOCK components are model-specific (Mistral-7B). Porting to other models requires retraining.

## Evaluation Results Summary

### TRYLOCK on Internal Benchmark (299 samples)

| Configuration | ASR | Reduction |
|---------------|-----|-----------|
| Baseline (Mistral-7B) | 58% | - |
| + Layer 1 (DPO) | 47% | 19% |
| + Layer 2 (RepE α=2.0) | 10% | 83% |
| Full TRYLOCK (adaptive α) | 8% | 86% |

### Comparison Notes

**Why TRYLOCK Outperforms Single-Layer Defenses:**

1. **Complementary Mechanisms**: Each layer catches different attack patterns
   - DPO: Attacks similar to training data
   - RepE: Novel patterns via activation-space steering
   - Sidecar: Enables appropriate response calibration

2. **Defense-in-Depth**: Attackers must bypass ALL layers, not just one

3. **Adaptive Response**: Strong defense only when needed, preserving UX on benign queries

**Limitations vs. Llama Guard:**

1. TRYLOCK is model-specific; Llama Guard works with any model
2. Llama Guard provides output classification; TRYLOCK focuses on input defense
3. Llama Guard has multi-language support; TRYLOCK is English-only

## Recommended Use Cases

| Use Case | Best Choice | Reason |
|----------|-------------|--------|
| Maximum jailbreak resistance | TRYLOCK | 86% ASR reduction |
| Model-agnostic deployment | Llama Guard | Works with any LLM |
| Complex rule requirements | NeMo Guardrails | Programmable policies |
| Low-latency applications | TRYLOCK (no sidecar) | Minimal overhead |
| Multi-language support | Llama Guard 3 | 8 language support |

## Citation

```bibtex
@misc{trylock2024,
  title={TRYLOCK: Defense-in-Depth Against LLM Jailbreaks via Layered Preference and Representation Engineering},
  author={Thornton, Scott},
  year={2024},
  organization={perfecXion.ai},
  url={https://huggingface.co/scthornton}
}
```

## References

- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench) - NeurIPS 2024
- [HarmBench](https://github.com/centerforaisafety/HarmBench) - Center for AI Safety
- [Llama Guard](https://ai.meta.com/research/publications/llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations/) - Meta AI
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) - NVIDIA
- [Representation Engineering](https://arxiv.org/abs/2310.01405) - Zou et al.
