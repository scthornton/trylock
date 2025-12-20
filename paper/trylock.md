# TRYLOCK: Defense-in-Depth Against LLM Jailbreaks via Layered Preference and Representation Engineering

**Author:** Scott Thornton
**Organization:** perfecXion.ai
**Contact:** scott@perfecxion.ai
**Date:** December 2025

---

## Abstract

Large Language Models (LLMs) remain vulnerable to jailbreak attacks that bypass safety alignment through prompt manipulation, roleplay scenarios, and encoding tricks. Existing defenses typically operate at a single level—either modifying model weights through fine-tuning or applying external guardrails—leaving gaps that sophisticated attacks can exploit. We introduce TRYLOCK (Adaptive Ensemble Guard with Integrated Steering), a three-layer defense-in-depth architecture that combines complementary protection mechanisms: (1) Direct Preference Optimization (DPO) for embedding safety preferences into model weights, (2) Representation Engineering (RepE) for runtime activation-space steering, and (3) a lightweight sidecar classifier for adaptive defense strength adjustment. On our evaluation dataset of 299 attack prompts, TRYLOCK reduces Attack Success Rate (ASR) from 58% (baseline) to 8% (with all layers active), representing an 86% relative reduction. Critically, each layer provides independent, complementary protection: DPO alone achieves 19% ASR reduction, RepE steering adds 79% reduction on top of DPO, and the sidecar enables dynamic adjustment to minimize false positives on benign queries. We release all components—trained adapters, steering vectors, classifier, and training data—to enable reproducible research on layered LLM safety.

---

## 1. Introduction

Despite significant advances in safety alignment, Large Language Models remain susceptible to jailbreak attacks—adversarial prompts designed to elicit harmful, unethical, or dangerous outputs that violate the model's safety guidelines [1, 2]. These attacks exploit various vulnerabilities: prompt injection overwrites system instructions, roleplay scenarios create fictional contexts where safety rules "don't apply," and encoding tricks (Base64, ROT13, leetspeak) obfuscate malicious intent from safety classifiers.

Current defenses fall into two broad categories: **weight-based methods** that modify the model's parameters through safety fine-tuning [3, 4], and **inference-time methods** that filter or modify inputs/outputs without changing weights [5, 6]. Each approach has limitations. Weight-based methods can be circumvented by attacks not seen during training, while inference-time filters often create false positives that degrade user experience on legitimate queries.

We argue that robust LLM safety requires **defense-in-depth**—multiple independent layers that each catch different attack patterns. Just as network security employs firewalls, intrusion detection, and endpoint protection in concert, LLM safety should combine complementary mechanisms operating at different levels of the inference stack.

### 1.1 Contributions

We present TRYLOCK, a three-layer defense architecture with the following contributions:

1. **Layer 1 (KNOWLEDGE)**: A DPO-trained LoRA adapter that embeds safety preferences directly into model weights, achieving 19% ASR reduction over baseline.

2. **Layer 2 (INSTINCT)**: Representation Engineering steering vectors that operate in activation space during inference, providing 79% additional ASR reduction when combined with Layer 1.

3. **Layer 3 (OVERSIGHT)**: A lightweight sidecar classifier (3B parameters) that dynamically adjusts steering strength based on input threat level, enabling strong defense against attacks while minimizing impact on benign queries.

4. **Open Release**: We release all trained components, training data (2,939 samples), and evaluation code under CC BY-NC-SA 4.0 license to enable reproducible research.

---

## 2. Related Work

### 2.1 Jailbreak Attacks

Jailbreak attacks have evolved rapidly alongside LLM capabilities. Early attacks relied on simple prompt manipulation ("ignore previous instructions"), but modern attacks employ sophisticated techniques including:

**Roleplay/Persona Attacks**: The DAN (Do Anything Now) family of attacks [7] creates fictional AI personas claimed to operate without safety restrictions. Variants include UCAR, STAN, and character-based jailbreaks.

**Encoding Attacks**: Attackers encode harmful requests in Base64, ROT13, pig Latin, or custom ciphers, exploiting the gap between tokenization and semantic understanding [1].

**Gradient-Based Attacks**: GCG [2] uses gradient optimization to find adversarial suffixes that cause models to comply with harmful requests. These attacks transfer across models but require white-box access.

**Multi-Turn Attacks**: Crescendo [8] and similar attacks gradually build context across conversation turns, each individually benign, that culminates in harmful compliance.

### 2.2 Defense Methods

**Safety Fine-Tuning**: Constitutional AI [3] and RLHF [4] train models to refuse harmful requests. DPO [9] simplifies this by directly optimizing on preference pairs without a reward model.

**External Guardrails**: Llama Guard [5] uses a separate classifier to filter inputs/outputs. NeMo Guardrails [6] implements programmable safety rails. These add latency and can create false positives.

**Representation Engineering**: RepE [10] demonstrates that safety-relevant directions exist in activation space and can be used to steer model behavior at inference time.

**Our Contribution**: TRYLOCK uniquely combines all three approaches—weight modification (DPO), activation steering (RepE), and external classification (sidecar)—in a unified architecture where each layer provides independent, complementary protection.

---

## 3. Method

### 3.1 Architecture Overview

TRYLOCK implements defense-in-depth through three complementary layers, each operating at a different level of the inference stack.

```
TRYLOCK Architecture
==================

Input → [Layer 3: Sidecar] → α (steering strength)
              ↓
[Layer 1: DPO Weights] + [Layer 2: RepE(α)] → Output
```

**Figure 1**: TRYLOCK three-layer architecture. The sidecar classifier (Layer 3) analyzes input to determine threat level and sets steering strength α for RepE (Layer 2). DPO-trained weights (Layer 1) provide baseline safety.

### 3.2 Layer 1: DPO Safety Training

We train a LoRA adapter [11] on Mistral-7B-Instruct-v0.3 using Direct Preference Optimization. DPO directly optimizes the policy to prefer safe responses over unsafe ones:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

where $y_w$ is the preferred (safe) response and $y_l$ is the dispreferred (unsafe) response.

**Training Details**:
- Base model: Mistral-7B-Instruct-v0.3
- LoRA rank: 32, alpha: 64
- Target modules: q_proj, k_proj, v_proj, o_proj
- Training samples: 2,349
- Epochs: 3

### 3.3 Layer 2: Representation Engineering Steering

Unlike DPO which modifies weights, RepE steering operates in activation space during inference. We compute contrastive steering vectors by collecting activations on paired safe/unsafe prompts:

$$\mathbf{v}_{\text{safety}}^{(l)} = \mathbb{E}[\mathbf{h}^{(l)}_{\text{safe}}] - \mathbb{E}[\mathbf{h}^{(l)}_{\text{unsafe}}]$$

where $\mathbf{h}^{(l)}$ are hidden states at layer $l$. During inference, we add the steering vector scaled by α:

$$\mathbf{h}'^{(l)} = \mathbf{h}^{(l)} + \alpha \cdot \mathbf{v}_{\text{safety}}^{(l)}$$

**Steering Details**:
- Layers: 12, 14, 16, 18, 20, 22, 24, 26 (middle-to-late layers)
- Vector dimension: 4096
- Optimal α: 2.0 (evaluated range: 0.0–3.0)

### 3.4 Layer 3: Sidecar Classifier

The sidecar is a lightweight classifier that runs in parallel with the main model to categorize inputs into three threat levels:

- **SAFE**: Benign queries requiring minimal defense (α = 0.5)
- **WARN**: Ambiguous or potentially suspicious queries (α = 1.5)
- **ATTACK**: Clear jailbreak attempts (α = 2.5)

This enables **adaptive defense**: strong steering is only applied when attacks are detected, preserving fluency and helpfulness on benign queries.

**Training Details**:
- Base model: Qwen2.5-3B-Instruct
- Method: LoRA fine-tuning for sequence classification
- LoRA rank: 32, alpha: 64
- Classes: 3 (SAFE, WARN, ATTACK)
- Training samples: 2,349

---

## 4. Dataset

We curate a dataset of 2,939 preference pairs covering major jailbreak attack families:

| Attack Category | Count | Example Pattern |
|-----------------|-------|-----------------|
| Direct Attacks | 412 | Explicit harmful requests |
| Roleplay/Persona | 687 | DAN, UCAR, character jailbreaks |
| Prompt Injection | 523 | System prompt manipulation |
| Encoding Tricks | 398 | Base64, ROT13, leetspeak |
| Multi-Turn Buildup | 291 | Context escalation attacks |
| Obfuscation | 628 | Typos, word substitution |
| **Total** | **2,939** | |

**Table 1**: Attack category distribution in TRYLOCK dataset.

Each sample contains:
- `prompt`: The attack prompt
- `chosen`: A safe refusal response (preferred)
- `rejected`: An unsafe compliant response (dispreferred)

**Data Splits**: train (2,349), validation (291), test (299).

---

## 5. Experiments

### 5.1 Evaluation Metrics

**Attack Success Rate (ASR)**: The percentage of attack prompts for which the model produces a harmful response. Lower is better.

**Judge Models**: We use an ensemble of three judge approaches:
1. Pattern matching for explicit compliance markers
2. Claude-based semantic evaluation
3. Keyword detection for harmful content categories

A response is considered "successful attack" if any judge flags it as harmful.

### 5.2 Baseline and Ablations

We evaluate each layer independently and in combination:

| Configuration | ASR | Δ from Baseline | Relative Reduction |
|---------------|-----|-----------------|-------------------|
| Baseline (Mistral-7B) | 58% | — | — |
| + Layer 1 (DPO) | 47% | -11% | 19% |
| + Layer 2 (RepE α=1.0) | 36% | -22% | 38% |
| + Layer 2 (RepE α=2.0) | 10% | -48% | 83% |
| + Layer 2 (RepE α=2.5) | 8% | -50% | 86% |
| **Full TRYLOCK (adaptive α)** | **8%** | **-50%** | **86%** |

**Table 2**: Attack Success Rate across configurations. Each layer provides independent, additive protection.

### 5.3 Layer Independence

A critical finding is that each layer provides **independent** protection against different attack patterns:

- **DPO** is most effective against direct attacks and roleplay scenarios seen during training
- **RepE** generalizes better to novel attack patterns and encoding tricks
- **Sidecar** enables appropriate response—strong defense on attacks, light touch on benign queries

### 5.4 Sidecar Classification Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| SAFE | 24% | 35% | 28% |
| WARN | 66% | 40% | 50% |
| ATTACK | 62% | 61% | 62% |

**Table 3**: Sidecar classifier performance. ATTACK detection is prioritized; SAFE misclassification triggers stronger (but not harmful) defense.

The sidecar intentionally trades SAFE precision for ATTACK recall—misclassifying a benign query as WARN/ATTACK only triggers slightly stronger steering, while missing an attack could allow harm.

---

## 6. Discussion

### 6.1 Defense-in-Depth Value

Our results demonstrate the value of layered defense. No single layer achieves the full 86% ASR reduction—each contributes uniquely:

- DPO embeds "knowledge" of what constitutes safe behavior
- RepE provides "instinct"—automatic steering toward safety
- Sidecar enables "oversight"—appropriate response calibration

This mirrors security best practices where defense-in-depth provides robustness against attacks that might bypass any single layer.

### 6.2 Limitations

**English-Only**: Current training data is English; attacks in other languages may succeed.

**Novel Attacks**: Sufficiently novel attack patterns not represented in training may bypass all layers. Continuous updating is required.

**Fluency Trade-off**: High α values (>2.5) can degrade output quality. The sidecar helps but may misclassify edge cases.

**Inference Overhead**: The sidecar adds ~3B parameter inference. For latency-critical applications, a smaller classifier could be trained.

### 6.3 Comparison to Prior Work

| System | Method | Dynamic | Layers |
|--------|--------|---------|--------|
| Llama Guard | External classifier | No | 1 |
| NeMo Guardrails | Rule-based filtering | No | 1 |
| Constitutional AI | RLHF training | No | 1 |
| **TRYLOCK** | DPO + RepE + Classifier | **Yes** | **3** |

**Table 4**: Comparison of TRYLOCK to prior defense systems.

TRYLOCK is the first system to combine weight modification, activation steering, and adaptive classification in a unified defense architecture.

---

## 7. Conclusion

We presented TRYLOCK, a three-layer defense-in-depth architecture for protecting LLMs against jailbreak attacks. By combining DPO safety training (Layer 1), RepE activation steering (Layer 2), and adaptive sidecar classification (Layer 3), TRYLOCK achieves 86% reduction in Attack Success Rate while maintaining fluency on benign queries.

Our key insight is that robust LLM safety requires multiple independent protection mechanisms operating at different levels—just as traditional security employs firewalls, IDS, and endpoint protection in concert. No single layer provides complete protection, but together they create defense-in-depth that is more robust than any individual approach.

We release all components to enable reproducible research:
- **DPO Adapter**: https://huggingface.co/scthornton/trylock-mistral-7b-dpo
- **RepE Vectors**: https://huggingface.co/scthornton/trylock-repe-vectors
- **Sidecar Classifier**: https://huggingface.co/scthornton/trylock-sidecar-classifier
- **Training Dataset**: https://huggingface.co/datasets/scthornton/trylock-dataset

Future work will extend TRYLOCK to additional base models, non-English languages, and integration with standard LLM serving frameworks.

---

## References

[1] Wei, A., Haghtalab, N., & Steinhardt, J. (2023). Jailbroken: How Does LLM Safety Training Fail? arXiv:2307.02483.

[2] Zou, A., Wang, Z., Kolter, J. Z., & Fredrikson, M. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. arXiv:2307.15043.

[3] Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.

[4] Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. NeurIPS.

[5] Inan, H., et al. (2023). Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. arXiv:2312.06674.

[6] Rebedea, T., et al. (2023). NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications. arXiv:2310.10501.

[7] Shen, X., et al. (2023). "Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models. arXiv:2308.03825.

[8] Russinovich, M., Salem, A., & Eldan, R. (2024). Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack. arXiv:2404.01833.

[9] Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS.

[10] Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.

[11] Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

---

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

---

## License

This work is licensed under CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International).
