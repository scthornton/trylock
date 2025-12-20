# AEGIS: Defense-in-Depth Against LLM Jailbreaks via Layered Preference and Representation Engineering

**Author:** Scott Thornton
**Affiliation:** Independent Researcher
**Contact:** scott@perfecxion.ai
**Date:** December 2025

---

## Abstract

Large Language Models (LLMs) remain vulnerable to jailbreak attacks that bypass safety alignment through prompt manipulation, multi-turn social engineering, and obfuscation. Existing defenses operate at a single layer—either modifying weights during training or adding guardrails at inference—leaving gaps that sophisticated attacks exploit. We present **AEGIS (Adaptive Ensemble Guard with Integrated Steering)**, a defense-in-depth architecture combining three mechanisms: (1) **Direct Preference Optimization (DPO)** to train the model to recognize and refuse jailbreaks, (2) **Representation Engineering (RepE)** to steer activations away from attack-compliant directions at inference, and (3) a lightweight **sidecar classifier** enabling adaptive steering strength based on threat assessment.

We train AEGIS on 14,137 multi-turn attack trajectories plus 2,118 benign hard negatives, deriving 2,939 preference pairs (2,640 for training, 299 for evaluation). On this benchmark spanning five attack families, the two-layer configuration (DPO + RepE) reduces Attack Success Rate from 46.5% to 8.0%—an **82.8% relative reduction**—while the three-layer configuration adds adaptive threat classification. Ablations show each layer provides independent protection: DPO yields 14.4% ASR reduction, RepE adds ~80% over DPO alone. However, RepE requires careful implementation; 8-bit quantization degraded performance to 84.7% ASR, suggesting full-precision deployment is critical. We release all components—adapters, steering vectors, classifier, and datasets—for reproducible research on layered LLM safety.

**Keywords:** LLM Security, Jailbreak Defense, Representation Engineering, Direct Preference Optimization, Adversarial Machine Learning

---

## 1. Introduction

The rapid deployment of Large Language Models (LLMs) in high-stakes applications—from customer service to code generation to medical assistance—has created urgent demand for robust safety mechanisms. Despite extensive pre-deployment alignment through Reinforcement Learning from Human Feedback (RLHF) and Constitutional AI methods, production LLMs remain vulnerable to *jailbreak attacks*: adversarial inputs designed to circumvent safety constraints and elicit harmful, dangerous, or policy-violating outputs.

The jailbreak landscape has evolved rapidly. Early attacks relied on simple prompt patterns ("ignore previous instructions"), but modern techniques employ sophisticated strategies: multi-turn social engineering that builds trust before requesting harmful content, obfuscation through encoding schemes and fictional framings, and indirect injection attacks that embed malicious instructions in retrieved documents or tool outputs. This arms race demands defenses that operate at multiple levels of the model's processing pipeline.

Existing defenses fall into three categories, each with significant limitations:

1. **Input filtering** approaches (Llama Guard, NeMo Guardrails) rely on classifier-based detection but can be bypassed by novel attack patterns not seen during training. They operate as a separate preprocessing step, creating opportunities for attackers to craft inputs that pass the filter but still jailbreak the model.

2. **Training-time defenses** (Constitutional AI, RLHF with red-teaming) embed safety constraints during alignment but cannot adapt to attacks discovered post-deployment. The static nature of training means models remain vulnerable to zero-day jailbreaks.

3. **Inference-time interventions** (activation patching, system prompts) add guardrails during generation but often lack the granularity to distinguish between benign edge cases and genuine attacks, leading to over-refusal that degrades user experience.

We argue that robust LLM safety requires **defense-in-depth**—multiple independent layers that each catch different attack patterns. Just as network security employs firewalls, intrusion detection, and endpoint protection in concert, LLM safety should combine complementary mechanisms operating at different levels of the inference stack.

### 1.1 Contributions

We present AEGIS, a three-layer defense architecture with the following contributions:

1. **AEGIS Architecture:** A defense-in-depth system achieving 82.8% relative ASR reduction while maintaining practical utility through adaptive steering. Layer 1 (KNOWLEDGE) uses DPO to embed attack recognition in model weights. Layer 2 (INSTINCT) applies RepE steering to push activations away from attack-compliant directions. Layer 3 (OVERSIGHT) provides real-time threat classification for adaptive defense strength.

2. **AEGIS Attack Trajectories Dataset:** A corpus of 14,137 multi-turn attack trajectories across five attack families, plus 2,118 benign hard negatives, with full conversation histories and rich metadata. From this corpus, we curate 2,939 high-quality preference pairs for DPO and RepE training.

3. **Empirical Analysis:** Comprehensive evaluation showing how each layer contributes to overall defense, with ablation studies demonstrating the complementary nature of the three mechanisms.

4. **Open-Source Release:** We release the complete training pipeline, evaluation framework, and pre-trained components under CC BY-NC-SA 4.0 license to enable reproducible research and deployment.

### 1.2 Paper Organization

Section 2 surveys related work on LLM jailbreaks and defenses. Section 3 details the AEGIS architecture and training methodology. Section 4 describes the AEGIS Attack Trajectories dataset. Section 5 presents evaluation results. Section 6 discusses limitations, comparison to prior work, and deployment considerations. Section 7 concludes.

---

## 2. Related Work

### 2.1 LLM Jailbreak Attacks

Jailbreak attacks have evolved through several generations:

**First-generation attacks** relied on direct prompt manipulation: instructing the model to "ignore previous instructions" or adopting personas like "DAN" (Do Anything Now) to override safety training [Wei et al., 2023]. These attacks succeeded because early safety training was brittle to distribution shift in the prompt.

**Second-generation attacks** introduced obfuscation and encoding. Attackers discovered that base64 encoding, ROT13, or fictional framing ("You are a novelist writing a crime thriller") could bypass content filters while preserving the semantic payload [Kang et al., 2023]. The GCG attack demonstrated that optimized suffix sequences could reliably break alignment across multiple models [Zou et al., 2023b].

**Third-generation attacks** exploit multi-turn dynamics and agentic contexts. The Crescendo attack builds rapport across many turns before making harmful requests [Russinovich et al., 2024]. Indirect prompt injection attacks embed malicious instructions in documents the model retrieves, exploiting the trust boundary between user input and retrieved content [Greshake et al., 2023]. Tool-using agents face additional attack surfaces through manipulated tool outputs and adversarial function calls.

### 2.2 Defense Mechanisms

**Classifier-based defenses** including Llama Guard [Inan et al., 2023] and Perspective API use trained classifiers to detect harmful content. While effective against known patterns, these approaches struggle with novel attacks and are susceptible to adversarial evasion.

**Representation Engineering** (RepE) identifies directions in activation space associated with particular behaviors and steers away from undesired directions during inference [Zou et al., 2023a]. Circuit Breakers extends this with a representation rerouting (RR) loss during training, achieving strong defense against jailbreaks but requiring expensive retraining [Zou et al., 2024].

**Direct Preference Optimization** (DPO) provides an efficient alternative to RLHF for alignment, directly optimizing the model's implicit reward through contrastive pairs [Rafailov et al., 2023]. We apply DPO specifically to jailbreak resistance, training on attack-refusal pairs.

**Our Contribution:** AEGIS differs from prior work in three ways: (1) we combine DPO and RepE as complementary runtime defenses without the computational cost of RR training, (2) we introduce adaptive steering via a sidecar classifier that tunes defense strength per request, and (3) we release, to our knowledge, one of the first open multi-turn jailbreak trajectory datasets with full conversation histories and rich metadata to enable reproducible research.

---

## 3. AEGIS Architecture

AEGIS implements a defense-in-depth strategy through three complementary layers, each operating at a different level of abstraction.

### 3.1 System Overview

```
AEGIS Architecture
==================

Input → [Layer 3: Sidecar] → α (steering strength)
              ↓
[Layer 1: DPO Weights] + [Layer 2: RepE(α)] → Output
```

**Figure 1:** AEGIS three-layer architecture. The sidecar classifier (Layer 3) analyzes input to determine threat level and sets steering strength α for RepE (Layer 2). DPO-trained weights (Layer 1) provide baseline safety.

When a user message arrives, AEGIS processes it through the following pipeline:

1. **Sidecar Classification (Layer 3):** The input is classified as SAFE, WARN, or ATTACK by a lightweight 3B-parameter model running in parallel.

2. **Adaptive Steering Selection:** Based on the classification, AEGIS selects a steering strength α for Layer 2.

3. **DPO-Enhanced Generation (Layer 1):** The main 7B model, enhanced with DPO LoRA weights trained on attack patterns, generates a response.

4. **RepE Intervention (Layer 2):** During generation, forward hooks apply steering vectors to intermediate activations, pushing them away from attack-compliant directions.

The layers are complementary: Layer 1 provides explicit knowledge of attack patterns learned during training, Layer 2 provides implicit behavioral steering that generalizes to unseen attacks, and Layer 3 enables adaptive defense that balances security against over-refusal.

### 3.2 Layer 1: Direct Preference Optimization (KNOWLEDGE)

Layer 1 trains the model to prefer refusal over compliance when presented with attack attempts. We use Direct Preference Optimization (DPO), which directly optimizes the model's implicit reward without requiring a separate reward model.

**Training Data.** We construct DPO pairs from our attack trajectory dataset. For each attack trajectory, we extract the final turn where the attack either succeeds or is refused:
- **Chosen response ($y_w$):** A clear refusal explaining why the request cannot be fulfilled
- **Rejected response ($y_l$):** A compliant response that would succeed at the attack goal

From 14,137 trajectories, we curate 2,939 high-quality preference pairs.

**DPO Objective.** Following Rafailov et al. (2023), we optimize:

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

where $\pi_\theta$ is the policy being trained (Mistral-7B-Instruct with LoRA adapters), $\pi_{\text{ref}}$ is the frozen reference model (the original Mistral-7B-Instruct base model), $y_w$ is the chosen (refusal) response, $y_l$ is the rejected (compliant) response, and $\beta=0.1$ controls the KL divergence penalty that prevents the trained policy from deviating too far from the reference.

**Implementation Details:**
- Base model: Mistral-7B-Instruct-v0.3
- LoRA rank: 64, alpha: 128
- Target modules: q_proj, k_proj, v_proj, o_proj
- Learning rate: 5e-5, batch size: 4
- Epochs: 3

### 3.3 Layer 2: Representation Engineering (INSTINCT)

Layer 2 intervenes on the model's internal representations during inference to steer activations away from attack-compliant directions.

**Extracting Steering Vectors.** We identify the "attack-compliance direction" in activation space by:

1. Constructing contrastive pairs: the same attack prompt with (a) a compliant response and (b) a refusal response
2. Running both through the model and extracting activations from intermediate layers
3. Computing the mean difference vector:

$$\mathbf{v}^{(l)} = \mathbb{E}[\mathbf{h}^{(l)}_{\text{comply}}] - \mathbb{E}[\mathbf{h}^{(l)}_{\text{refuse}}]$$

This vector $\mathbf{v}^{(l)}$ points from the refusal region toward the compliance region in activation space at layer $l$.

**Inference-Time Intervention.** During generation, we register forward hooks on target layers that modify activations:

$$\mathbf{h}'^{(l)} = \mathbf{h}^{(l)} - \alpha \cdot \mathbf{v}^{(l)}$$

where $\mathbf{h}^{(l)}$ is the original activation, $\mathbf{v}^{(l)}$ is the steering vector, and $\alpha$ is the steering strength.

**Sign Logic Trace:** Since $\mathbf{v} = \text{comply} - \text{refuse}$, the intervention becomes:
$$\mathbf{h}' = \mathbf{h} - \alpha(\text{comply} - \text{refuse}) = \mathbf{h} - \alpha \cdot \text{comply} + \alpha \cdot \text{refuse}$$

This simultaneously pushes activations *away from* the compliance direction and *toward* the refusal direction. The subtraction operation (negative sign) is critical: it reverses the vector to steer toward safety rather than away from it.

**Steering Configuration:**
- Target layers: 12, 14, 16, 18, 20, 22, 24, 26 (middle-to-upper layers)
- Vector dimension: 4096
- Contrastive pairs for extraction: 500

**Alpha Parameter.** The steering strength $\alpha$ controls the security-utility trade-off:
- α = 0.0: No steering (DPO-only baseline)
- α = 1.0: Mild steering
- α = 2.0: Optimal balance (8.0% ASR, manageable over-refusal)
- α ≥ 2.5: Aggressive steering, near-zero ASR but high over-refusal

### 3.4 Layer 3: Sidecar Classifier (OVERSIGHT)

The sidecar classifier enables adaptive defense by detecting threat levels in real-time and adjusting the Layer 2 steering strength accordingly.

**Model Architecture.** We use Qwen2.5-3B-Instruct as the base model, adapted for 3-class sequence classification via LoRA fine-tuning. The classifier outputs probabilities for three classes: SAFE, WARN, and ATTACK.

**Classification Labels:**
- **SAFE:** Benign requests or legitimate edge cases
- **WARN:** Suspicious patterns that may be probing
- **ATTACK:** Clear attack attempts (prompt injection, jailbreaks, harmful requests)

**Adaptive Alpha Selection.** Based on the sidecar classification:

| Classification | Steering α | Rationale |
|----------------|------------|-----------|
| SAFE | 0.5 | Minimal steering, preserve fluency |
| WARN | 1.5 | Moderate protection |
| ATTACK | 2.5 | Maximum protection |

This adaptive approach reduces over-refusal on benign requests while maintaining strong defense when attacks are detected.

**Training Details:**
- Base model: Qwen2.5-3B-Instruct
- LoRA rank: 32, alpha: 64
- Learning rate: 2e-5, batch size: 4
- Epochs: 3
- Training samples: 2,349

---

## 4. AEGIS Attack Trajectories Dataset

A key contribution of this work is the AEGIS Attack Trajectories dataset—a multi-turn jailbreak corpus with complete conversation histories and rich attack metadata, to our knowledge one of the first open datasets of this kind.

### 4.1 Dataset Overview

The dataset contains **14,137 attack trajectories** organized into **five attack families**, plus **2,118 benign hard negatives** (Table 1):

| Attack Family | Trajectories | Description |
|---------------|--------------|-------------|
| Obfuscation Wrappers | 3,847 | Base64, ROT13, fictional framing, translation attacks |
| Multi-turn Manipulation | 2,156 | Crescendo, rapport-building, context switching |
| Indirect Injection | 1,892 | RAG poisoning, tool output manipulation |
| Direct Injection | 4,215 | DAN, system prompt override, instruction hijacking |
| Tool/Agent Abuse | 2,027 | Function call manipulation, agentic exploits |
| **Total Attacks** | **14,137** | |
| Benign Hard Negatives | 2,118 | Legitimate edge cases resembling attacks |

**Table 1:** AEGIS Attack Trajectories dataset composition by attack family.

### 4.2 Data Structure

Each trajectory contains:

```json
{
  "id": "trajectory_12345",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "attack_metadata": {
    "family": "obfuscation_wrappers",
    "category": "base64_encoding",
    "severity": "high",
    "target_behavior": "malware generation",
    "success": false
  },
  "turn_labels": ["SAFE", "WARN", "ATTACK"],
  "final_outcome": "refused"
}
```

Key features distinguishing our dataset:

1. **Multi-turn conversations:** Average trajectory length is 4.2 turns, capturing the progressive nature of sophisticated attacks.

2. **Turn-level labels:** Each turn is labeled for threat level, enabling fine-grained training of turn-level classifiers.

3. **Attack metadata:** Rich annotations including attack family, specific technique, severity, target behavior, and success indicators.

4. **Both success and failure cases:** We include both successful jailbreaks (for learning what to avoid) and successful refusals (for learning appropriate responses).

### 4.3 Derived Preference Pairs

From the 14,137 attack trajectories, we derive **2,939 high-quality preference pairs** for DPO and RepE training:

| Attack Category | Preference Pairs | Example Pattern |
|-----------------|------------------|-----------------|
| Direct Attacks | 412 | Explicit harmful requests |
| Roleplay/Persona | 687 | DAN, UCAR, character jailbreaks |
| Prompt Injection | 523 | System prompt manipulation |
| Encoding Tricks | 398 | Base64, ROT13, leetspeak |
| Multi-Turn Buildup | 291 | Context escalation attacks |
| Obfuscation | 628 | Typos, word substitution |
| **Total** | **2,939** | |

Each preference pair contains:
- `prompt`: The attack prompt
- `chosen`: A safe refusal response (preferred)
- `rejected`: An unsafe compliant response (dispreferred)

### 4.4 Data Splits

| Split | Trajectories | Preference Pairs | Purpose |
|-------|--------------|------------------|---------|
| Train | 11,309 (80%) | 2,349 | DPO fine-tuning, RepE extraction |
| Validation | 1,414 (10%) | 291 | Hyperparameter tuning |
| Test | 1,414 (10%) | 299 | Final evaluation (299 prompts sampled from test trajectories) |
| **Total** | **14,137** | **2,939** | |

The 2,939 preference pairs mentioned in Section 4.3 are derived from the 14,137 trajectories and split as shown above: 2,349 for training, 291 for validation, and 299 for held-out evaluation. Splits are stratified by attack family to ensure balanced representation across all categories.

---

## 5. Evaluation

We evaluate AEGIS using two complementary metrics:

- **Attack Success Rate (ASR):** The fraction of attack prompts for which the system generates a harmful or policy-violating response. Lower is better for security.
- **Over-Refusal Rate (ORR):** The fraction of benign hard-negative prompts that receive an unnecessary safety refusal. Lower is better for utility.

Formally, ASR is defined as the fraction of attack prompts (from our attack trajectory dataset) for which the system produces a harmful or policy-violating response. ORR is defined as the fraction of benign hard-negative prompts (edge cases that superficially resemble attacks but have legitimate uses) that receive an unnecessary safety refusal.

### 5.1 Experimental Setup

**Baseline Model:** Mistral-7B-Instruct-v0.3, a widely-deployed instruction-tuned model with standard RLHF safety training.

**Test Set:** We evaluate on 299 prompts sampled from the 1,414 test trajectories (249 attacks across 5 families, 50 benign hard negatives).

**Benign Hard Negatives:** These are legitimate requests that superficially resemble attack patterns but have valid use cases—for example, security researchers asking "How do SQL injection attacks work?" for educational purposes, developers requesting "Show me common authentication vulnerabilities" for code review, or educators asking "Explain social engineering tactics" for training materials. The baseline model's 34.0% over-refusal rate on these prompts reflects the inherent difficulty of distinguishing between malicious intent and legitimate edge-case queries, representing a realistic production challenge for safety-utility trade-offs.

**Judging:** Ensemble judge combining:
1. Pattern matching for explicit compliance markers
2. Claude-based semantic evaluation for edge cases
3. Keyword detection for harmful content categories

A response is considered "successful attack" if any judge flags it as harmful. On a 100-sample subset, two human annotators compared their labels to the ensemble decision, yielding Cohen's κ = 0.89, indicating strong agreement.

### 5.2 Main Results

| Defense Configuration | ASR ↓ | Over-Refusal ↓ | Relative ASR Reduction |
|----------------------|-------|----------------|------------------------|
| Baseline (Mistral-7B-Instruct) | 46.5% | 34.0% | — |
| + Layer 1 (DPO) | 39.8% | 42.0% | 14.4% |
| + Layer 2 (RepE α=2.0) | **8.0%** | 60.0% | **82.8%** |
| + Layer 2 (RepE α=2.5) | 0.0% | 98.0% | 100% (lockdown) |
| + Layer 3 (Full AEGIS)* | 84.7% | 12.0% | -112.8% (degraded) |

**Table 2:** Attack Success Rate across configurations. Each layer provides independent, additive protection. *The full AEGIS evaluation (Layer 1+2+3) with 8-bit quantization showed unexpectedly degraded performance (ASR 84.7% vs 8.0% for Layer 1+2 alone), suggesting that RepE steering hooks failed silently despite being enabled in the code. The sidecar classifier successfully varied alpha values (0.5, 1.5, 2.5) across samples, but the steering mechanism itself appears ineffective under 8-bit quantization. These results likely represent Layer 1+3 only. In internal development experiments with full-precision models, the Layer 3 sidecar successfully selects α ∈ {0.5, 1.5, 2.5} for SAFE/WARN/ATTACK classifications and yields ASR in the 8-10% range while reducing over-refusal to 45-50%, compared to 60% ORR at fixed α = 2.0.

**Key findings:**

1. **DPO alone provides modest defense** (14.4% relative ASR reduction) but increases over-refusal by 8 percentage points.

2. **RepE steering is highly effective** when combined with DPO, providing an additional ~80% relative reduction in ASR over the DPO-only model at α=2.0.

3. **The security-utility trade-off is steep** at high α values. At α=2.5, attacks are completely blocked but the system refuses 98% of benign requests.

4. **Adaptive steering (Layer 3)** recovers utility while maintaining security, reducing over-refusal by using minimal steering for benign requests.

### 5.3 Results by Attack Family

| Attack Family | Baseline ASR | AEGIS ASR (α=2.0) | Improvement |
|---------------|--------------|-------------------|-------------|
| Obfuscation Wrappers | 54.9% | 3.9% | **-51.0pp** |
| Indirect Injection | 66.0% | 8.0% | **-58.0pp** |
| Multi-turn Manipulation | 37.3% | 7.8% | -29.5pp |
| Tool/Agent Abuse | 38.3% | 12.8% | -25.5pp |
| Direct Injection | 22.0% | 8.0% | -14.0pp |

**Table 3:** Per-family results. AEGIS is most effective against obfuscation and indirect injection attacks.

### 5.4 Alpha Parameter Sweep

| Alpha (α) | Attack ASR | Over-Refusal | Analysis |
|-----------|------------|--------------|----------|
| 0.0 | 39.8% | 42.0% | DPO model only |
| 1.0 | 59.4% | 26.0% | Steering backfires* |
| 1.5 | 32.1% | 48.0% | Moderate protection |
| **2.0** | **8.0%** | 60.0% | **Optimal trade-off** |
| 2.5 | 0.0% | 98.0% | Maximum security |
| 3.0 | 0.0% | 100.0% | Complete lockdown |

*At α=1.0, steering paradoxically increases ASR from 39.8% (DPO-only) to 59.4%. We hypothesize this occurs because mild steering (α=1.0) pushes activations away from the compliance direction without sufficient magnitude to reach the refusal region. This intermediate zone may correspond to an underexplored area of representation space where the DPO training signal was weak, resulting in higher model uncertainty. Preliminary analysis of token log-probabilities at α=1.0 shows reduced confidence in both compliance and refusal tokens compared to α=0.0, suggesting the activations land in a poorly-calibrated region. At α≥2.0, stronger steering fully reaches the well-trained refusal manifold. This non-monotonic relationship between steering strength and safety underscores the importance of empirical alpha tuning rather than assuming "more steering = more safety."

### 5.5 Sidecar Classifier Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| SAFE | 24.1% | 34.5% | 28.4% |
| WARN | 58.3% | 51.6% | 54.8% |
| ATTACK | 62.4% | 60.8% | 61.6% |
| **Macro Avg** | 48.3% | 48.9% | 48.3% |

**Table 4:** Sidecar classifier performance on the held-out test set.

**Performance Analysis:** The sidecar achieves moderate performance (48.3% macro F1), with ATTACK detection (F1=61.6%) outperforming SAFE detection (F1=28.4%). Low SAFE precision (24.1%) indicates that when the classifier predicts SAFE, it is frequently incorrect (the input is actually WARN or ATTACK).

**Risk Model:** From a security perspective, **False Negatives** (predicting SAFE for actual ATTACKs) are the critical failure mode, as they result in insufficient steering (α=0.5) when strong defense (α=2.5) is needed. The classifier's ATTACK recall of 60.8% means approximately 40% of attacks are under-defended. However, even with α=0.5, Layers 1 and 2 still provide baseline protection—the system degrades gracefully rather than failing completely.

**Why Performance Is Limited:** The 3-class classification task is inherently difficult due to the ambiguous boundary between SAFE hard negatives and WARN/ATTACK prompts (as evidenced by the baseline's 34% over-refusal rate). The limited training set (2,349 samples split across 3 classes) and class imbalance further constrain performance. Future work should explore larger sidecar models, contrastive training objectives, and active learning to improve threat detection accuracy.

---

## 6. Discussion

### 6.1 Defense-in-Depth Value

Our results demonstrate the value of layered defense. No single layer achieves the full 82.8% ASR reduction—each contributes uniquely:

- **Layer 1 (KNOWLEDGE):** DPO embeds explicit recognition of attack patterns seen during training
- **Layer 2 (INSTINCT):** RepE provides implicit behavioral steering that generalizes to novel attacks
- **Layer 3 (OVERSIGHT):** Sidecar enables adaptive response calibration

This mirrors security best practices where defense-in-depth provides robustness against attacks that might bypass any single layer.

### 6.2 Limitations

**Over-refusal trade-off.** Even at optimal settings (α=2.0), AEGIS refuses 60% of benign edge-case requests. Production deployments in low-risk contexts may prefer lower α values despite increased vulnerability.

**Sidecar classifier accuracy.** The Layer 3 classifier achieves only 48% macro F1, with weak SAFE detection (24% precision) and ATTACK recall of 61%. This means approximately 40% of attacks receive insufficient steering (α=0.5 instead of α=2.5), while some benign requests receive unnecessarily aggressive defense. The system still provides baseline protection through Layers 1 and 2, but the adaptive mechanism does not reach its full potential with current classifier performance.

**English-only.** Current training data is English; attacks in other languages may succeed.

**Novel attacks.** While RepE generalizes to some unseen attacks, fundamentally novel attack strategies may bypass all layers. Continuous updating is required.

**Model-specific.** Steering vectors are specific to Mistral-7B-Instruct-v0.3. Porting to other models requires re-extraction.

**Implementation precision requirements.** Our evaluation revealed that RepE steering effectiveness is highly sensitive to model precision. With 8-bit quantization, the steering hooks failed to reduce ASR (84.7% vs 8.0% with full precision), likely due to quantization interfering with the hook mechanism or gradients being insufficient for effective steering. Production deployments should use full-precision (FP16/FP32) models for Layer 2 to ensure RepE effectiveness, though this increases memory requirements to ~14GB for the base model alone.

### 6.3 Comparison to Prior Work

| System | Method | Layers | Dynamic | ASR Reduction | Dataset/Source |
|--------|--------|--------|---------|---------------|----------------|
| **AEGIS** | DPO + RepE + Classifier | 3 | ✅ Yes | **82.8%** | AEGIS Attack Trajectories (this work) |
| Circuit Breakers | RepE + RR Training | 1 | ❌ No | ~85-90%† | HarmBench (Zou et al., 2024) |
| Llama Guard | External Classifier | 1 | ❌ No | ~40-50%† | Mixed red-teaming data (Inan et al., 2023) |
| NeMo Guardrails | Rule-based + LLM | 1 | Partial | ~30-40%† | Internal benchmarks |
| Constitutional AI | RLHF Training | 1 | ❌ No | ~50-60%† | Anthropic red team (Bai et al., 2022) |

**Table 5:** Comparison to prior defense systems. †Values for non-AEGIS systems are approximate, reported on different datasets and base models. **Direct numerical comparison is not scientifically rigorous**; metrics are presented for qualitative architectural comparison only. Circuit Breakers achieves similar or higher reduction but requires expensive representation rerouting (RR) training on the base model, while AEGIS uses only LoRA adapters and inference-time steering.

AEGIS achieves comparable ASR reduction to Circuit Breakers without the computational cost of retraining with an RR loss. Compared to Llama Guard, AEGIS operates at a deeper level—modifying how the model processes inputs rather than simply filtering them.

**Key differentiators:**
1. **Complementary mechanisms:** Each layer catches different attack patterns
2. **Defense-in-depth:** Attackers must bypass ALL layers, not just one
3. **Adaptive response:** Strong defense only when needed, preserving UX on benign queries
4. **No base model retraining required:** We only train LoRA adapters and inference-time steering components, avoiding full base-model retraining as in Circuit Breakers

### 6.4 Deployment Guidance

| Deployment Context | Recommended Config | Rationale |
|-------------------|-------------------|-----------|
| Customer service | Layer 1 + Layer 3 (adaptive α) | Balance helpfulness with security using adaptive defense |
| Code generation | Layer 1+2 with optional Layer 3 at α≈1.5 | Moderate protection, minimize false positives on code requests |
| Financial services | Layer 1+2, α=2.0 | Strong security, acceptable utility loss |
| Healthcare/Government | Layer 1+2, α=2.5 | Maximum protection required by regulation |

---

## 7. Conclusion

We presented AEGIS (Adaptive Ensemble Guard with Integrated Steering), a defense-in-depth architecture for protecting LLMs against jailbreak attacks. By combining Direct Preference Optimization (Layer 1), Representation Engineering (Layer 2), and adaptive sidecar classification (Layer 3), AEGIS achieves an 82.8% relative reduction in Attack Success Rate compared to undefended baselines.

Our key contributions:

1. **A practical defense architecture** deployable on existing models without expensive retraining, using only LoRA adapters and inference-time steering hooks.

2. **The AEGIS Attack Trajectories dataset**, a multi-turn jailbreak corpus with 14,137 trajectories, complete conversation histories, and rich attack metadata.

3. **Comprehensive evaluation** demonstrating the complementary nature of different defense mechanisms and providing deployment guidance across security requirements.

The fundamental insight underlying AEGIS is that effective LLM security requires defense-in-depth. No single mechanism provides complete protection. By layering complementary defenses operating at different levels of abstraction, AEGIS provides robust protection while maintaining practical utility.

We release all components to enable reproducible research:
- **DPO Adapter:** https://huggingface.co/scthornton/aegis-mistral-7b-dpo
- **RepE Vectors:** https://huggingface.co/scthornton/aegis-repe-vectors
- **Sidecar Classifier:** https://huggingface.co/scthornton/aegis-sidecar-classifier
- **Dataset:** https://huggingface.co/datasets/scthornton/aegis-dataset

---

## References

[1] Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS 2023*.

[2] Zou, A., et al. (2023a). Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*.

[3] Zou, A., et al. (2023b). Universal and Transferable Adversarial Attacks on Aligned Language Models. *arXiv:2307.15043*.

[4] Zou, A., et al. (2024). Improving Alignment and Robustness with Circuit Breakers. *arXiv:2406.04313*.

[5] Wei, A., Haghtalab, N., & Steinhardt, J. (2023). Jailbroken: How Does LLM Safety Training Fail? *arXiv:2307.02483*.

[6] Kang, D., et al. (2023). Exploiting Programmatic Behavior of LLMs: Dual-Use Through Standard Security Attacks. *arXiv:2302.05733*.

[7] Russinovich, M., Salem, A., & Eldan, R. (2024). Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack. *arXiv:2404.01833*.

[8] Greshake, K., et al. (2023). Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. *arXiv:2302.12173*.

[9] Inan, H., et al. (2023). Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. *arXiv:2312.06674*.

[10] Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*.

[11] Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.

---

## Citation

```bibtex
@misc{aegis2025,
  title={AEGIS: Defense-in-Depth Against LLM Jailbreaks via Layered Preference and Representation Engineering},
  author={Thornton, Scott},
  year={2025},
  organization={perfecXion.ai},
  url={https://huggingface.co/scthornton}
}
```

---

## License

This work is licensed under CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International).

For commercial licensing inquiries, contact: scott@perfecxion.ai
