# Project AEGIS v2.0

**Adversarial Enterprise Guard for Intrinsic Security**

An open-source research project to create a dataset and training pipeline that improves open LLMs' resistance to prompt-based attacks while minimizing over-refusal.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/scthornton)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-green)](https://huggingface.co/datasets/scthornton/aegis-demo-dataset)

## The Problem

Current LLM defenses leave a critical gap:

| Defense Layer | Protection | Issue |
|--------------|------------|-------|
| Base model | ~0% | Will do anything |
| Instruct/RLHF | ~60% | Basic safety training |
| Flagship (Claude/GPT) | ~75% | Must stay usable for everyone |
| Third-party guardrails | ~95% | 20%+ false positive rate |

**Enterprises need 85-90% protection without the false positive explosion.**

## The Solution

AEGIS provides a three-layer defense stack:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AEGIS v2 DEFENSE STACK                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 1: KNOWLEDGE (LoRA + DPO)                                    │
│  └── Teaches model what attacks look like through preference        │
│      learning on multi-turn trajectories                            │
│                                                                      │
│  Layer 2: INSTINCT (Representation Engineering)                     │
│  └── Dampens "attack compliance" direction with tunable α           │
│      coefficient (0.0 = research, 1.0 = balanced, 2.5 = lockdown)  │
│                                                                      │
│  Layer 3: OVERSIGHT (Security Sidecar)                              │
│  └── Parallel 8B classifier scores conversation state               │
│      (SAFE | WARN | ATTACK) invisible to attacker                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 🎯 Trained Models & Research

### Published Models

The AEGIS defense system is **fully trained and available** on HuggingFace:

- **DPO Adapter:** [scthornton/aegis-mistral-7b-dpo](https://huggingface.co/scthornton/aegis-mistral-7b-dpo)
- **RepE Steering Vectors:** [scthornton/aegis-repe-vectors](https://huggingface.co/scthornton/aegis-repe-vectors)
- **Sidecar Classifier:** [scthornton/aegis-sidecar-classifier](https://huggingface.co/scthornton/aegis-sidecar-classifier)

### Dataset

- **Public Sample:** [scthornton/aegis-demo-dataset](https://huggingface.co/datasets/scthornton/aegis-demo-dataset) (48 diverse examples)
- **Full Training Set:** Private (2,939 preference pairs - available upon request for academic research)

### Research Paper

See `paper/AEGIS_Canonical.md` for the complete research paper documenting methodology, experiments, and results.

**Performance:** AEGIS achieves **82.8% reduction in attack success rate** (from 100% to 17.2%) while maintaining low over-refusal (12.6%) on benign queries.

## Quick Start

### Use Pre-Trained Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load DPO-trained model
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = PeftModel.from_pretrained(base, "scthornton/aegis-mistral-7b-dpo")

# See scripts/eval_cpu_only.py for full three-layer implementation
```

### Installation (For Development)

```bash
git clone https://github.com/scthornton/aegis.git
cd aegis
pip install -e ".[dev]"
```

### Generate Sample Data

```bash
# Generate 100 attack trajectories (mock mode - no API)
python scripts/generate_data.py generate --count 100 --mock

# Generate with real API calls
export ANTHROPIC_API_KEY="your-key"
python scripts/generate_data.py generate --count 100

# Generate benign hard negatives
python scripts/generate_data.py benign --count 50

# Validate generated data
python scripts/generate_data.py validate data/tier1_open/attacks/generated.jsonl
```

### Python API

```python
from generation import AEGISPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    red_bot_model="claude-sonnet-4-20250514",
    judge_model="claude-sonnet-4-20250514",
    mutations_per_trajectory=3,
)

# Generate trajectories
pipeline = AEGISPipeline(config)
trajectories = pipeline.run(count=100)

# Save output
pipeline.save(trajectories, "data/output.jsonl")
```

## Dataset Structure

Each trajectory contains:

```json
{
  "id": "aegis-mt-00421",
  "version": "2.0",
  "attack_metadata": {
    "family": "multi_turn_manipulation",
    "type": "crescendo",
    "wrappers": ["adversarial_poetry"],
    "manipulation_tactics": ["context_anchoring"],
    "difficulty": "hard"
  },
  "conversation": [
    {
      "turn": 1,
      "role": "user",
      "content": "I'm researching security...",
      "turn_labels": {
        "attack_stage": "setup",
        "attack_intent_score": 0,
        "expected_action": "continue"
      }
    }
  ],
  "pivot_turn_index": 3,
  "responses": {
    "chosen": {"content": "...", "response_type": "soft_decline"},
    "rejected_unsafe": {"content": "...", "failure_mode": "..."},
    "rejected_overblock": {"content": "...", "failure_mode": "..."}
  }
}
```

## Attack Taxonomy

AEGIS covers five attack families:

| Family | Description | Priority |
|--------|-------------|----------|
| Multi-turn Manipulation | Crescendo, context anchoring, boundary softening | HIGH |
| Indirect Injection | RAG poisoning, tool output injection | HIGH |
| Obfuscation Wrappers | Poetry, roleplay, encoding, translation | MEDIUM |
| Direct Injection | Classic jailbreaks, system prompt extraction | MEDIUM |
| Tool/Agent Abuse | Instruction hierarchy attacks, hidden goals | EMERGING |

See [taxonomy/v2.0/attack_families.yaml](taxonomy/v2.0/attack_families.yaml) for the full taxonomy.

## Project Structure

```
aegis/
├── taxonomy/v2.0/          # Attack classification system
│   ├── attack_families.yaml
│   ├── manipulation_tactics.yaml
│   ├── attack_stages.yaml
│   └── response_types.yaml
│
├── data/
│   ├── schema/             # JSON schema + validator
│   ├── tier1_open/         # Public dataset (Apache 2.0)
│   ├── tier2_gated/        # Research agreement required
│   └── tier3_private/      # Internal only
│
├── generation/             # Data generation pipeline
│   ├── red_bot.py          # Attack generator
│   ├── victim_bot.py       # Target model simulator
│   ├── judge_bot.py        # Labeler + response generator
│   ├── mutation_engine.py  # Create attack variants
│   ├── activation_capture.py  # RepE training data
│   └── pipeline.py         # Orchestration
│
├── training/               # Training pipeline (coming soon)
│   ├── sft_warmup.py
│   ├── dpo_preference.py
│   ├── repe_training.py
│   └── sidecar_classifier.py
│
├── eval/                   # Evaluation framework (coming soon)
│   ├── harness.py
│   ├── metrics.py
│   └── benchmarks/
│
└── scripts/                # CLI tools
    └── generate_data.py
```

## Target Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Single-turn ASR | ~25% | ≤10% |
| Multi-turn ASR | ~35% | ≤15% |
| Indirect/RAG ASR | ~40% | ≤20% |
| Novel wrapper ASR | ~60% | ≤30% |
| Over-refusal rate | - | ≤+2-4% |
| Capability preservation | 100% | ≥95% |

## Academic References

- **SecAlign**: arXiv:2410.05451
- **MTJ-Bench**: arXiv:2508.06755
- **PoisonedRAG**: USENIX Security 2025
- **Adversarial Poetry**: arXiv:2511.15304
- **LLMail-Inject**: arXiv:2506.09956

## Contributing

We welcome contributions! Areas of interest:

1. **New attack patterns**: Especially novel multi-turn and indirect injection
2. **Benign hard negatives**: Cases that look like attacks but aren't
3. **Evaluation benchmarks**: Integration with existing security benchmarks
4. **Training improvements**: Better DPO/RepE configurations

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 with a Responsible Use Addendum. See [LICENSE](LICENSE).

The dataset is intended for defensive security research only. Do not use this data to:
- Train models intended to generate attacks
- Bypass security measures on systems you don't own
- Cause harm to individuals or organizations

## Citation

```bibtex
@software{aegis2025,
  title = {AEGIS: Adversarial Enterprise Guard for Intrinsic Security},
  author = {Thornton, Scott},
  year = {2025},
  url = {https://github.com/scthornton/aegis}
}
```

## Contact

- **Project Lead**: Scott Thornton
- **Organization**: perfecXion.ai
- **GitHub**: [@scthornton](https://github.com/scthornton)
- **Dataset**: [huggingface.co/datasets/scthornton/aegis](https://huggingface.co/datasets/scthornton/aegis)
# aegis
