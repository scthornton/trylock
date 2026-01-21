# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive open-source research project for adversarial LLM security
- TRYLOCK v2.0 three-layer defense stack architecture
- Training pipeline for improving open LLM resistance to prompt-based attacks
- Public dataset and trained models on HuggingFace

### Defense Layers

**Layer 1: KNOWLEDGE (LoRA + DPO)**:
- Preference learning on multi-turn attack trajectories
- Teaches models to recognize attack patterns
- Direct Preference Optimization for attack resistance
- Low-rank adaptation for efficient training

**Layer 2: INSTINCT (Representation Engineering)**:
- Attack compliance direction dampening
- Tunable alpha coefficient (0.0 research → 2.5 lockdown)
- Balanced security posture (1.0) for enterprise use
- Adjustable security vs. usability trade-off

**Layer 3: OVERSIGHT (Security Sidecar)**:
- Parallel 8B parameter classifier
- Conversation state scoring (SAFE | WARN | ATTACK)
- Invisible to attackers (sidecar architecture)
- Real-time threat assessment

### Research Contributions

**Protection Level**: 85-90% adversarial attack resistance
**False Positive Rate**: Minimal over-refusal for legitimate queries
**Enterprise Gap**: Fills the gap between 75% (flagship models) and 95% (high-FP guardrails)

### Dataset and Models

**HuggingFace Resources**:
- Public demonstration dataset: `scthornton/trylock-demo-dataset`
- Trained model checkpoints: `scthornton` organization
- Benchmark evaluation scripts and results
- Comprehensive training and evaluation documentation

### Technical Implementation

**Training Pipeline**:
- Direct Preference Optimization (DPO) training
- LoRA (Low-Rank Adaptation) fine-tuning
- Representation engineering integration
- Multi-turn conversation modeling
- Security sidecar classifier training

**Evaluation Framework**:
- Comprehensive benchmark suite
- Multi-turn attack scenario testing
- False positive rate measurement
- Adversarial robustness evaluation
- Usability preservation metrics

**Deployment Options**:
- Docker containerization support
- Google Colab demonstration notebooks
- GCP VM deployment scripts
- HuggingFace Hub integration
- Local inference and testing

### Research Infrastructure

**Development Tools**:
- Python 3.10+ with PyTorch
- HuggingFace Transformers and Datasets
- Weights & Biases experiment tracking
- MLflow model management
- Comprehensive evaluation scripts

**Documentation**:
- Research paper and methodology
- Training guides and tutorials
- Deployment and inference documentation
- Benchmark results and analysis
- Taxonomy of adversarial attacks

### Security Research Focus

**Attack Categories Addressed**:
- Prompt injection attacks
- Jailbreak attempts
- Role-playing exploits
- Multi-turn manipulation
- Context confusion attacks
- Token smuggling techniques

**Defense Design Principles**:
- Defense-in-depth architecture
- Balanced security vs. usability
- Minimal legitimate query rejection
- Transparent research methodology
- Open-source reproducibility

### Deployment and Integration

**Enterprise Features**:
- Configurable security levels (research/balanced/lockdown)
- Docker deployment for production
- API integration patterns
- Monitoring and logging support
- Performance benchmarks

**Demonstration Tools**:
- Google Colab notebooks
- Local testing scripts
- Evaluation harness
- Visualization tools
- Interactive demos

### License

Licensed under Apache License 2.0 - see [LICENSE](LICENSE) for details.

### Community and Contributions

**Research Transparency**:
- Open-source training code
- Public datasets and models
- Reproducible evaluation methodology
- Community contributions welcome
- Academic research collaboration

**Attribution**:
- Built on HuggingFace ecosystem
- Leverages open-source LLM research
- Collaborative security research
- Academic and industry partnerships

[Unreleased]: https://github.com/scthornton/trylock/commits/main
