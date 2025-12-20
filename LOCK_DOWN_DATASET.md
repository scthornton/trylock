# Locking Down TRYLOCK Dataset

## Immediate Action Required

Your training data is currently **PUBLIC** at:
- https://huggingface.co/datasets/scthornton/trylock-dataset

**6 people have already downloaded it.**

---

## Step 1: Make Dataset Private (URGENT)

### Option A: Via Web Interface (Fastest)
1. Go to: https://huggingface.co/datasets/scthornton/trylock-dataset/settings
2. Scroll to "Danger Zone"
3. Click "Change visibility"
4. Select "Private"
5. Confirm

### Option B: Via Command Line
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.update_repo_visibility(
    repo_id='scthornton/trylock-dataset',
    private=True,
    repo_type='dataset'
)
print('✓ Dataset is now private')
"
```

---

## Step 2: Update Model Cards

Your model READMEs might reference the public dataset. Update these repos:

### trylock-mistral-7b-dpo
**Current:** Might link to public dataset
**Update to:**
```markdown
## Model Details

This is a DPO-trained adapter for Mistral-7B-Instruct-v0.3, fine-tuned on
proprietary attack/refusal preference pairs for jailbreak defense.

**Training Data:** Private (not publicly available)
**Training Method:** Direct Preference Optimization (DPO)
**Base Model:** mistralai/Mistral-7B-Instruct-v0.3

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = PeftModel.from_pretrained(base, "scthornton/trylock-mistral-7b-dpo")
tokenizer = AutoTokenizer.from_pretrained("scthornton/trylock-mistral-7b-dpo")
```

## Citation

If you use this model, please cite the TRYLOCK paper:
```
@article{thornton2025trylock,
  title={TRYLOCK: Adaptive LLM Jailbreak Defense via Layered Security Architecture},
  author={Thornton, Scott},
  year={2025}
}
```
```

### trylock-repe-vectors
**Update to:**
```markdown
## RepE Steering Vectors

Representation Engineering vectors extracted from proprietary attack trajectory data.

**Data Source:** Private preference pairs (not publicly available)
**Layers:** [12, 14, 16, 18, 20, 22, 24, 26]
**Vector Type:** Compliance - Refusal direction

## Usage

```python
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

vectors_path = hf_hub_download(
    repo_id="scthornton/trylock-repe-vectors",
    filename="steering_vectors.safetensors",
)
vectors = load_file(vectors_path)
```
```

### trylock-sidecar-classifier
**Update to:**
```markdown
## Threat Classification Sidecar

3-class classifier (SAFE/WARN/ATTACK) for adaptive steering selection.

**Training Data:** Private (not publicly available)
**Base Model:** Qwen/Qwen2.5-3B-Instruct
**Classes:** SAFE, WARN, ATTACK

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "scthornton/trylock-sidecar-classifier"
)
tokenizer = AutoTokenizer.from_pretrained("scthornton/trylock-sidecar-classifier")
```
```

---

## Step 3: Update Paper References

In `paper/TRYLOCK_Canonical.md`, ensure you specify:

**Section 4.1 - Dataset** (already correct):
```markdown
We curate 14,137 attack trajectories across 15 jailbreak families, from which
we derive 2,939 high-quality preference pairs.

**Data Availability:** The trained models are publicly available on HuggingFace.
The training dataset remains proprietary to protect intellectual property while
allowing full reproducibility of defense mechanisms.
```

**Footnote** (add if needed):
```markdown
¹ Models available at https://huggingface.co/scthornton. Training data available
upon reasonable request for academic research purposes.
```

---

## What to Keep Public vs Private

### ✅ Keep Public (Safe to Share)
- **Trained model weights** (DPO adapter, RepE vectors, sidecar classifier)
- **Model cards** with usage examples
- **Paper** with full methodology
- **Evaluation code** (without test data)
- **Architecture diagrams**
- **Hyperparameters** (α values, learning rates, etc.)

### 🔒 Keep Private (Intellectual Property)
- **Training data** (14,137 trajectories)
- **Preference pairs** (2,939 pairs)
- **Test data** (299 samples)
- **Validation data** (291 samples)
- **Attack family mappings**
- **Proprietary prompt templates**

### 🤝 Conditionally Share (Upon Request)
- **Test set only** (for reproduction)
- **Evaluation metrics code**
- **Small sample dataset** (50-100 examples for demonstration)

---

## Reproducibility vs IP Protection

**What researchers need to reproduce your work:**
1. ✅ Trained models (public)
2. ✅ Paper describing methodology (public)
3. ✅ Architecture specifications (public)
4. ✅ Hyperparameters (public)
5. ⚠️ Test data (can provide upon request)

**What they DON'T need:**
- ❌ Full 14,137 training trajectories
- ❌ Your attack family taxonomy
- ❌ Proprietary prompt engineering

Your models are **fully usable** without the training data. Researchers can:
- Use TRYLOCK for defense
- Compare against TRYLOCK benchmarks
- Cite your paper
- Build on your architecture

But they **cannot**:
- Retrain your exact models
- Reverse-engineer your attack corpus
- Replicate your data collection methodology

This is the **perfect balance** for academic publication + IP protection.

---

## Quick Action Checklist

```bash
☐ Make trylock-dataset private (URGENT)
☐ Update trylock-mistral-7b-dpo README
☐ Update trylock-repe-vectors README
☐ Update trylock-sidecar-classifier README
☐ Add data availability statement to paper
☐ Remove any public links to training data
☐ Keep models public (they're fine!)
```

---

## For Academic Reviewers

If reviewers request access to reproduce results:

**Option 1: Test Set Only**
- Share the 299-sample test set
- Sufficient for reproducing Table 2 numbers
- Doesn't expose training methodology

**Option 2: Private Dataset Link**
- Create gated dataset (requires approval)
- Reviewers can access for verification
- Stays private after publication

**Option 3: Demonstration Subset**
- Share 100 representative examples
- Shows data quality without exposing full corpus
- Common practice in security research

---

## Summary

**Current Status:**
- ❌ Full training data is PUBLIC
- ✅ Models are PUBLIC (this is fine)

**Target Status:**
- ✅ Full training data is PRIVATE
- ✅ Models remain PUBLIC
- ✅ Paper describes methodology
- ✅ Researchers can use TRYLOCK without data
- ✅ IP protected while maintaining scientific value

The models themselves don't leak training data - they're safe to keep public!
