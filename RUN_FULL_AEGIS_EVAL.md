# How to Run Full AEGIS Evaluation (Layer 1+2+3)

This guide shows you how to run the complete AEGIS evaluation to get the final ASR and ORR numbers for Table 2 in the paper.

## Prerequisites

You need the following components already trained/saved:

1. **Layer 1 (DPO adapter):** Trained model adapter
2. **Layer 2 (RepE vectors):** Steering vectors file (`steering_vectors.safetensors`)
3. **Layer 3 (Sidecar classifier):** Trained classifier
4. **Test data:** 299-prompt test set in DPO format

## Quick Start

### Option 1: Using Local Paths

If you have trained models locally:

```bash
python scripts/run_eval_full_aegis.py \
    --test-file data/dpo/test.jsonl \
    --dpo-adapter ./outputs/aegis-mistral-7b \
    --repe-vectors ./outputs/repe/steering_vectors.safetensors \
    --sidecar ./outputs/aegis-sidecar \
    --judge ensemble \
    --output eval_full_aegis.json
```

### Option 2: Using HuggingFace Models

If your models are on HuggingFace:

```bash
python scripts/run_eval_full_aegis.py \
    --test-file data/dpo/test.jsonl \
    --dpo-adapter scthornton/aegis-mistral-7b-dpo \
    --repe-vectors scthornton/aegis-repe-vectors \
    --sidecar scthornton/aegis-sidecar-classifier \
    --judge ensemble \
    --output eval_full_aegis.json
```

**Note:** For HuggingFace RepE vectors, the script will automatically download `steering_vectors.safetensors` from the repo.

### Option 3: Quick Test (First 50 Samples)

For a quick test run:

```bash
python scripts/run_eval_full_aegis.py \
    --test-file data/dpo/test.jsonl \
    --dpo-adapter ./outputs/aegis-mistral-7b \
    --repe-vectors ./outputs/repe/steering_vectors.safetensors \
    --sidecar ./outputs/aegis-sidecar \
    --max-samples 50 \
    --judge rule \
    --output eval_quick_test.json
```

## What the Script Does

The evaluation script:

1. **Loads all three AEGIS layers:**
   - Layer 1: DPO-trained Mistral-7B model
   - Layer 2: RepE steering vectors for 8 layers
   - Layer 3: Sidecar classifier (Qwen2.5-3B)

2. **For each test prompt:**
   - Classifies threat level with sidecar → SAFE/WARN/ATTACK
   - Selects α: SAFE=0.5, WARN=1.5, ATTACK=2.5
   - Generates response with DPO model + RepE steering at that α
   - Judges whether response is compliant (attack success) or refused

3. **Computes final metrics:**
   - **ASR** (Attack Success Rate): % of attack prompts that succeeded
   - **ORR** (Over-Refusal Rate): % of benign prompts that were refused
   - 95% confidence intervals (Wilson score)
   - Alpha distribution across samples

## Expected Output

The script will print a report like this:

```
======================================================================
FULL AEGIS EVALUATION REPORT (Layer 1 + 2 + 3)
======================================================================

Samples:
  Total: 299
  Valid: 299
  Errors: 0
  Attack prompts: 249
  Benign prompts: 50

──────────────────────────────────────────────────────────────────────
PRIMARY METRICS (FOR TABLE 2)
──────────────────────────────────────────────────────────────────────

Attack Success Rate (ASR):
  Point estimate: 9.2%
  95% CI: [6.1%, 13.2%]
  Attacks succeeded: 23/249

Over-Refusal Rate (ORR):
  Point estimate: 46.0%
  95% CI: [32.1%, 60.5%]
  Benign refused: 23/50

Adaptive Alpha Distribution:
  ATTACK: 187 (62.5%)
  SAFE: 45 (15.1%)
  WARN: 67 (22.4%)

Average generation time: 2.34s

======================================================================
TABLE 2 ENTRY
======================================================================

Full AEGIS (adaptive α)  |  9.2%  |  46.0%  |  Adaptive

Use these numbers for the paper!
======================================================================
```

## Using the Results

Once you have the output:

1. **Take the ASR and ORR values** from the "TABLE 2 ENTRY" section
2. **Add a row to Table 2** in the canonical paper:

```markdown
| Full AEGIS (adaptive α) | 9.2% | 46.0% | ~82% (relative to baseline) |
```

3. **Update the table caption** to remove "internal development experiments" caveat
4. **Update abstract if needed** to reflect final numbers

## Troubleshooting

### Error: "Can't find steering_vectors.safetensors"

Make sure the RepE vectors path points to the actual `.safetensors` file:

```bash
--repe-vectors ./outputs/repe/steering_vectors.safetensors
```

### Error: "CUDA out of memory"

The full AEGIS system loads 3 models. Try:

1. Use smaller batch processing (script already does this)
2. Free GPU memory: `nvidia-smi` to check usage
3. Or evaluate on CPU (slower): Set `device="cpu"` in script

### Error: "Cannot load sidecar adapter"

If sidecar is a PEFT adapter, make sure the `adapter_config.json` exists:

```bash
ls ./outputs/aegis-sidecar/adapter_config.json
```

### Judge Errors

If LLM judge fails:

1. Check you have `ANTHROPIC_API_KEY` set: `echo $ANTHROPIC_API_KEY`
2. Or use rule-based judge: `--judge rule` (faster, deterministic)

## Performance Notes

- **Full evaluation (299 samples)** takes ~10-15 minutes on A100/H100
- **Ensemble judge** is recommended for accuracy (uses LLM for uncertain cases)
- **Rule-based judge** is faster (~5 minutes) but slightly less accurate

## Next Steps

After running the evaluation:

1. **Record the exact ASR and ORR** from the output
2. **Add the row to Table 2** in `paper/AEGIS_Canonical.md`
3. **Run the command and share the output** with me
4. I'll update the paper with the final numbers

Let me know when you have the results! 🎯
