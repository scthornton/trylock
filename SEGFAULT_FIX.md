# Segfault Fix Guide

## Problem

You're getting segmentation faults when running Python scripts that import `transformers`. This is a **known issue on macOS** with:
- torch 2.8.0
- transformers 4.57.1
- macOS Metal/MPS backend

## Root Cause

The segfault happens because `transformers` tries to initialize the MPS (Metal Performance Shaders) backend on macOS, which crashes with certain torch/transformers version combinations.

## Solutions (Try in Order)

### Solution 1: Use the Wrapper Script (Easiest)

I've created a wrapper that sets the correct environment variables:

```bash
./scripts/run_eval.sh \
    --test-file data/dpo/test.jsonl \
    --dpo-adapter scthornton/trylock-mistral-7b-dpo \
    --repe-vectors ./path/to/steering_vectors.safetensors \
    --sidecar scthornton/trylock-sidecar-classifier \
    --max-samples 10 \
    --output eval_results.json
```

This script:
- Disables MPS/Metal backend
- Forces CPU-only execution
- Sets safe threading limits
- Runs evaluation in stable mode

### Solution 2: Downgrade transformers (Recommended)

The issue is fixed in transformers 4.45.x:

```bash
pip install transformers==4.45.2 --force-reinstall
```

Then retry:

```bash
python scripts/eval_cpu_only.py \
    --test-file data/dpo/test.jsonl \
    --dpo-adapter scthornton/trylock-mistral-7b-dpo \
    --repe-vectors ./path/to/steering_vectors.safetensors \
    --sidecar scthornton/trylock-sidecar-classifier \
    --max-samples 10
```

### Solution 3: Manual Environment Variables

Set these before running Python:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false

python scripts/eval_cpu_only.py [args...]
```

### Solution 4: Use Google Colab (If Local Fails)

If nothing works locally, use Colab:

1. Upload `eval_cpu_only.py` to Colab
2. Install requirements:
   ```python
   !pip install transformers==4.45.2 peft safetensors
   ```
3. Run evaluation with GPU (much faster)

### Solution 5: Reinstall PyTorch (Nuclear Option)

```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.4.1 torchvision torchaudio
pip install transformers==4.45.2 --force-reinstall
```

## Quick Test

After trying a solution, test if it worked:

```bash
python -c "import transformers; print('SUCCESS:', transformers.__version__)"
```

If this prints `SUCCESS: 4.45.2` (or similar) without crashing, the segfault is fixed.

## Expected Behavior

When evaluation runs successfully, you should see:

```
==========================================
TRYLOCK Evaluation (CPU Mode)
==========================================

Environment:
  PYTORCH_ENABLE_MPS_FALLBACK=1
  CUDA_VISIBLE_DEVICES=
  TOKENIZERS_PARALLELISM=false

[1/3] Loading DPO model...
   ✓ DPO model loaded
[2/3] Loading RepE vectors...
   ✓ 8 vectors loaded
[3/3] Loading sidecar...
   ✓ Sidecar loaded

Evaluating 10 samples...
[Progress bar]

==========================================
RESULTS
==========================================

ASR: 12.5%
ORR: 50.0%
```

## Performance Note

**CPU evaluation is SLOW** (~2-3 minutes per sample). For full 299 samples:
- CPU: ~10-15 hours
- GPU (Colab/Cloud): ~10-15 minutes

## Recommended Path Forward

1. **Try Solution 2** (downgrade transformers) - This is the cleanest fix
2. **If that works**, run quick test (10 samples) to verify everything loads
3. **If quick test succeeds**, run full evaluation overnight or use cloud GPU

## Need Help?

If you're still getting segfaults after trying all solutions:

1. Check your exact error message
2. Share output of: `pip list | grep -E "(torch|transformers|peft)"`
3. Share your macOS version: `sw_vers`

I can then provide more specific guidance.
