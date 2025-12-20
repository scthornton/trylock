# Run TRYLOCK Evaluation on GCP VM

## Quick Start

### 1. Start Your VM
```bash
# Turn on your V100 or A100 instance
gcloud compute instances start <your-vm-name> --zone=<your-zone>

# SSH into it
gcloud compute ssh <your-vm-name> --zone=<your-zone>
```

### 2. Setup Environment (First Time Only)
```bash
# Install dependencies
pip install transformers peft safetensors accelerate bitsandbytes tqdm torch

# Or if you have a requirements.txt
pip install -r requirements.txt
```

### 3. Upload Files
```bash
# From your local machine (in trylock directory)
gcloud compute scp scripts/run_full_evaluation.py <your-vm>:~/ --zone=<your-zone>
gcloud compute scp data/dpo/test.jsonl <your-vm>:~/ --zone=<your-zone>
```

### 4. Run Evaluation
```bash
# On the VM
python run_full_evaluation.py --test-file test.jsonl --output eval_full_trylock.json

# Or with full precision (if you have enough memory)
python run_full_evaluation.py --test-file test.jsonl --output eval_full_trylock.json --no-8bit
```

### 5. Download Results
```bash
# From your local machine
gcloud compute scp <your-vm>:~/eval_full_trylock.json ./Downloads/ --zone=<your-zone>
```

### 6. Shutdown VM
```bash
gcloud compute instances stop <your-vm-name> --zone=<your-zone>
```

---

## Alternative: Run in Background

If you want to disconnect while it runs:

```bash
# Start in screen/tmux
screen -S trylock
python run_full_evaluation.py --test-file test.jsonl

# Detach: Ctrl+A then D

# Reconnect later
screen -r trylock
```

---

## Expected Runtime

- **V100**: ~10-15 minutes (299 samples)
- **A100**: ~5-10 minutes (faster generation)
- **CPU**: ~10-15 hours (not recommended)

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Use 8-bit quantization (default)
python run_full_evaluation.py --test-file test.jsonl
```

### Missing Dependencies
```bash
pip install transformers peft safetensors accelerate bitsandbytes tqdm torch
```

### File Not Found
```bash
# Check test.jsonl is in current directory
ls -lh test.jsonl

# Or specify full path
python run_full_evaluation.py --test-file /path/to/data/dpo/test.jsonl
```

---

## What You'll See

```
Loading TRYLOCK models...
8-bit quantization: True
GPU available: True
GPU: Tesla V100-SXM2-16GB
GPU Memory: 16.0 GB

[1/3] Loading DPO model...
✓ DPO loaded

[2/3] Loading RepE vectors...
✓ 8 vectors loaded

[3/3] Loading sidecar...
✓ Sidecar loaded

✅ All models ready!

Loading test data from test.jsonl...
Loaded 299 test samples

Evaluating: 100%|████████████████████| 299/299 [12:34<00:00,  2.52s/it]

============================================================
TRYLOCK FULL SYSTEM EVALUATION RESULTS
============================================================

Samples: 299 (XXX attacks, YYY benign)

ASR (Attack Success Rate): XX.X%
ORR (Over-Refusal Rate): YY.Y%

============================================================

✅ Results saved to eval_full_trylock.json
```

---

## Next Steps

Once you have the results:
1. Copy the ASR and ORR numbers
2. Add them to Table 2 in the paper
3. Polish the paper for submission
