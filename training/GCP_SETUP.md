# TRYLOCK Model Training - GCP Setup Guide

## Recommended VM Configuration

### Option 1: A100 (Best for speed)
```
Machine type: a2-highgpu-1g
GPU: 1x NVIDIA A100 40GB
vCPUs: 12
RAM: 85 GB
Disk: 200 GB SSD
Cost: ~$3.67/hr (on-demand), ~$1.10/hr (spot)
```

### Option 2: L4 (Good balance)
```
Machine type: g2-standard-8
GPU: 1x NVIDIA L4 24GB
vCPUs: 8
RAM: 32 GB
Disk: 200 GB SSD
Cost: ~$0.70/hr (on-demand), ~$0.21/hr (spot)
```

### Option 3: T4 (Budget)
```
Machine type: n1-standard-8 + T4
GPU: 1x NVIDIA T4 16GB
vCPUs: 8
RAM: 30 GB
Disk: 200 GB SSD
Cost: ~$0.35/hr (on-demand)
Note: Will need QLoRA due to limited VRAM
```

## Quick Setup Commands

### 1. Create VM (using gcloud CLI)

```bash
# A100 option (recommended)
gcloud compute instances create trylock-training \
    --zone=us-central1-a \
    --machine-type=a2-highgpu-1g \
    --accelerator=type=nvidia-tesla-a100,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"

# L4 option (cheaper)
gcloud compute instances create trylock-training \
    --zone=us-central1-a \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE
```

### 2. SSH into VM

```bash
gcloud compute ssh trylock-training --zone=us-central1-a
```

### 3. Setup Environment

```bash
# Clone/upload TRYLOCK
git clone https://github.com/scthornton/trylock.git
cd trylock

# Or upload from local
# gcloud compute scp --recurse ~/perfecxion/datasets/trylock trylock-training:~ --zone=us-central1-a

# Install dependencies
pip install -r training/requirements.txt
```

### 4. Verify GPU

```bash
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### 5. Start Training

```bash
cd trylock
python training/train_dpo.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --data_path data/dpo \
    --output_dir outputs/trylock-llama-8b \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-7
```

## Cost Estimation

| GPU | Training Time (est.) | Cost (spot) | Cost (on-demand) |
|-----|---------------------|-------------|------------------|
| A100 | ~2-3 hours | ~$3-4 | ~$10-12 |
| L4 | ~4-6 hours | ~$1-2 | ~$4-5 |
| T4 | ~8-12 hours | ~$3-4 | ~$4-5 |

## Cleanup

```bash
# Stop VM (keeps disk, no GPU charges)
gcloud compute instances stop trylock-training --zone=us-central1-a

# Delete VM completely
gcloud compute instances delete trylock-training --zone=us-central1-a
```

## Tips

1. **Use spot/preemptible instances** - 70% cheaper, may get terminated
2. **Save checkpoints frequently** - In case of preemption
3. **Use tmux/screen** - Keep training running if SSH disconnects
4. **Download model after training** - Before deleting VM
