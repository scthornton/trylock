# AEGIS Multi-Stage Dockerfile
# Supports: gateway, sidecar, training

ARG BASE_IMAGE=python:3.11-slim

# ============================================
# Base Stage - Common dependencies
# ============================================
FROM ${BASE_IMAGE} AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# ============================================
# Gateway Stage - Security Gateway Service
# ============================================
FROM base AS gateway

EXPOSE 8080

ENV AEGIS_COMPONENT=gateway
ENV GATEWAY_HOST=0.0.0.0
ENV GATEWAY_PORT=8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "deployment.gateway", \
     "--host", "0.0.0.0", \
     "--port", "8080"]

# ============================================
# Sidecar Stage - Classifier Service
# ============================================
FROM base AS sidecar

EXPOSE 8081

ENV AEGIS_COMPONENT=sidecar
ENV SIDECAR_HOST=0.0.0.0
ENV SIDECAR_PORT=8081

# Default model path (override with volume mount)
ENV MODEL_PATH=/models/aegis-sidecar

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

CMD ["python", "-m", "deployment.sidecar_service", \
     "--host", "0.0.0.0", \
     "--port", "8081", \
     "--model", "/models/aegis-sidecar"]

# ============================================
# GPU Stage - For training and GPU inference
# ============================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS gpu-base

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install PyTorch with CUDA
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# ============================================
# Training Stage - For model training
# ============================================
FROM gpu-base AS training

ENV AEGIS_COMPONENT=training

# Install training-specific dependencies
RUN pip install --no-cache-dir \
    accelerate \
    bitsandbytes \
    deepspeed \
    wandb

CMD ["python", "-m", "training.sft_warmup", "--help"]

# ============================================
# Sidecar GPU Stage - GPU-accelerated classifier
# ============================================
FROM gpu-base AS sidecar-gpu

EXPOSE 8081

ENV AEGIS_COMPONENT=sidecar-gpu
ENV SIDECAR_HOST=0.0.0.0
ENV SIDECAR_PORT=8081
ENV MODEL_PATH=/models/aegis-sidecar

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

CMD ["python", "-m", "deployment.sidecar_service", \
     "--host", "0.0.0.0", \
     "--port", "8081", \
     "--model", "/models/aegis-sidecar", \
     "--device", "cuda"]
