#!/bin/bash
#
# Wrapper script to run AEGIS evaluation with proper environment settings
# This avoids macOS torch/transformers segfaults
#

set -e

# Disable MPS/Metal backend (causes segfaults on macOS)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Force CPU only
export CUDA_VISIBLE_DEVICES=""

# Disable tokenizers parallelism (can cause hangs)
export TOKENIZERS_PARALLELISM=false

# Set threading
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "=========================================="
echo "AEGIS Evaluation (CPU Mode)"
echo "=========================================="
echo ""
echo "Environment:"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=1"
echo "  CUDA_VISIBLE_DEVICES="
echo "  TOKENIZERS_PARALLELISM=false"
echo ""
echo "This will run on CPU (slow but stable)"
echo ""

# Default arguments
TEST_FILE="${TEST_FILE:-data/dpo/test.jsonl}"
DPO_ADAPTER="${DPO_ADAPTER:-scthornton/aegis-mistral-7b-dpo}"
REPE_VECTORS="${REPE_VECTORS:-./steering_vectors.safetensors}"
SIDECAR="${SIDECAR:-scthornton/aegis-sidecar-classifier}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
OUTPUT="${OUTPUT:-eval_results.json}"

# Allow overriding via arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-file)
            TEST_FILE="$2"
            shift 2
            ;;
        --dpo-adapter)
            DPO_ADAPTER="$2"
            shift 2
            ;;
        --repe-vectors)
            REPE_VECTORS="$2"
            shift 2
            ;;
        --sidecar)
            SIDECAR="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Test file: $TEST_FILE"
echo "  DPO adapter: $DPO_ADAPTER"
echo "  RepE vectors: $REPE_VECTORS"
echo "  Sidecar: $SIDECAR"
echo "  Max samples: $MAX_SAMPLES"
echo "  Output: $OUTPUT"
echo ""
echo "Starting evaluation..."
echo ""

# Run the evaluation
python scripts/eval_cpu_only.py \
    --test-file "$TEST_FILE" \
    --dpo-adapter "$DPO_ADAPTER" \
    --repe-vectors "$REPE_VECTORS" \
    --sidecar "$SIDECAR" \
    --max-samples "$MAX_SAMPLES" \
    --output "$OUTPUT"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT"
echo "=========================================="
