#!/bin/bash
# =============================================================================
# GSM-Infinity Pre-training Run Script (Paper §3 Skewed Distribution)
# =============================================================================
# Pre-trains a 100M Qwen2 model on 10B tokens of GSM-Infinity data
# Data mix: 20% op=2-4, 30% op=5-7, 50% op=8-10 (matching Table 2 §3)
# Operation range: 2-10 | Templates: All (zoo, teacher, movie)
# Hardware: 8x H100 GPUs
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG="${PROJECT_ROOT}/LLaMA-Factory/examples/gsm_infinity/pt_op2-10_10B_alltemps_skewed.yaml"
RUN_NAME="pt_op2-10_10B_alltemps_skewed_$(date +%Y%m%d_%H%M%S)"

# GPU Configuration - 8x H100 GPUs
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Wandb Configuration
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-pretrain}"
export WANDB_RUN_NAME="${RUN_NAME}"

# Skip LLaMA-Factory's strict transformers version check.
export DISABLE_VERSION_CHECK=1

# Redirect all HuggingFace / temp caches to /fast to avoid filling up ~/ (500GB limit)
export HF_HOME="/fast/pmayilvahanan/.cache/huggingface"
export HF_DATASETS_CACHE="/fast/pmayilvahanan/.cache/huggingface/datasets"
export TMPDIR="/fast/pmayilvahanan/tmp"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}"

# =============================================================================
# Environment Setup
# =============================================================================
echo "=============================================="
echo "GSM-Infinity Pre-training (Paper §3 Skewed)"
echo "=============================================="
echo "Run name: ${RUN_NAME}"
echo "Config: ${CONFIG}"
echo "GPUs: ${GPU_LIST}"
echo ""
echo "Data distribution (Paper Table 2 §3):"
echo "  - 20% easy  (op=2-4):  ~0.067 per op -> ~2B tokens total"
echo "  - 30% medium (op=5-7): ~0.100 per op -> ~3B tokens total"
echo "  - 50% hard  (op=8-10): ~0.167 per op -> ~5B tokens total"
echo ""
echo "Hyperparameters:"
echo "  - Learning rate: 1e-4"
echo "  - Min LR: 3e-5 (cosine decay)"
echo "  - Batch size: 64 * 8 GPUs * 2048 = ~1M tokens/step"
echo "  - Weight decay: 0.1"
echo "  - Warmup: 5%"
echo ""

# Load modules
module load cuda/12.1
module load cudnn/8.9.1-cu12.x

# Activate virtual environment
source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Derive NPROC from GPU_LIST
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT}"

# H100 optimizations
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# =============================================================================
# Pre-flight Checks
# =============================================================================
echo "[Check] Verifying configuration..."

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

# Check PRESET.json exists and has the correct entry
PRESET="${PROJECT_ROOT}/data/PRESET.json"
if [[ ! -f "${PRESET}" ]]; then
    echo "ERROR: PRESET.json not found: ${PRESET}"
    exit 1
fi
echo "[Check] PRESET.json found: ${PRESET}"

# Check if data exists
DATA_DIR="${PROJECT_ROOT}/data/composition_hf/train"
if [[ ! -d "${DATA_DIR}" ]]; then
    echo ""
    echo "WARNING: Training data not found at ${DATA_DIR}"
    echo "You need to prepare the data first. Run:"
    echo "  python ${PROJECT_ROOT}/scripts/download_gsm_infinity_data.py"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python environment
echo "[Check] Python environment:"
which python
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

echo ""
echo "[Info] Starting training with ${NPROC} GPU(s): ${GPU_LIST}"
echo "[Info] Expected steps: ~10,000 (10B tokens / 1M tokens per step)"
echo ""

# =============================================================================
# Run Training
# =============================================================================
cd "${PROJECT_ROOT}/LLaMA-Factory"

llamafactory-cli train "${CONFIG}" \
    run_name="${RUN_NAME}" \
    output_dir="saves/gsm_infinity/${RUN_NAME}"

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Output saved to: saves/gsm_infinity/${RUN_NAME}"
