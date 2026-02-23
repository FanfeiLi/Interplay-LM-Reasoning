#!/bin/bash
# =============================================================================
# GSM-Infinity Pre-training — 200M Transformer (Next-Token Prediction)
# =============================================================================
# Architecture: 768 hidden × 24 layers × 3072 FFN ≈ 205M params
# Data: 10B tokens of GSM-Infinity (op range 2-10, all templates)
# Hardware: 8x H100 GPUs
# Effective batch: 32 per GPU × 2 accum × 8 GPUs × 2048 tokens ≈ 1M tokens/step
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG="${PROJECT_ROOT}/LLaMA-Factory/examples/gsm_infinity/pt_200M.yaml"
RUN_NAME="pt_200M_ar_$(date +%Y%m%d_%H%M%S)"

# GPU Configuration
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Wandb Configuration
export WANDB_PROJECT="${WANDB_PROJECT:-gsm-infinity-pretrain}"
export WANDB_RUN_NAME="${RUN_NAME}"

export DISABLE_VERSION_CHECK=1

# =============================================================================
# Environment Setup
# =============================================================================
echo "=============================================="
echo "GSM-Infinity Pre-training — 200M AR / NTP"
echo "=============================================="
echo "Run name:   ${RUN_NAME}"
echo "Config:     ${CONFIG}"
echo "GPUs:       ${GPU_LIST}"
echo ""
echo "Architecture:"
echo "  hidden_size=768, num_layers=24, intermediate_size=3072"
echo "  num_heads=12, kv_heads=2  →  ~205M params"
echo ""
echo "Hyperparameters:"
echo "  lr=1e-4, min_lr=3e-5 (cosine), weight_decay=0.1, warmup=5%"
echo "  batch=32/GPU × accum=2 × 8 GPUs × 2048 tokens ≈ 1M tokens/step"
echo ""

module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9.1-cu12.x 2>/dev/null || true

source "${PROJECT_ROOT}/gsm_pretrain/bin/activate"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT}"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# =============================================================================
# Pre-flight Checks
# =============================================================================
if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

MODEL_CONFIG="${PROJECT_ROOT}/model_configs/qwen2_200M/config.json"
if [[ ! -f "${MODEL_CONFIG}" ]]; then
    echo "ERROR: Model config not found: ${MODEL_CONFIG}"
    exit 1
fi

DATA_DIR="${PROJECT_ROOT}/data/composition_hf"
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "WARNING: Training data not found at ${DATA_DIR}"
    echo "Prepare data first, then re-run."
    exit 1
fi

echo "[Check] Python: $(which python)"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
echo ""
echo "[Info] Starting training with ${NPROC} GPU(s)"
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
echo "Output saved to: LLaMA-Factory/saves/gsm_infinity/${RUN_NAME}"
echo "=============================================="
