#!/bin/bash
# =============================================================================
# GSM-Infinity Pre-training — 400M Mamba-2 (SSM)
# =============================================================================
# Architecture: dim=1024, n_layers=150, n_heads=16, state_dim=128, conv_size=4
# ~399M params (weight_tying=True)
# Data: 10B tokens of GSM-Infinity (op range 2-10, all templates)
# Hardware: 8x H100 GPUs
# Effective batch: 8/GPU × 8 accum × 8 GPUs × 2048 tokens ≈ 1M tokens/step
#
# Usage:
#   bash lingua/apps/mamba/gsm_infinity/run_pretrain_400M.sh           # full run
#   bash lingua/apps/mamba/gsm_infinity/run_pretrain_400M.sh preprocess # data only
#
# Environment variables:
#   GPU_LIST       GPU IDs (default: 0,1,2,3,4,5,6,7)
#   TOKEN_BUDGET   Token budget (default: 10B)
#   WANDB_PROJECT  Wandb project name (default: lingua-gsm-infinity)
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU Configuration
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"

# Data
TOKEN_BUDGET="${TOKEN_BUDGET:-10B}"
RAW_DATA="${PROJECT_ROOT}/data/composition_hf/train"
TOKENIZER_PATH="${PROJECT_ROOT}/model_configs/qwen2_400M"
PROCESSED_DATA="${PROJECT_ROOT}/data/composition_lingua/gsm_infinity"

# Config
CONFIG="${SCRIPT_DIR}/configs/mamba_400M_gsm.yaml"
RUN_NAME="mamba_400M_gsm_$(date +%Y%m%d_%H%M%S)"
DUMP_DIR="${LINGUA_ROOT}/saves/gsm_infinity/${RUN_NAME}"

# Wandb
export WANDB_PROJECT="${WANDB_PROJECT:-lingua-gsm-infinity}"

# =============================================================================
# Environment Setup
# =============================================================================
setup_env() {
    echo "=============================================="
    echo "GSM-Infinity Pre-training — 400M Mamba-2 (SSM)"
    echo "=============================================="
    echo "Run name:   ${RUN_NAME}"
    echo "Config:     ${CONFIG}"
    echo "GPUs:       ${GPU_LIST} (${NPROC} processes)"
    echo "Dump dir:   ${DUMP_DIR}"
    echo ""

    module load cuda/12.1 2>/dev/null || true
    module load cudnn/8.9.1-cu12.x 2>/dev/null || true

    source "${VENV}"
    export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-29600}"

    echo "[Setup] Python: $(which python)"
    python --version
}

# =============================================================================
# Step 1: Preprocess data (shared across model sizes)
# =============================================================================
preprocess() {
    if [[ -d "${PROCESSED_DATA}" ]] && ls "${PROCESSED_DATA}"/gsm_infinity.chunk.*.jsonl 1>/dev/null 2>&1; then
        echo "[Preprocess] Data already exists at ${PROCESSED_DATA}, skipping."
        return
    fi

    echo "=============================================="
    echo "Preprocessing GSM-Infinity data -> ${PROCESSED_DATA}"
    echo "Budget: ${TOKEN_BUDGET}, Tokenizer: ${TOKENIZER_PATH}"
    echo "=============================================="

    python -m apps.gsm_infinity.preprocess_data \
        --raw_data_dir "${RAW_DATA}" \
        --output_dir "${PROCESSED_DATA}" \
        --op_min 2 --op_max 10 \
        --token_budget "${TOKEN_BUDGET}" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --lines_per_chunk 10000
}

# =============================================================================
# Step 2: Train
# =============================================================================
train() {
    echo "=============================================="
    echo "Training Mamba-2 400M on GSM-Infinity"
    echo "  dim=1024, layers=150, heads=16, state_dim=128, conv=4"
    echo "  batch=8/GPU × accum=8 × ${NPROC} GPUs × 2048 tokens"
    echo "=============================================="

    mkdir -p "${DUMP_DIR}"

    cd "${LINGUA_ROOT}"
    torchrun \
        --nproc_per_node="${NPROC}" \
        --master_addr="${MASTER_ADDR}" \
        --master_port="${MASTER_PORT}" \
        -m apps.mamba.train \
        config="${CONFIG}" \
        name="${RUN_NAME}" \
        dump_dir="${DUMP_DIR}" \
        data.root_dir="${PROJECT_ROOT}/data/composition_lingua" \
        data.tokenizer.path="${TOKENIZER_PATH}"

    echo ""
    echo "=============================================="
    echo "Training complete!"
    echo "Output: ${DUMP_DIR}"
    echo "=============================================="
}

# =============================================================================
# Main dispatch
# =============================================================================
COMMAND="${1:-train}"

case "${COMMAND}" in
    preprocess)
        setup_env
        cd "${LINGUA_ROOT}"
        preprocess
        ;;
    train)
        setup_env
        cd "${LINGUA_ROOT}"
        preprocess
        train
        ;;
    *)
        echo "Usage: $0 {preprocess|train}"
        echo ""
        echo "Commands:"
        echo "  preprocess  Convert composition_hf data to lingua format (run once)"
        echo "  train       Preprocess (if needed) + train Mamba-2 400M"
        ;;
esac
