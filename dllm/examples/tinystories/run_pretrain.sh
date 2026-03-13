#!/bin/bash
# =============================================================================
# TinyStories Pre-training Run Script for DLLM (MDLM)
# =============================================================================
# Trains A2D-MDLM on TinyStories using the n-gram paper's SentencePiece
# tokenizer (vocab 32768), enabling direct comparison against pre-computed
# n-gram rules from gs://transformer-ngrams/TinyStories/
#
# Prerequisites:
#   1. Download tokenizer.model into the relevant model_configs directory
#      (see that directory's README.md)
#   2. Download training parquets from gs://transformer-ngrams/TinyStories/training_data/
#      to $TRAINING_DATA_DIR
#
# Usage:
#   bash dllm/examples/tinystories/run_pretrain.sh 100M
#   bash dllm/examples/tinystories/run_pretrain.sh 200M
#   bash dllm/examples/tinystories/run_pretrain.sh 400M
#
# Environment variables:
#   GPU_LIST           GPU IDs (default: 0,1,2,3,4,5,6,7)
#   ACCEL_CONFIG       Accelerate config name (default: ddp)
#   TRAINING_DATA_DIR  Path to TinyStories parquets
#   SEED               Random seed (default: 42)
#   WANDB_PROJECT      W&B project name (default: dllm-tinystories-ngram)
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT="/home/fli/dllm/Interplay-LM-Reasoning"
FAST_ROOT="/fast/fli/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="/fast/fli/Interplay-LM-Reasoning/gsm_pretrain/bin/activate"

TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-${FAST_ROOT}/data/tinystories/training_data}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"

ACCEL_CONFIG="${ACCEL_CONFIG:-ddp}"

export WANDB_PROJECT="${WANDB_PROJECT:-dllm-tinystories-ngram}"

# =============================================================================
# Per-size batch configuration
# (keeps effective batch = 64 * seq_len tokens/step across all sizes)
# =============================================================================
get_batch_config() {
    local SIZE="$1"
    case "${SIZE}" in
        100M) BATCH_SIZE=64; GRAD_ACCUM=1; GRAD_CKPT="False" ;;
        200M) BATCH_SIZE=32; GRAD_ACCUM=2; GRAD_CKPT="True"  ;;
        400M) BATCH_SIZE=16; GRAD_ACCUM=4; GRAD_CKPT="True"  ;;
        *)
            echo "Error: Unknown size '${SIZE}'. Use 100M, 200M, or 400M."
            exit 1
            ;;
    esac
}

# =============================================================================
# Environment Setup
# =============================================================================
echo "[Setup] Activating environment..."
source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
cd "${DLLM_ROOT}"

echo "[Setup] Python: $(which python)"
echo "[Setup] GPUs: ${GPU_LIST} (${NPROC} processes)"

# =============================================================================
# Parse args
# =============================================================================
SIZE="${1:-100M}"
shift 1 2>/dev/null || true

get_batch_config "${SIZE}"

MODEL_CONFIG="${PROJECT_ROOT}/dllm/model_configs/a2d_qwen2_tinystories_${SIZE}"

# =============================================================================
# Pre-flight checks
# =============================================================================
TOKENIZER_MODEL="${MODEL_CONFIG}/tokenizer.model"
if [[ ! -f "${TOKENIZER_MODEL}" ]]; then
    echo "ERROR: tokenizer.model not found at ${TOKENIZER_MODEL}"
    echo "Download it first — see ${MODEL_CONFIG}/README.md"
    exit 1
fi

if [[ ! -d "${TRAINING_DATA_DIR}" ]]; then
    echo "ERROR: Training data not found at ${TRAINING_DATA_DIR}"
    echo "Download from gs://transformer-ngrams/TinyStories/training_data/"
    echo "Set TRAINING_DATA_DIR env var if stored elsewhere."
    exit 1
fi

# =============================================================================
# Train
# =============================================================================
SEED="${SEED:-42}"
RUN_NAME="a2d_mdlm_tinystories_${SIZE}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${FAST_ROOT}/saves/tinystories/${RUN_NAME}"

echo "=============================================="
echo "Training A2D-MDLM ${SIZE} on TinyStories"
echo "  Model config: ${MODEL_CONFIG}"
echo "  Batch: ${BATCH_SIZE}/GPU × accum=${GRAD_ACCUM} × ${NPROC} GPUs"
echo "  Grad checkpointing: ${GRAD_CKPT}"
echo "  Training data: ${TRAINING_DATA_DIR}"
echo "  Run name: ${RUN_NAME}"
echo "=============================================="

accelerate launch \
    --config_file "scripts/accelerate_configs/${ACCEL_CONFIG}.yaml" \
    --num_processes "${NPROC}" \
    examples/tinystories/pt_mdlm.py \
    --model_name_or_path "${MODEL_CONFIG}" \
    --training_data_dir "${TRAINING_DATA_DIR}" \
    --max_length 2048 \
    --num_train_epochs 8 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --gradient_checkpointing "${GRAD_CKPT}" \
    --bf16 True \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 50 \
    --eval_strategy "no" \
    --report_to wandb \
    --run_name "${RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    "$@"

echo "Training complete! Output: ${OUTPUT_DIR}"
