#!/bin/bash
# =============================================================================
# TinyStories Pre-training Run Script for DLLM (MDLM)
# =============================================================================
# Trains A2D-MDLM 100M on TinyStories for 4 epochs using the n-gram paper's
# SentencePiece tokenizer (vocab 32768), enabling direct comparison against
# pre-computed n-gram rules from gs://transformer-ngrams/TinyStories/
#
# Prerequisites:
#   1. Download tokenizer.model into dllm/model_configs/a2d_qwen2_tinystories_100M/
#      (see that directory's README.md)
#   2. Download training parquets from gs://transformer-ngrams/TinyStories/training_data/
#      to $TRAINING_DATA_DIR
#
# Usage:
#   bash dllm/examples/tinystories/run_pretrain.sh
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT="/home/fli/dllm/Interplay-LM-Reasoning"
FAST_ROOT="/fast/fli/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="/fast/fli/Interplay-LM-Reasoning/gsm_pretrain/bin/activate"

# Path to downloaded TinyStories training parquets
TRAINING_DATA_DIR="${TRAINING_DATA_DIR:-${FAST_ROOT}/data/tinystories/training_data}"

# GPU Configuration
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"

# Accelerate config
ACCEL_CONFIG="${ACCEL_CONFIG:-ddp}"

# Wandb
export WANDB_PROJECT="${WANDB_PROJECT:-dllm-tinystories-ngram}"

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
# Pre-flight checks
# =============================================================================
TOKENIZER_MODEL="${FAST_ROOT}/dllm/model_configs/a2d_qwen2_tinystories_100M/tokenizer.model"
if [[ ! -f "${TOKENIZER_MODEL}" ]]; then
    echo "ERROR: tokenizer.model not found at ${TOKENIZER_MODEL}"
    echo "Download it first — see dllm/model_configs/a2d_qwen2_tinystories_100M/README.md"
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
RUN_NAME="a2d_mdlm_tinystories_100M_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${FAST_ROOT}/saves/tinystories/${RUN_NAME}"

echo "=============================================="
echo "Training A2D-MDLM 100M on TinyStories"
echo "Run name: ${RUN_NAME}"
echo "Training data: ${TRAINING_DATA_DIR}"
echo "GPUs: ${NPROC}, Accel config: ${ACCEL_CONFIG}"
echo "=============================================="

accelerate launch \
    --config_file "scripts/accelerate_configs/${ACCEL_CONFIG}.yaml" \
    --num_processes "${NPROC}" \
    examples/tinystories/pt_mdlm.py \
    --model_name_or_path "${PROJECT_ROOT}/dllm/model_configs/a2d_qwen2_tinystories_100M" \
    --training_data_dir "${TRAINING_DATA_DIR}" \
    --max_length 2048 \
    --num_train_epochs 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_grad_norm 1.0 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --bf16 True \
    --gradient_checkpointing False \
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
