#!/bin/bash
# =============================================================================
# FineWeb-Edu Pre-training — Diffusion Models (A2D-MDLM / A2D-BD3LM)
# =============================================================================
# Pre-trains diffusion language models on FineWeb-Edu (streamed from HF).
# Uses the full Qwen2.5 tokenizer (151K vocab) for fair comparison with
# AR baselines trained in lingua.
#
# Usage:
#   # Train MDLM at various sizes
#   bash dllm/examples/fineweb/run_pretrain.sh mdlm 100M
#   bash dllm/examples/fineweb/run_pretrain.sh mdlm 200M
#   bash dllm/examples/fineweb/run_pretrain.sh mdlm 400M
#
#   # Train BD3LM at various sizes
#   bash dllm/examples/fineweb/run_pretrain.sh bd3lm 100M
#   bash dllm/examples/fineweb/run_pretrain.sh bd3lm 400M
#
# Environment variables:
#   GPU_LIST       GPU IDs (default: 0,1,2,3,4,5,6,7)
#   ACCEL_CONFIG   Accelerate config (default: zero2)
#   BLOCK_SIZE     BD3LM block size (default: 16)
#   MAX_STEPS      Training steps (default: 10000)
#   WANDB_PROJECT  Wandb project (default: dllm-fineweb)
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT="/fast/fli/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="/fast/fli/Interplay-LM-Reasoning/gsm_pretrain/bin/activate"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"
ACCEL_CONFIG="${ACCEL_CONFIG:-zero2}"
MAX_STEPS="${MAX_STEPS:-10000}"

export WANDB_PROJECT="${WANDB_PROJECT:-dllm-fineweb}"

# =============================================================================
# Environment Setup
# =============================================================================
setup_env() {
    echo "[Setup] Activating environment..."
    source "${VENV}"
    export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
    export HF_HOME="${PROJECT_ROOT}/.hf_cache"
    export HF_DATASETS_CACHE="${PROJECT_ROOT}/.hf_cache/datasets"
    cd "${DLLM_ROOT}"
    echo "[Setup] Python: $(which python)"
    echo "[Setup] GPUs:   ${GPU_LIST} (${NPROC} processes)"
}

# =============================================================================
# Per-size batch configuration
# =============================================================================
get_batch_config() {
    local SIZE="$1"
    local METHOD="$2"

    case "${SIZE}" in
        100M)
            if [[ "${METHOD}" == "mdlm" ]]; then
                BATCH_SIZE=64; GRAD_ACCUM=1; GRAD_CKPT="False"
            else
                BATCH_SIZE=32; GRAD_ACCUM=2; GRAD_CKPT="False"
            fi
            ;;
        200M)
            if [[ "${METHOD}" == "mdlm" ]]; then
                BATCH_SIZE=32; GRAD_ACCUM=2; GRAD_CKPT="True"
            else
                BATCH_SIZE=16; GRAD_ACCUM=4; GRAD_CKPT="True"
            fi
            ;;
        400M)
            if [[ "${METHOD}" == "mdlm" ]]; then
                BATCH_SIZE=16; GRAD_ACCUM=4; GRAD_CKPT="True"
            else
                BATCH_SIZE=8; GRAD_ACCUM=8; GRAD_CKPT="True"
            fi
            ;;
        *)
            echo "Error: Unknown size '${SIZE}'. Use 100M, 200M, or 400M."
            exit 1
            ;;
    esac
}

# =============================================================================
# Train MDLM
# =============================================================================
train_mdlm() {
    local SIZE="$1"
    shift
    get_batch_config "${SIZE}" "mdlm"

    local MODEL_CONFIG="${DLLM_ROOT}/model_configs/a2d_qwen2_fineweb_${SIZE}"
    local RUN_NAME="a2d_mdlm_fineweb_${SIZE}_$(date +%Y%m%d_%H%M%S)"
    local OUTPUT_DIR="${DLLM_ROOT}/saves/fineweb/${RUN_NAME}"

    echo "=============================================="
    echo "Training A2D-MDLM ${SIZE} on FineWeb-Edu"
    echo "  Model config: ${MODEL_CONFIG}"
    echo "  Batch: ${BATCH_SIZE}/GPU × accum=${GRAD_ACCUM} × ${NPROC} GPUs"
    echo "  Steps: ${MAX_STEPS}"
    echo "=============================================="

    accelerate launch \
        --config_file "scripts/accelerate_configs/${ACCEL_CONFIG}.yaml" \
        --num_processes "${NPROC}" \
        examples/fineweb/pt_mdlm.py \
        --model_name_or_path "${MODEL_CONFIG}" \
        --dataset_args "HuggingFaceFW/fineweb-edu" \
        --streaming True \
        --text_field "text" \
        --max_length 2048 \
        --insert_eos True \
        --max_steps "${MAX_STEPS}" \
        --learning_rate 1e-4 \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.05 \
        --max_grad_norm 1.0 \
        --per_device_train_batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --bf16 True \
        --gradient_checkpointing "${GRAD_CKPT}" \
        --logging_steps 10 \
        --save_steps 500 \
        --save_total_limit 25 \
        --eval_strategy "no" \
        --report_to wandb \
        --run_name "${RUN_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        "$@"

    echo "Training complete! Output: ${OUTPUT_DIR}"
}

# =============================================================================
# Train BD3LM
# =============================================================================
train_bd3lm() {
    local SIZE="$1"
    shift
    get_batch_config "${SIZE}" "bd3lm"

    local BS="${BLOCK_SIZE:-16}"
    local MODEL_CONFIG="${DLLM_ROOT}/model_configs/a2d_qwen2_fineweb_${SIZE}"
    local RUN_NAME="a2d_bd3lm_fineweb_${SIZE}_bs${BS}_$(date +%Y%m%d_%H%M%S)"
    local OUTPUT_DIR="${DLLM_ROOT}/saves/fineweb/${RUN_NAME}"

    echo "=============================================="
    echo "Training A2D-BD3LM ${SIZE} on FineWeb-Edu"
    echo "  Model config: ${MODEL_CONFIG}"
    echo "  Block size:   ${BS}"
    echo "  Batch: ${BATCH_SIZE}/GPU × accum=${GRAD_ACCUM} × ${NPROC} GPUs"
    echo "  Steps: ${MAX_STEPS}"
    echo "=============================================="

    accelerate launch \
        --config_file "scripts/accelerate_configs/${ACCEL_CONFIG}.yaml" \
        --num_processes "${NPROC}" \
        examples/fineweb/pt_bd3lm.py \
        --model_name_or_path "${MODEL_CONFIG}" \
        --dataset_args "HuggingFaceFW/fineweb-edu" \
        --streaming True \
        --text_field "text" \
        --max_length 2048 \
        --insert_eos True \
        --max_steps "${MAX_STEPS}" \
        --learning_rate 1e-4 \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.05 \
        --max_grad_norm 1.0 \
        --per_device_train_batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --block_size "${BS}" \
        --attn_implementation flex_attention \
        --bf16 True \
        --gradient_checkpointing "${GRAD_CKPT}" \
        --logging_steps 10 \
        --save_steps 500 \
        --save_total_limit 25 \
        --eval_strategy "no" \
        --report_to wandb \
        --run_name "${RUN_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        "$@"

    echo "Training complete! Output: ${OUTPUT_DIR}"
}

# =============================================================================
# Main dispatch
# =============================================================================
METHOD="${1:-help}"
SIZE="${2:-400M}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true

case "${METHOD}" in
    mdlm)
        setup_env
        train_mdlm "${SIZE}" "$@"
        ;;
    bd3lm)
        setup_env
        train_bd3lm "${SIZE}" "$@"
        ;;
    all)
        setup_env
        for S in 100M 200M 400M; do
            train_mdlm "${S}" "$@"
            train_bd3lm "${S}" "$@"
        done
        ;;
    help|*)
        echo "Usage: $0 {mdlm|bd3lm|all} [SIZE] [extra_args...]"
        echo ""
        echo "  SIZE: 100M, 200M, or 400M (default: 400M)"
        echo ""
        echo "Examples:"
        echo "  bash $0 mdlm 400M"
        echo "  bash $0 bd3lm 100M"
        echo "  BLOCK_SIZE=32 bash $0 bd3lm 400M"
        echo "  MAX_STEPS=20000 bash $0 mdlm 400M"
        echo "  bash $0 all  # all sizes, both methods"
        ;;
esac
