#!/bin/bash
# =============================================================================
# GSM-Infinity Pre-training Run Script for DLLM Variants
# =============================================================================
# Pre-trains A2D-MDLM or A2D-BD3LM (~100M params) on GSM-Infinity data.
#
# Usage:
#   # Step 1: Preprocess data (only once)
#   bash dllm/examples/gsm_infinity/run_pretrain.sh preprocess
#
#   # Step 2: Train A2D-MDLM
#   bash dllm/examples/gsm_infinity/run_pretrain.sh mdlm
#
#   # Step 3: Train A2D-BD3LM
#   bash dllm/examples/gsm_infinity/run_pretrain.sh bd3lm
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

# GPU Configuration
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"

# Accelerate config (zero2 recommended for ~100M model)
ACCEL_CONFIG="${ACCEL_CONFIG:-zero2}"

# Wandb
export WANDB_PROJECT="${WANDB_PROJECT:-dllm-gsm-infinity}"

# =============================================================================
# Environment Setup
# =============================================================================
setup_env() {
    echo "[Setup] Activating environment..."
    source "${VENV}"
    export PYTHONPATH="${PROJECT_ROOT}:${DLLM_ROOT}:${PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
    cd "${DLLM_ROOT}"
    echo "[Setup] Python: $(which python)"
    echo "[Setup] GPUs: ${GPU_LIST} (${NPROC} processes)"
}

# =============================================================================
# Step 0: Convert model config (if not already done)
# =============================================================================
convert_config() {
    echo "=============================================="
    echo "Converting Qwen2 100M config to A2D-Qwen2"
    echo "=============================================="

    if [ -f "${DLLM_ROOT}/model_configs/a2d_qwen2_100M/config.json" ]; then
        echo "Config already exists, skipping."
        return
    fi

    python examples/gsm_infinity/convert_config.py \
        --src_dir "${PROJECT_ROOT}/model_configs/qwen2_100M" \
        --output_dir "${DLLM_ROOT}/model_configs/a2d_qwen2_100M"
}

# =============================================================================
# Step 1: Preprocess data
# =============================================================================
preprocess() {
    echo "=============================================="
    echo "Preprocessing composition_hf data for dLLM"
    echo "=============================================="

    if [ -d "${PROJECT_ROOT}/data/composition_hf_dllm/train" ]; then
        echo "Preprocessed data already exists, skipping."
        return
    fi

    python examples/gsm_infinity/preprocess_data.py \
        --data_dir "${PROJECT_ROOT}/data/composition_hf/train" \
        --output_dir "${PROJECT_ROOT}/data/composition_hf_dllm" \
        --op_min 2 --op_max 10 \
        --test_split_size 10000
}

# =============================================================================
# Step 2a: Train A2D-MDLM
# =============================================================================
train_mdlm() {
    echo "=============================================="
    echo "Training A2D-MDLM 100M on GSM-Infinity"
    echo "=============================================="
    echo "GPUs: ${NPROC}, Accel config: ${ACCEL_CONFIG}"

    RUN_NAME="a2d_mdlm_100M_$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${DLLM_ROOT}/saves/gsm_infinity/${RUN_NAME}"

    accelerate launch \
        --config_file "scripts/accelerate_configs/${ACCEL_CONFIG}.yaml" \
        --num_processes "${NPROC}" \
        examples/gsm_infinity/pt_mdlm.py \
        --model_name_or_path "${DLLM_ROOT}/model_configs/a2d_qwen2_100M" \
        --dataset_args "${PROJECT_ROOT}/data/composition_hf_dllm" \
        --load_preprocessed_data True \
        --text_field "text" \
        --max_length 2048 \
        --streaming False \
        --insert_eos True \
        --max_steps 10000 \
        --learning_rate 1e-4 \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.05 \
        --max_grad_norm 1.0 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --bf16 True \
        --gradient_checkpointing True \
        --logging_steps 10 \
        --save_steps 1000 \
        --save_total_limit 10 \
        --eval_strategy "no" \
        --report_to wandb \
        --run_name "${RUN_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        "$@"

    echo "Training complete! Output: ${OUTPUT_DIR}"
}

# =============================================================================
# Step 2b: Train A2D-BD3LM
# =============================================================================
train_bd3lm() {
    echo "=============================================="
    echo "Training A2D-BD3LM 100M on GSM-Infinity"
    echo "=============================================="
    echo "GPUs: ${NPROC}, Accel config: ${ACCEL_CONFIG}"

    RUN_NAME="a2d_bd3lm_100M_$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="${DLLM_ROOT}/saves/gsm_infinity/${RUN_NAME}"

    accelerate launch \
        --config_file "scripts/accelerate_configs/${ACCEL_CONFIG}.yaml" \
        --num_processes "${NPROC}" \
        examples/gsm_infinity/pt_bd3lm.py \
        --model_name_or_path "${DLLM_ROOT}/model_configs/a2d_qwen2_100M" \
        --dataset_args "${PROJECT_ROOT}/data/composition_hf_dllm" \
        --load_preprocessed_data True \
        --text_field "text" \
        --max_length 2048 \
        --streaming False \
        --insert_eos True \
        --max_steps 10000 \
        --learning_rate 1e-4 \
        --weight_decay 0.1 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.05 \
        --max_grad_norm 1.0 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 8 \
        --block_size 32 \
        --attn_implementation sdpa \
        --bf16 True \
        --gradient_checkpointing True \
        --logging_steps 10 \
        --save_steps 1000 \
        --save_total_limit 10 \
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
COMMAND="${1:-help}"
shift 2>/dev/null || true

case "${COMMAND}" in
    preprocess)
        setup_env
        convert_config
        preprocess
        ;;
    mdlm)
        setup_env
        convert_config
        train_mdlm "$@"
        ;;
    bd3lm)
        setup_env
        convert_config
        train_bd3lm "$@"
        ;;
    all)
        setup_env
        convert_config
        preprocess
        train_mdlm "$@"
        train_bd3lm "$@"
        ;;
    help|*)
        echo "Usage: $0 {preprocess|mdlm|bd3lm|all} [extra_args...]"
        echo ""
        echo "Commands:"
        echo "  preprocess  Preprocess composition_hf data for dLLM (run once)"
        echo "  mdlm        Train A2D-MDLM variant"
        echo "  bd3lm       Train A2D-BD3LM variant"
        echo "  all         Run preprocess + both training variants"
        echo ""
        echo "Environment variables:"
        echo "  GPU_LIST      GPU IDs (default: 0,1,2,3,4,5,6,7)"
        echo "  ACCEL_CONFIG  Accelerate config name (default: zero2)"
        echo "  WANDB_PROJECT Wandb project name (default: dllm-gsm-infinity)"
        ;;
esac

