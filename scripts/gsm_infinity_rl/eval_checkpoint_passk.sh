#!/bin/bash
# Evaluate a single checkpoint with pass@k metrics (k=1,2,4,8,16,32,64,128)
# Usage: ./eval_checkpoint_passk.sh <checkpoint_dir> [pass_k] [config_file]

set -ex

# Environment setup
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
cd "$PROJECT_ROOT"

# Parse arguments
CHECKPOINT_DIR=$1
PASS_K=${2:-128}  # Default to pass@128
CONFIG_FILE=${3:-"scripts/gsm_infinity_rl/configs/grpo_mixed.yaml"}

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory required"
    echo "Usage: $0 <checkpoint_dir> [pass_k] [config_file]"
    exit 1
fi

# Extract checkpoint info
CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")
EXPERIMENT_DIR=$(dirname "$CHECKPOINT_DIR")
EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")

# Output directory for evaluation results
EVAL_OUTPUT_DIR="${CHECKPOINT_DIR}/eval_pass${PASS_K}"
mkdir -p "$EVAL_OUTPUT_DIR"

echo "================================================================================"
echo "Evaluating checkpoint: $CHECKPOINT_DIR"
echo "Pass@k: $PASS_K"
echo "Config: $CONFIG_FILE"
echo "Output: $EVAL_OUTPUT_DIR"
echo "================================================================================"

# Get actor path from checkpoint
ACTOR_PATH="${CHECKPOINT_DIR}/actor"
if [ ! -d "$ACTOR_PATH" ]; then
    echo "Error: Actor checkpoint not found at $ACTOR_PATH"
    exit 1
fi

# Run evaluation with val_only mode
python3 -m verl.trainer.main_ppo \
    --config-path "$PROJECT_ROOT/$(dirname "$CONFIG_FILE")" \
    --config-name "$(basename "$CONFIG_FILE" .yaml)" \
    actor_rollout_ref.model.path="${ACTOR_PATH}" \
    actor_rollout_ref.rollout.n=${PASS_K} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    trainer.val_only=true \
    trainer.val_before_train=true \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${EVAL_OUTPUT_DIR}" \
    trainer.experiment_name="${EXPERIMENT_NAME}_${CHECKPOINT_NAME}_pass${PASS_K}" \
    trainer.logger='[console,local_json]' \
    data.val_batch_size=8

echo "================================================================================"
echo "Evaluation complete!"
echo "Results saved to: ${EVAL_OUTPUT_DIR}"
echo "================================================================================"

