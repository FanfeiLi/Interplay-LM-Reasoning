#!/bin/bash
# Evaluate the base model (before RL training) with pass@k metrics
# The config already has val_kwargs.n=128 and test_small (200 samples/op),
# so this just runs val_only mode on the base model to get pass@1 through pass@128.
#
# Usage: ./eval_base_model.sh [config_file]

set -ex

# Environment setup
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/fli/Interplay-LM-Reasoning"
cd "$PROJECT_ROOT"

# Parse arguments
CONFIG_FILE=${1:-"scripts/gsm_infinity_rl/configs/grpo_mixed.yaml"}

# Handle relative paths
if [[ ! "$CONFIG_FILE" = /* ]]; then
    if [[ -f "$PROJECT_ROOT/scripts/gsm_infinity_rl/configs/$CONFIG_FILE" ]]; then
        CONFIG_FILE="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs/$CONFIG_FILE"
    elif [[ -f "$PROJECT_ROOT/scripts/gsm_infinity_rl/$CONFIG_FILE" ]]; then
        CONFIG_FILE="$PROJECT_ROOT/scripts/gsm_infinity_rl/$CONFIG_FILE"
    elif [[ -f "$PROJECT_ROOT/$CONFIG_FILE" ]]; then
        CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
    fi
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Output directory for base model evaluation
EVAL_OUTPUT_DIR="results/gsm_infinity_rl/base_model_eval_pass128"
mkdir -p "$EVAL_OUTPUT_DIR"

echo "================================================================================"
echo "Evaluating Base Model (pass@1 through pass@128)"
echo "================================================================================"
echo "Config: $CONFIG_FILE"
echo "Output: $EVAL_OUTPUT_DIR"
echo "Using test_small (200 prompts/op x 128 rollouts)"
echo "================================================================================"

# Run evaluation using val_only mode
# The config already has:
#   - val_kwargs.n: 128 (generates 128 samples per prompt)
#   - test_small val files (200 samples per op level)
# This will compute pass@1, pass@2, pass@4, ..., pass@128
python3 -m verl.trainer.main_ppo \
    --config-path "$(dirname "$CONFIG_FILE")" \
    --config-name "$(basename "$CONFIG_FILE" .yaml)" \
    trainer.val_only=true \
    trainer.val_before_train=true \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${EVAL_OUTPUT_DIR}" \
    trainer.experiment_name="base_model_pass128" \
    trainer.logger='[console,local_json]'

echo "================================================================================"
echo "Base model evaluation complete!"
echo "Results saved to: ${EVAL_OUTPUT_DIR}"
echo "================================================================================"
