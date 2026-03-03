#!/bin/bash
# =============================================================================
# Evaluate the skewed-pretrained base model with process-verified pass@128
#
# Uses a GRPO config as template but runs val_only=true on the base model.
# Overrides reward to use process verification.
#
# Usage:
#   bash scripts/gsm_infinity_rl/eval_base_model_v2_process.sh
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

BASE_MODEL="/fast/pmayilvahanan/Interplay-LM-Reasoning/LLaMA-Factory/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_20260227_233751"
EVAL_DIR="results/gsm_infinity_rl/base_model_eval_skewed_pass128"

if [ -f "${EVAL_DIR}/metrics.jsonl" ]; then
    echo "Base model evaluation already exists at: ${EVAL_DIR}/metrics.jsonl"
    echo "Delete it first if you want to re-evaluate."
    exit 0
fi

mkdir -p "$EVAL_DIR"

echo "================================================================"
echo "Evaluating skewed base model with process-verified pass@128"
echo "Model: ${BASE_MODEL}"
echo "Output: ${EVAL_DIR}"
echo "================================================================"

python3 -m verl.trainer.main_ppo \
    --config-path "$CONFIG_DIR" \
    --config-name "grpo_edge_v2" \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    custom_reward_function.path="verl/reward_fn.py" \
    custom_reward_function.name="compute_score_with_step_process" \
    +custom_reward_function.reward_kwargs.answer_weight=1.0 \
    +custom_reward_function.reward_kwargs.step_weight=0.0 \
    +custom_reward_function.reward_kwargs.value_tolerance=1e-6 \
    +custom_reward_function.reward_kwargs.zero_on_process_mismatch=true \
    trainer.val_only=true \
    trainer.val_before_train=true \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${EVAL_DIR}" \
    trainer.experiment_name="base_model_skewed_process_eval" \
    trainer.logger='[console,local_json]'

echo ""
echo "================================================================"
echo "Base model evaluation complete!"
echo "Results: ${EVAL_DIR}/metrics.jsonl"
echo "================================================================"
