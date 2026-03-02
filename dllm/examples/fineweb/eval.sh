#!/usr/bin/env bash
# =============================================================================
# Evaluate dllm (MDLM/BD3LM) FineWeb checkpoints on downstream benchmarks
# =============================================================================
# Uses the lm-evaluation-harness via dllm/pipelines/a2d/eval.py.
#
# Usage:
#   bash dllm/examples/fineweb/eval.sh --model_name_or_path <checkpoint_dir> [--model_type mdlm|bd3lm] [--num_gpu 1]
#
# Examples:
#   bash dllm/examples/fineweb/eval.sh --model_name_or_path dllm/saves/fineweb/a2d_mdlm_fineweb_400M_.../checkpoint-5000
#   bash dllm/examples/fineweb/eval.sh --model_name_or_path dllm/saves/fineweb/a2d_bd3lm_fineweb_400M_bs16_.../checkpoint-5000 --model_type bd3lm
# =============================================================================

set -euo pipefail

export PYTHONPATH=".:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

model_name_or_path=""
num_gpu=1
model_type="mdlm"
block_size=16

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path) model_name_or_path="$2"; shift 2 ;;
    --num_gpu) num_gpu="$2"; shift 2 ;;
    --model_type) model_type="$2"; shift 2 ;;
    --block_size) block_size="$2"; shift 2 ;;
    *) echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "${model_name_or_path}" ]]; then
    echo "Error: --model_name_or_path is required"
    exit 1
fi

if [[ "${model_type}" == "mdlm" ]]; then
    model_arg="a2d_mdlm"
    block_arg="block_size=256"
elif [[ "${model_type}" == "bd3lm" ]]; then
    model_arg="a2d_bd3lm"
    block_arg="block_size=${block_size}"
else
    echo "Error: model_type must be mdlm or bd3lm, got: ${model_type}"
    exit 1
fi

eval_script="dllm/pipelines/a2d/eval.py"
common_args="--model ${model_arg} --apply_chat_template"

echo "=== Evaluating ${model_type} checkpoint: ${model_name_or_path} ==="

# ---- Loglikelihood tasks (short generation) ----
for task in hellaswag arc_easy arc_challenge piqa winogrande openbookqa; do
    echo ">>> Task: ${task}"
    accelerate launch --num_processes "${num_gpu}" "${eval_script}" \
        --tasks "${task}" --num_fewshot 0 ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=3,steps=3,${block_arg},cfg_scale=0.0"
done

# ---- MMLU (generative, short answer) ----
echo ">>> Task: mmlu_generative_dream"
accelerate launch --num_processes "${num_gpu}" "${eval_script}" \
    --tasks mmlu_generative_dream --num_fewshot 0 ${common_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=3,steps=3,${block_arg},cfg_scale=0.0"

# ---- Generative tasks (longer generation) ----
for task in gsm8k_cot bbh; do
    echo ">>> Task: ${task}"
    fewshot=0
    [[ "${task}" == "bbh" ]] && fewshot=3
    accelerate launch --num_processes "${num_gpu}" "${eval_script}" \
        --tasks "${task}" --num_fewshot "${fewshot}" ${common_args} \
        --model_args "pretrained=${model_name_or_path},max_new_tokens=256,steps=256,${block_arg},cfg_scale=0.0"
done

echo "=== Evaluation complete ==="
