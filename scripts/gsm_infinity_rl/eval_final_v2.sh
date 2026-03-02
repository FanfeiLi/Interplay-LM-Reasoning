#!/bin/bash
# =============================================================================
# Evaluate final checkpoint of each GRPO v2 run with pass@128
# Sequentially: merge FSDP -> HF, then run val_only eval
#
# Usage:
#   bash scripts/gsm_infinity_rl/eval_final_v2.sh
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

RUNS=("grpo_id_v2" "grpo_edge_v2" "grpo_hard_v2")
CONFIGS=("grpo_id_v2" "grpo_edge_v2" "grpo_hard_v2")

TOTAL=${#RUNS[@]}
TOTAL_START=$(date +%s)

for ((i=0; i<TOTAL; i++)); do
    RUN="${RUNS[$i]}"
    CFG="${CONFIGS[$i]}"
    RESULTS_DIR="results/gsm_infinity_rl/${RUN}"

    LATEST=$(cat "${RESULTS_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)
    if [ -z "$LATEST" ]; then
        echo "ERROR: No latest_checkpointed_iteration.txt for $RUN, skipping."
        continue
    fi

    CKPT_DIR="${RESULTS_DIR}/global_step_${LATEST}"
    ACTOR_DIR="${CKPT_DIR}/actor"
    HF_DIR="${ACTOR_DIR}/huggingface"
    EVAL_DIR="${CKPT_DIR}/eval_pass128"

    echo ""
    echo "================================================================"
    echo "[$((i+1))/${TOTAL}] ${RUN} — final checkpoint: global_step_${LATEST}"
    echo "================================================================"

    if [ -f "${EVAL_DIR}/metrics.jsonl" ]; then
        echo "SKIP (already evaluated)"
        continue
    fi

    # Step 1: Merge FSDP shards -> HuggingFace if needed
    if [ ! -f "${HF_DIR}/model.safetensors" ] && [ ! -f "${HF_DIR}/pytorch_model.bin" ]; then
        echo "Merging FSDP shards ..."
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$ACTOR_DIR" \
            --target_dir "$HF_DIR"
        echo "Merge done: ${HF_DIR}"
    else
        echo "HF weights already present, skipping merge."
    fi

    # Step 2: Run pass@128 evaluation
    mkdir -p "$EVAL_DIR"
    echo "Running pass@128 evaluation ..."

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$CFG" \
        actor_rollout_ref.model.path="${HF_DIR}" \
        trainer.val_only=true \
        trainer.val_before_train=true \
        trainer.resume_mode=disable \
        trainer.default_local_dir="${EVAL_DIR}" \
        trainer.experiment_name="${RUN}_step${LATEST}_eval" \
        trainer.logger='[console,local_json]'

    echo "[$((i+1))/${TOTAL}] ${RUN} evaluation complete."
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))

echo ""
echo "================================================================"
echo "All evaluations complete! Total wall time: ${HOURS}h ${MINS}m"
echo "================================================================"
echo "Results:"
for ((i=0; i<TOTAL; i++)); do
    RUN="${RUNS[$i]}"
    RESULTS_DIR="results/gsm_infinity_rl/${RUN}"
    LATEST=$(cat "${RESULTS_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)
    echo "  ${RUN}: ${RESULTS_DIR}/global_step_${LATEST}/eval_pass128/"
done
echo "================================================================"
