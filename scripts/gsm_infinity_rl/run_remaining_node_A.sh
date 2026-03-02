#!/bin/bash
# =============================================================================
# NODE A — Evaluate 4 remaining checkpoints (pass@128)
#
#   1. grpo_rup_id_v2          (merge + eval, ~35min)
#   2. grpo_rup_edge_v2        (merge + eval, ~35min)
#   3. grpo_rup_strong_id_v2   (merge + eval, ~35min)
#   4. grpo_rup_strong_edge_v2 (merge + eval, ~35min)
#
#   Estimated total: ~2.5h
#
# Usage:
#   bash scripts/gsm_infinity_rl/run_remaining_node_A.sh
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

TOTAL_START=$(date +%s)

EVAL_RUNS=("grpo_rup_id_v2" "grpo_rup_edge_v2" "grpo_rup_strong_id_v2" "grpo_rup_strong_edge_v2")
EVAL_CFGS=("grpo_rup_id_v2" "grpo_rup_edge_v2" "grpo_rup_strong_id_v2" "grpo_rup_strong_edge_v2")
TOTAL=${#EVAL_RUNS[@]}

for ((i=0; i<TOTAL; i++)); do
    RUN="${EVAL_RUNS[$i]}"
    CFG="${EVAL_CFGS[$i]}"
    RESULTS_DIR="results/gsm_infinity_rl/${RUN}"

    LATEST=$(cat "${RESULTS_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)
    if [ -z "$LATEST" ]; then
        echo "[A-$((i+1))/$TOTAL] ERROR: No checkpoint for $RUN, skipping."
        continue
    fi

    CKPT_DIR="${RESULTS_DIR}/global_step_${LATEST}"
    ACTOR_DIR="${CKPT_DIR}/actor"
    HF_DIR="${ACTOR_DIR}/huggingface"
    EVAL_DIR="${CKPT_DIR}/eval_pass128"

    echo ""
    echo "================================================================"
    echo "[A-$((i+1))/$TOTAL] Eval: ${RUN} — global_step_${LATEST}"
    echo "================================================================"

    if [ -f "${EVAL_DIR}/metrics.jsonl" ]; then
        echo "SKIP (already evaluated)"
        continue
    fi

    if [ ! -f "${HF_DIR}/model.safetensors" ] && [ ! -f "${HF_DIR}/pytorch_model.bin" ]; then
        echo "Merging FSDP shards ..."
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$ACTOR_DIR" \
            --target_dir "$HF_DIR"
        echo "Merge done."
    else
        echo "HF weights already present, skipping merge."
    fi

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

    echo "[A-$((i+1))/$TOTAL] ${RUN} eval done."
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))

echo ""
echo "================================================================"
echo "Node A complete! Wall time: ${HOURS}h ${MINS}m"
echo "================================================================"
