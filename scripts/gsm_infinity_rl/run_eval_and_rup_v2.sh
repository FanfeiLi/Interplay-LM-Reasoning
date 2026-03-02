#!/bin/bash
# =============================================================================
# Sequential pipeline on one 8-GPU node:
#   Part A: Evaluate final GRPO v2 checkpoints (pass@128)
#     A1. grpo_id_v2   final checkpoint
#     A2. grpo_edge_v2 final checkpoint
#     A3. grpo_hard_v2 final checkpoint
#   Part B: Train GRPO + RUP v2 (4 regimes)
#     B1. grpo_rup_id_v2
#     B2. grpo_rup_edge_v2
#     B3. grpo_rup_hard_v2
#     B4. grpo_rup_mixed_v2
#
# Usage:
#   bash scripts/gsm_infinity_rl/run_eval_and_rup_v2.sh
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

TOTAL_START=$(date +%s)

# ─────────────────────────────────────────────────────────────────────────────
# Part A: Evaluate final GRPO v2 checkpoints with pass@128
# ─────────────────────────────────────────────────────────────────────────────
EVAL_RUNS=("grpo_id_v2" "grpo_edge_v2" "grpo_hard_v2")
EVAL_TOTAL=${#EVAL_RUNS[@]}

for ((i=0; i<EVAL_TOTAL; i++)); do
    RUN="${EVAL_RUNS[$i]}"
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
    echo "[A$((i+1))/${EVAL_TOTAL}] Eval ${RUN} — global_step_${LATEST}"
    echo "================================================================"

    if [ -f "${EVAL_DIR}/metrics.jsonl" ]; then
        echo "SKIP (already evaluated)"
        continue
    fi

    # Merge FSDP -> HuggingFace if needed
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
        --config-name "$RUN" \
        actor_rollout_ref.model.path="${HF_DIR}" \
        trainer.val_only=true \
        trainer.val_before_train=true \
        trainer.resume_mode=disable \
        trainer.default_local_dir="${EVAL_DIR}" \
        trainer.experiment_name="${RUN}_step${LATEST}_eval" \
        trainer.logger='[console,local_json]'

    echo "[A$((i+1))/${EVAL_TOTAL}] ${RUN} eval done."
done

echo ""
echo "================================================================"
echo "Part A complete — all GRPO v2 evaluations done."
echo "================================================================"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Part B: Train GRPO + RUP v2 (4 regimes)
# ─────────────────────────────────────────────────────────────────────────────
RUP_CONFIGS=("grpo_rup_id_v2" "grpo_rup_edge_v2" "grpo_rup_hard_v2" "grpo_rup_mixed_v2")
RUP_LABELS=("ID (op=7-10)" "Edge (op=11-14)" "Hard (op=17-20)" "Mixed (all)")
RUP_TOTAL=${#RUP_CONFIGS[@]}

for ((i=0; i<RUP_TOTAL; i++)); do
    CFG="${RUP_CONFIGS[$i]}"
    LABEL="${RUP_LABELS[$i]}"

    echo ""
    echo "================================================================"
    echo "[B$((i+1))/${RUP_TOTAL}] GRPO+RUP ${LABEL}, 1 epoch, n=6"
    echo "================================================================"

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$CFG"

    echo "[B$((i+1))/${RUP_TOTAL}] ${CFG} done."
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))

echo ""
echo "================================================================"
echo "All 7 jobs complete! Total wall time: ${HOURS}h ${MINS}m"
echo "================================================================"
echo "Part A — GRPO v2 evals:"
for RUN in "${EVAL_RUNS[@]}"; do
    LATEST=$(cat "results/gsm_infinity_rl/${RUN}/latest_checkpointed_iteration.txt" 2>/dev/null)
    echo "  ${RUN}: results/gsm_infinity_rl/${RUN}/global_step_${LATEST}/eval_pass128/"
done
echo ""
echo "Part B — GRPO+RUP v2 training:"
for CFG in "${RUP_CONFIGS[@]}"; do
    echo "  ${CFG}: results/gsm_infinity_rl/${CFG}/"
done
echo "================================================================"
