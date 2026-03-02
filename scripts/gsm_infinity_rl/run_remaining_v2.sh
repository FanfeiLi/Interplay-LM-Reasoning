#!/bin/bash
# =============================================================================
# Complete all remaining v2 work on one 8-GPU node:
#
#   PHASE 1 — Finish incomplete training (1 job)
#     1. grpo_rup_strong_mixed_v2  (resume from step 60, needs ~194)
#
#   PHASE 2 — Run all 4 GRPO+RUP v2 trainings (4 jobs)
#     2. grpo_rup_id_v2
#     3. grpo_rup_edge_v2
#     4. grpo_rup_hard_v2
#     5. grpo_rup_mixed_v2
#
#   PHASE 3 — Evaluate final checkpoints pass@128 (9 evals)
#     Merge FSDP -> HF, then val_only for:
#     6.  grpo_mixed_v2
#     7.  grpo_rup_id_v2
#     8.  grpo_rup_edge_v2
#     9.  grpo_rup_hard_v2
#     10. grpo_rup_mixed_v2
#     11. grpo_rup_strong_id_v2
#     12. grpo_rup_strong_edge_v2
#     13. grpo_rup_strong_hard_v2
#     14. grpo_rup_strong_mixed_v2
#
# Usage:
#   bash scripts/gsm_infinity_rl/run_remaining_v2.sh
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

TOTAL_START=$(date +%s)
JOB=0

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Finish grpo_rup_strong_mixed_v2 (resume from step 60)
# ─────────────────────────────────────────────────────────────────────────────
JOB=$((JOB+1))
echo ""
echo "================================================================"
echo "[$JOB/14] PHASE 1: Resume grpo_rup_strong_mixed_v2 (from step 60)"
echo "================================================================"

python3 -m verl.trainer.main_ppo \
    --config-path "$CONFIG_DIR" \
    --config-name grpo_rup_strong_mixed_v2

echo "[$JOB/14] grpo_rup_strong_mixed_v2 training done."

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Train all 4 GRPO+RUP v2
# ─────────────────────────────────────────────────────────────────────────────
RUP_CONFIGS=("grpo_rup_id_v2" "grpo_rup_edge_v2" "grpo_rup_hard_v2" "grpo_rup_mixed_v2")
RUP_LABELS=("ID (op=7-10)" "Edge (op=11-14)" "Hard (op=17-20)" "Mixed")

for ((i=0; i<${#RUP_CONFIGS[@]}; i++)); do
    JOB=$((JOB+1))
    CFG="${RUP_CONFIGS[$i]}"
    LABEL="${RUP_LABELS[$i]}"

    echo ""
    echo "================================================================"
    echo "[$JOB/14] PHASE 2: GRPO+RUP ${LABEL}, scale=4.8, 1 epoch, n=6"
    echo "================================================================"

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$CFG"

    echo "[$JOB/14] ${CFG} done."
done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: Evaluate all 9 missing final checkpoints (pass@128)
# ─────────────────────────────────────────────────────────────────────────────
EVAL_RUNS=(
    "grpo_mixed_v2"
    "grpo_rup_id_v2"
    "grpo_rup_edge_v2"
    "grpo_rup_hard_v2"
    "grpo_rup_mixed_v2"
    "grpo_rup_strong_id_v2"
    "grpo_rup_strong_edge_v2"
    "grpo_rup_strong_hard_v2"
    "grpo_rup_strong_mixed_v2"
)

# Map each run to the config to use for eval (need val_files definition)
# RUP/strong configs are supersets of the base — any will work for val_only.
# Use the matching base grpo config for plain grpo, rup config for rup runs.
EVAL_CFGS=(
    "grpo_mixed_v2"
    "grpo_rup_id_v2"
    "grpo_rup_edge_v2"
    "grpo_rup_hard_v2"
    "grpo_rup_mixed_v2"
    "grpo_rup_strong_id_v2"
    "grpo_rup_strong_edge_v2"
    "grpo_rup_strong_hard_v2"
    "grpo_rup_strong_mixed_v2"
)

for ((i=0; i<${#EVAL_RUNS[@]}; i++)); do
    JOB=$((JOB+1))
    RUN="${EVAL_RUNS[$i]}"
    CFG="${EVAL_CFGS[$i]}"
    RESULTS_DIR="results/gsm_infinity_rl/${RUN}"

    LATEST=$(cat "${RESULTS_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null)
    if [ -z "$LATEST" ]; then
        echo "[$JOB/14] ERROR: No checkpoint for $RUN, skipping."
        continue
    fi

    CKPT_DIR="${RESULTS_DIR}/global_step_${LATEST}"
    ACTOR_DIR="${CKPT_DIR}/actor"
    HF_DIR="${ACTOR_DIR}/huggingface"
    EVAL_DIR="${CKPT_DIR}/eval_pass128"

    echo ""
    echo "================================================================"
    echo "[$JOB/14] PHASE 3: Eval ${RUN} — global_step_${LATEST}"
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
        --config-name "$CFG" \
        actor_rollout_ref.model.path="${HF_DIR}" \
        trainer.val_only=true \
        trainer.val_before_train=true \
        trainer.resume_mode=disable \
        trainer.default_local_dir="${EVAL_DIR}" \
        trainer.experiment_name="${RUN}_step${LATEST}_eval" \
        trainer.logger='[console,local_json]'

    echo "[$JOB/14] ${RUN} eval done."
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
echo "All 14 jobs complete! Total wall time: ${HOURS}h ${MINS}m"
echo "================================================================"
echo ""
echo "Training results:"
echo "  grpo_rup_strong_mixed_v2: results/gsm_infinity_rl/grpo_rup_strong_mixed_v2/"
for CFG in "${RUP_CONFIGS[@]}"; do
    echo "  ${CFG}: results/gsm_infinity_rl/${CFG}/"
done
echo ""
echo "Evaluation results:"
for RUN in "${EVAL_RUNS[@]}"; do
    LATEST=$(cat "results/gsm_infinity_rl/${RUN}/latest_checkpointed_iteration.txt" 2>/dev/null)
    echo "  ${RUN}: results/gsm_infinity_rl/${RUN}/global_step_${LATEST}/eval_pass128/"
done
echo "================================================================"
