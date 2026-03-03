#!/bin/bash
# =============================================================================
# Node B: GRPO mixed v2 + GRPO+RUP v2 (id, edge, hard)
# Train with outcome reward, eval with process-verified reward
#
# Usage: bash scripts/gsm_infinity_rl/run_v2_node_B.sh
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

TOTAL_START=$(date +%s)

run_experiment() {
    local CFG="$1"
    local RUN="$2"
    local IDX="$3"
    local TOTAL="$4"

    echo ""
    echo "================================================================"
    echo "[${IDX}/${TOTAL}] Training: ${RUN}"
    echo "================================================================"

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$CFG"

    echo ""
    echo "[${IDX}/${TOTAL}] Training complete. Starting eval..."

    bash scripts/gsm_infinity_rl/eval_final_v2_process.sh "$RUN"
}

run_experiment "grpo_mixed_v2"    "grpo_mixed_v2"    "1" "4"
run_experiment "grpo_rup_id_v2"   "grpo_rup_id_v2"   "2" "4"
run_experiment "grpo_rup_edge_v2" "grpo_rup_edge_v2" "3" "4"
run_experiment "grpo_rup_hard_v2" "grpo_rup_hard_v2" "4" "4"

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))

echo ""
echo "================================================================"
echo "Node B complete! Total wall time: ${HOURS}h ${MINS}m"
echo "================================================================"
echo "Runs completed:"
echo "  - grpo_mixed_v2"
echo "  - grpo_rup_id_v2"
echo "  - grpo_rup_edge_v2"
echo "  - grpo_rup_hard_v2"
echo "================================================================"
