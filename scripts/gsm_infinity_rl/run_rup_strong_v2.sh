#!/bin/bash
# =============================================================================
# Train GRPO + RUP (strong, scale=9.6) on 4 data regimes sequentially
# on one 8-GPU node.
#
#   1. grpo_rup_strong_id_v2    (op=7-10)
#   2. grpo_rup_strong_edge_v2  (op=11-14)
#   3. grpo_rup_strong_hard_v2  (op=17-20)
#   4. grpo_rup_strong_mixed_v2 (all)
#
# Usage:
#   bash scripts/gsm_infinity_rl/run_rup_strong_v2.sh
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="$PROJECT_ROOT/scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

CONFIGS=("grpo_rup_strong_id_v2" "grpo_rup_strong_edge_v2" "grpo_rup_strong_hard_v2" "grpo_rup_strong_mixed_v2")
LABELS=("ID (op=7-10)" "Edge (op=11-14)" "Hard (op=17-20)" "Mixed (all)")
TOTAL=${#CONFIGS[@]}

TOTAL_START=$(date +%s)

for ((i=0; i<TOTAL; i++)); do
    CFG="${CONFIGS[$i]}"
    LABEL="${LABELS[$i]}"

    echo ""
    echo "================================================================"
    echo "[$((i+1))/${TOTAL}] GRPO+RUP strong ${LABEL}, scale=24.0, 1 epoch, n=6"
    echo "================================================================"

    python3 -m verl.trainer.main_ppo \
        --config-path "$CONFIG_DIR" \
        --config-name "$CFG"

    echo "[$((i+1))/${TOTAL}] ${CFG} done."
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))

echo ""
echo "================================================================"
echo "All 4 RUP-strong jobs complete! Total wall time: ${HOURS}h ${MINS}m"
echo "================================================================"
echo "Results:"
for CFG in "${CONFIGS[@]}"; do
    echo "  ${CFG}: results/gsm_infinity_rl/${CFG}/"
done
echo "================================================================"
