#!/bin/bash
# =============================================================================
# Run all 5 jobs sequentially on one 8-GPU node:
#   1. Base model eval (pass@128)
#   2. GRPO ID   (op=7-10,  1 epoch, n=6)
#   3. GRPO Edge (op=11-14, 1 epoch, n=6)
#   4. GRPO Hard (op=17-20, 1 epoch, n=6)
#   5. GRPO Mixed (all,     1 epoch, n=6)
#
# All use the skewed-pretrained model:
#   pt_op2-10_10B_alltemps_skewed_20260227_233751
#
# Usage:
#   bash scripts/gsm_infinity_rl/run_all_grpo_v2.sh
# =============================================================================

set -e

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
CONFIG_DIR="scripts/gsm_infinity_rl/configs"
cd "$PROJECT_ROOT"

TOTAL_START=$(date +%s)

# ─── Job 1: Base model eval ─────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "[1/5] Base model evaluation (pass@128)"
echo "================================================================"

BASE_EVAL_DIR="results/gsm_infinity_rl/base_model_eval_skewed_pass128"
mkdir -p "$BASE_EVAL_DIR"

python3 -m verl.trainer.main_ppo \
    --config-path "$PROJECT_ROOT/$CONFIG_DIR" \
    --config-name grpo_id_v2 \
    trainer.val_only=true \
    trainer.val_before_train=true \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$BASE_EVAL_DIR" \
    trainer.experiment_name="base_model_skewed_pass128" \
    trainer.logger='[console,local_json]'

echo "[1/5] Done."
echo ""

# ─── Job 2: GRPO ID (op=7-10) ───────────────────────────────────────────────
echo "================================================================"
echo "[2/5] GRPO ID (op=7-10), 1 epoch, n=6"
echo "================================================================"

python3 -m verl.trainer.main_ppo \
    --config-path "$PROJECT_ROOT/$CONFIG_DIR" \
    --config-name grpo_id_v2

echo "[2/5] Done."
echo ""

# ─── Job 3: GRPO Edge (op=11-14) ────────────────────────────────────────────
echo "================================================================"
echo "[3/5] GRPO Edge (op=11-14), 1 epoch, n=6"
echo "================================================================"

python3 -m verl.trainer.main_ppo \
    --config-path "$PROJECT_ROOT/$CONFIG_DIR" \
    --config-name grpo_edge_v2

echo "[3/5] Done."
echo ""

# ─── Job 4: GRPO Hard (op=17-20) ────────────────────────────────────────────
echo "================================================================"
echo "[4/5] GRPO Hard (op=17-20), 1 epoch, n=6"
echo "================================================================"

python3 -m verl.trainer.main_ppo \
    --config-path "$PROJECT_ROOT/$CONFIG_DIR" \
    --config-name grpo_hard_v2

echo "[4/5] Done."
echo ""

# ─── Job 5: GRPO Mixed ──────────────────────────────────────────────────────
echo "================================================================"
echo "[5/5] GRPO Mixed (ID+Edge+Hard), 1 epoch, n=6"
echo "================================================================"

python3 -m verl.trainer.main_ppo \
    --config-path "$PROJECT_ROOT/$CONFIG_DIR" \
    --config-name grpo_mixed_v2

echo "[5/5] Done."
echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
TOTAL_END=$(date +%s)
TOTAL_DURATION=$(( TOTAL_END - TOTAL_START ))
HOURS=$(( TOTAL_DURATION / 3600 ))
MINS=$(( (TOTAL_DURATION % 3600) / 60 ))

echo "================================================================"
echo "All 5 jobs complete! Total wall time: ${HOURS}h ${MINS}m"
echo "================================================================"
echo "Results:"
echo "  Base eval:  results/gsm_infinity_rl/base_model_eval_skewed_pass128/"
echo "  GRPO ID:    results/gsm_infinity_rl/grpo_id_v2/"
echo "  GRPO Edge:  results/gsm_infinity_rl/grpo_edge_v2/"
echo "  GRPO Hard:  results/gsm_infinity_rl/grpo_hard_v2/"
echo "  GRPO Mixed: results/gsm_infinity_rl/grpo_mixed_v2/"
echo "================================================================"
