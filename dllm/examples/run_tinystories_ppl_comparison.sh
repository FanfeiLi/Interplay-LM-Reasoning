#!/bin/bash
# ============================================================================
# Two-round perplexity evaluation for MDLM vs AR TinyStories checkpoints.
#
# Both model types are evaluated on the SAME validation data
# (HF dataset at data/tinystories/training_data/validation).
#
# Round 1 (fast sweep):
#   MDLM: VLB PPL (eval_ppl_checkpoints.py) — batched, fast.
#   AR:   cross-entropy PPL (eval_ppl_ar.py) — batched, exact.
#   → Run align_checkpoints_by_ppl.py to find candidate matching pairs.
#
# Round 2 (exact refinement):
#   MDLM: DUEL exact likelihood (eval_duel_ppl_checkpoints.py) — slow,
#          run only on the matched checkpoints from Round 1.
#   AR:   already exact from Round 1, no re-run needed.
#
# Usage:
#   # Round 1: fast sweep of all checkpoints
#   bash examples/run_tinystories_ppl_comparison.sh round1
#
#   # Then find candidate pairs:
#   python examples/align_checkpoints_by_ppl.py \
#       --results_dir $OUTPUT_DIR \
#       --output_file $OUTPUT_DIR/alignment.json
#
#   # Round 2: exact DUEL on matched MDLM checkpoints only
#   # Edit ROUND2_CHECKPOINTS below with the paths from alignment output,
#   # then run:
#   bash examples/run_tinystories_ppl_comparison.sh round2
#
#   # Final alignment with exact numbers:
#   python examples/align_checkpoints_by_ppl.py \
#       --results_dir $OUTPUT_DIR --use_duel
# ============================================================================

set -euo pipefail

SAVES_DIR="/fast/fli/Interplay-LM-Reasoning/saves/tinystories"
EVAL_DATA="/fast/fli/Interplay-LM-Reasoning/data/tinystories/training_data/validation"
OUTPUT_DIR="/fast/fli/Interplay-LM-Reasoning/saves/tinystories/ppl_results"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTPUT_DIR"

# Max eval samples — set to 0 for full eval, or e.g. 500 for quick test
MAX_SAMPLES=${MAX_SAMPLES:-0}
BATCH_SIZE=${BATCH_SIZE:-16}
# DUEL unmasking steps — 0 = one token per step (most precise but slowest),
# 128 = faster with slight approximation.
DUEL_STEPS=${DUEL_STEPS:-128}
DUEL_MAX_SAMPLES=${DUEL_MAX_SAMPLES:-0}

ROUND="${1:-round1}"

MDLM_RUNS=(
    "a2d_mdlm_tinystories_100M_seed0_20260312_082851"
    "a2d_mdlm_tinystories_200M_seed0_20260313_113448"
    "a2d_mdlm_tinystories_400M_seed0_20260313_143051"
)

AR_RUNS=(
    "ar_tinystories_100M_seed0_20260316_182256"
    "ar_tinystories_200M_seed42_20260317_130702"
    "ar_tinystories_400M_seed42_20260317_151423"
)

# ============================================================================
# Round 1: Fast sweep — VLB (MDLM) + cross-entropy (AR) on all checkpoints
# ============================================================================
if [[ "$ROUND" == "round1" || "$ROUND" == "all" ]]; then
    echo "############################################################"
    echo "# ROUND 1: Fast sweep (VLB + cross-entropy)"
    echo "############################################################"

    for run in "${MDLM_RUNS[@]}"; do
        echo "========================================"
        echo "[Round 1] MDLM VLB: $run"
        echo "========================================"
        python "$SCRIPT_DIR/eval_ppl_checkpoints.py" \
            --checkpoint_dir "$SAVES_DIR/$run" \
            --eval_data_path "$EVAL_DATA" \
            --model_type mdlm \
            --scheduler_type linear \
            --batch_size "$BATCH_SIZE" \
            --max_samples "$MAX_SAMPLES" \
            --output_file "$OUTPUT_DIR/${run}_vlb_ppl.json"
    done

    for run in "${AR_RUNS[@]}"; do
        echo "========================================"
        echo "[Round 1] AR cross-entropy: $run"
        echo "========================================"
        python "$SCRIPT_DIR/eval_ppl_ar.py" \
            --checkpoint_dir "$SAVES_DIR/$run" \
            --eval_data_path "$EVAL_DATA" \
            --batch_size "$BATCH_SIZE" \
            --max_samples "$MAX_SAMPLES" \
            --output_file "$OUTPUT_DIR/${run}_ar_ppl.json"
    done

    echo ""
    echo "========================================"
    echo "Round 1 complete. Results in: $OUTPUT_DIR/"
    echo ""
    echo "Next: find candidate pairs:"
    echo "  python $SCRIPT_DIR/align_checkpoints_by_ppl.py \\"
    echo "      --results_dir $OUTPUT_DIR \\"
    echo "      --output_file $OUTPUT_DIR/alignment.json"
    echo ""
    echo "Then edit ROUND2_CHECKPOINTS in this script with the matched"
    echo "MDLM checkpoint paths and run:"
    echo "  bash $0 round2"
    echo "========================================"
fi

# ============================================================================
# Round 2: Exact DUEL on matched MDLM checkpoints only
#
# After Round 1 alignment, paste the best-matching MDLM checkpoint paths
# below (one per size), plus a few neighbors for robustness.
# Format: comma-separated paths per run.
#
# Example (edit these after Round 1):
#   ROUND2_100M="$SAVES_DIR/a2d_mdlm_tinystories_100M_.../checkpoint-800,checkpoint-900"
# ============================================================================
ROUND2_100M="${ROUND2_100M:-}"
ROUND2_200M="${ROUND2_200M:-}"
ROUND2_400M="${ROUND2_400M:-}"

if [[ "$ROUND" == "round2" || "$ROUND" == "all" ]]; then
    echo "############################################################"
    echo "# ROUND 2: Exact DUEL on matched MDLM checkpoints"
    echo "# (steps=$DUEL_STEPS, max_samples=$DUEL_MAX_SAMPLES)"
    echo "############################################################"

    for i in 0 1 2; do
        run="${MDLM_RUNS[$i]}"
        case $i in
            0) ckpts="$ROUND2_100M" ;;
            1) ckpts="$ROUND2_200M" ;;
            2) ckpts="$ROUND2_400M" ;;
        esac

        if [[ -z "$ckpts" ]]; then
            echo "[Round 2] Skipping $run — no checkpoints specified."
            echo "  Set ROUND2_${run##*_tinystories_}= or pass via env var."
            continue
        fi

        echo "========================================"
        echo "[Round 2] MDLM DUEL: $run"
        echo "  Checkpoints: $ckpts"
        echo "========================================"
        python "$SCRIPT_DIR/eval_duel_ppl_checkpoints.py" \
            --checkpoints "$ckpts" \
            --eval_data_path "$EVAL_DATA" \
            --model_type mdlm \
            --scheduler_type linear \
            --steps "$DUEL_STEPS" \
            --max_samples "$DUEL_MAX_SAMPLES" \
            --output_file "$OUTPUT_DIR/${run}_duel_ppl.json"
    done

    echo ""
    echo "========================================"
    echo "Round 2 complete."
    echo ""
    echo "Final alignment with exact DUEL numbers:"
    echo "  python $SCRIPT_DIR/align_checkpoints_by_ppl.py \\"
    echo "      --results_dir $OUTPUT_DIR --use_duel \\"
    echo "      --output_file $OUTPUT_DIR/alignment_exact.json"
    echo "========================================"
fi
