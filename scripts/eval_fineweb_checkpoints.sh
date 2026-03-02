#!/bin/bash
# =============================================================================
# Evaluate FineWeb-Edu pretrained checkpoints on downstream benchmarks
# =============================================================================
# Dispatches to:
#   - lingua eval.py      for transformer / mamba checkpoints
#   - dllm A2D eval.py    for mdlm / bd3lm checkpoints
#
# Benchmarks: hellaswag, arc_easy, arc_challenge, piqa, winogrande,
#             openbookqa, mmlu, commonsense_qa, copa, social_iqa
#
# Usage:
#   bash scripts/eval_fineweb_checkpoints.sh transformer <run_dir>
#   bash scripts/eval_fineweb_checkpoints.sh mamba <run_dir>
#   bash scripts/eval_fineweb_checkpoints.sh mdlm <run_dir>
#   bash scripts/eval_fineweb_checkpoints.sh bd3lm <run_dir>
#
# Environment:
#   GPU_LIST       GPU IDs (default: 0,1,2,3,4,5,6,7)
#   NUM_CKPTS      Max checkpoints to evaluate (default: 10)
#   BLOCK_SIZE     BD3LM block size for eval (default: 16)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
DLLM_ROOT="${PROJECT_ROOT}/dllm"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"

RUN_TYPE="${1:?Error: Run type required (transformer, mamba, mtp, mdlm, bd3lm)}"
RUN_DIR="${2:?Error: Run directory required}"

if [[ ! "$RUN_DIR" = /* ]]; then
    RUN_DIR="$PROJECT_ROOT/$RUN_DIR"
fi

RUN_NAME=$(basename "$RUN_DIR")
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NGPUS="${#GPU_ARRAY[@]}"
NUM_CKPTS="${NUM_CKPTS:-10}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"

EVAL_CONFIG="${LINGUA_ROOT}/apps/main/configs_fineweb/eval_fineweb.yaml"

case "$RUN_TYPE" in
    transformer)  OUTPUT_BASE="$PROJECT_ROOT/results/fineweb_eval/transformer/$RUN_NAME" ;;
    mamba)        OUTPUT_BASE="$PROJECT_ROOT/results/fineweb_eval/mamba/$RUN_NAME" ;;
    mtp)          OUTPUT_BASE="$PROJECT_ROOT/results/fineweb_eval/mtp/$RUN_NAME" ;;
    mdlm)         OUTPUT_BASE="$PROJECT_ROOT/results/fineweb_eval/dllm/$RUN_NAME" ;;
    bd3lm)        OUTPUT_BASE="$PROJECT_ROOT/results/fineweb_eval/dllm/$RUN_NAME" ;;
    *)            echo "Error: Unknown run type '$RUN_TYPE'"; exit 1 ;;
esac

# =============================================================================
# Environment
# =============================================================================
module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9.1-cu12.x 2>/dev/null || true

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${DLLM_ROOT}:${PYTHONPATH:-}"
export HF_DATASETS_TRUST_REMOTE_CODE=True

echo "================================================================================"
echo "FineWeb Eval | Run: $RUN_NAME | Type: $RUN_TYPE | GPUs: $NGPUS"
echo "Output: $OUTPUT_BASE"
echo "================================================================================"

# =============================================================================
# Discover checkpoints
# =============================================================================
if [[ "$RUN_TYPE" == "mamba" ]] || [[ "$RUN_TYPE" == "transformer" ]] || [[ "$RUN_TYPE" == "mtp" ]]; then
    # lingua format: checkpoints/<10-digit>
    CKPT_DIR="$RUN_DIR/checkpoints"
    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "Error: No checkpoints/ directory in $RUN_DIR"
        exit 1
    fi
    CHECKPOINTS=$(find "$CKPT_DIR" -maxdepth 1 -type d -regex '.*/[0-9]+' | sort -t/ -k-1n)
else
    # dllm format: checkpoint-<number>
    CHECKPOINTS=$(find "$RUN_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -t- -k2n)
fi

TOTAL=$(echo "$CHECKPOINTS" | wc -l)
if [[ "$TOTAL" -eq 0 ]] || [[ -z "$CHECKPOINTS" ]]; then
    echo "Error: No checkpoints found in $RUN_DIR"
    exit 1
fi

# Evenly space if too many
if [[ "$TOTAL" -gt "$NUM_CKPTS" ]]; then
    STEP=$(( (TOTAL - 1) / (NUM_CKPTS - 1) ))
    SELECTED=""
    IDX=0
    while IFS= read -r ckpt; do
        if (( IDX % STEP == 0 )) || (( IDX == TOTAL - 1 )); then
            SELECTED="${SELECTED}${ckpt}"$'\n'
        fi
        IDX=$((IDX + 1))
    done <<< "$CHECKPOINTS"
    CHECKPOINTS="$SELECTED"
fi

echo "Checkpoints to evaluate:"
echo "$CHECKPOINTS" | head -20
echo ""

# =============================================================================
# Evaluate each checkpoint
# =============================================================================
GPU_IDX=0
PIDS=()

while IFS= read -r CKPT; do
    [[ -z "$CKPT" ]] && continue

    CKPT_NAME=$(basename "$CKPT")
    OUT_DIR="${OUTPUT_BASE}/${CKPT_NAME}"
    mkdir -p "$OUT_DIR"

    # Skip if already evaluated
    if [[ -f "$OUT_DIR/eval_results.json" ]] || [[ -f "$OUT_DIR/metrics.eval.jsonl" ]]; then
        echo "[Skip] $CKPT_NAME already evaluated"
        continue
    fi

    GPU="${GPU_ARRAY[$GPU_IDX]}"
    GPU_IDX=$(( (GPU_IDX + 1) % NGPUS ))

    echo "[Eval] $CKPT_NAME on GPU $GPU"

    case "$RUN_TYPE" in
        transformer)
            # lingua transformer eval (consolidate + lm-eval-harness)
            CUDA_VISIBLE_DEVICES=$GPU python -m apps.main.eval \
                config="${EVAL_CONFIG}" \
                ckpt_dir="$CKPT" \
                dump_dir="$OUT_DIR" \
                > "$OUT_DIR/eval.log" 2>&1 &
            ;;
        mamba)
            # lingua mamba eval
            CUDA_VISIBLE_DEVICES=$GPU python -m apps.mamba.eval \
                config="${EVAL_CONFIG}" \
                ckpt_dir="$CKPT" \
                dump_dir="$OUT_DIR" \
                > "$OUT_DIR/eval.log" 2>&1 &
            ;;
        mdlm)
            # MDLM supports loglikelihood via MC ELBO -- can do multiple-choice benchmarks
            cd "${DLLM_ROOT}"
            CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes 1 \
                dllm/pipelines/a2d/eval.py \
                --model a2d_mdlm \
                --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,mmlu \
                --num_fewshot 0 \
                --apply_chat_template \
                --model_args "pretrained=${CKPT},max_new_tokens=3,steps=3,block_size=256,cfg_scale=0.0,mc_num=128" \
                --output_path "$OUT_DIR" \
                > "$OUT_DIR/eval.log" 2>&1 &
            cd "${PROJECT_ROOT}"
            ;;
        bd3lm)
            # BD3LM loglikelihood via MC ELBO (same benchmarks as MDLM)
            cd "${DLLM_ROOT}"
            CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes 1 \
                dllm/pipelines/a2d/eval.py \
                --model a2d_bd3lm \
                --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,openbookqa,mmlu \
                --num_fewshot 0 \
                --apply_chat_template \
                --model_args "pretrained=${CKPT},max_new_tokens=3,steps=3,block_size=${BLOCK_SIZE},cfg_scale=0.0,mc_num=128" \
                --output_path "$OUT_DIR" \
                > "$OUT_DIR/eval.log" 2>&1 &
            cd "${PROJECT_ROOT}"
            ;;
        mtp)
            # lingua MTP eval (same harness as transformer, different model class)
            CUDA_VISIBLE_DEVICES=$GPU python -m apps.mtp.eval \
                config="${EVAL_CONFIG}" \
                ckpt_dir="$CKPT" \
                dump_dir="$OUT_DIR" \
                > "$OUT_DIR/eval.log" 2>&1 &
            ;;
    esac

    PIDS+=($!)

    # Wait when all GPUs are occupied
    if [[ "${#PIDS[@]}" -ge "$NGPUS" ]]; then
        echo "  Waiting for batch of ${#PIDS[@]} evals..."
        for pid in "${PIDS[@]}"; do
            wait "$pid" || echo "  Warning: PID $pid exited with error"
        done
        PIDS=()
    fi
done <<< "$CHECKPOINTS"

# Wait for remaining
if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "Waiting for final batch of ${#PIDS[@]} evals..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" || echo "  Warning: PID $pid exited with error"
    done
fi

echo ""
echo "================================================================================"
echo "All evaluations complete. Results in: $OUTPUT_BASE"
echo "================================================================================"
