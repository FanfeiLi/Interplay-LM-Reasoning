#!/bin/bash
# =============================================================================
# Evaluate checkpoints from pretraining runs (multi-GPU)
# PARALLELIZED across 8 GPUs (one checkpoint per GPU, batched in groups of 8)
#
# Supported run types: mdlm, bd3lm, transformer, mamba, mtp
#
# Notes:
# - Both BD3LM and MDLM evaluate ~10 evenly spaced checkpoints by default.
# - BD3LM defaults: pass@128, steps=256. block_size is auto-detected from
#   each run's training_args.bin (some runs use 32, others 128).
#   Override with BD3LM_BLOCK_SIZE=<val> if needed.
# - MDLM is much slower; defaults to pass@32, steps=128, max_new_tokens=1024.
# - Mamba and MTP use lingua's checkpoint format (numbered dirs like 0000005000).
# - Override with env vars, e.g.:
#     MDLM_N_SAMPLES=16 MDLM_STEPS=64 MDLM_NUM_CHECKPOINTS=12 \
#       ./scripts/eval_pretrain_checkpoints.sh mdlm <run_dir>
#     BD3LM_NUM_CHECKPOINTS=12 \
#       ./scripts/eval_pretrain_checkpoints.sh bd3lm <run_dir>
#     MAMBA_N_SAMPLES=64 MAMBA_NUM_CHECKPOINTS=8 \
#       ./scripts/eval_pretrain_checkpoints.sh mamba <run_dir>
# =============================================================================

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
cd "$PROJECT_ROOT"

RUN_TYPE="${1:?Error: Run type required (mdlm, bd3lm, transformer, mamba, or mtp)}"
RUN_DIR="${2:?Error: Run directory required}"

if [[ ! "$RUN_DIR" = /* ]]; then
    RUN_DIR="$PROJECT_ROOT/$RUN_DIR"
fi

RUN_NAME=$(basename "$RUN_DIR")

case "$RUN_TYPE" in
    mdlm|bd3lm)  OUTPUT_BASE="$PROJECT_ROOT/results/dllm_eval/$RUN_NAME" ;;
    transformer)  OUTPUT_BASE="$PROJECT_ROOT/results/transformer_eval/$RUN_NAME" ;;
    mamba)        OUTPUT_BASE="$PROJECT_ROOT/results/mamba_eval/$RUN_NAME" ;;
    mtp)          OUTPUT_BASE="$PROJECT_ROOT/results/mtp_eval/$RUN_NAME" ;;
    *)            echo "Error: Unknown run type '$RUN_TYPE'"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Evaluation defaults (env-overridable)
# ---------------------------------------------------------------------------
# General
TEST_DIR="${TEST_DIR:-$PROJECT_ROOT/data/composition_hf/test_small}"

# Diffusion defaults
BD3LM_N_SAMPLES="${BD3LM_N_SAMPLES:-128}"
BD3LM_STEPS="${BD3LM_STEPS:-256}"
BD3LM_BATCH_SIZE="${BD3LM_BATCH_SIZE:-64}"
BD3LM_MAX_NEW_TOKENS="${BD3LM_MAX_NEW_TOKENS:-1024}"
BD3LM_NUM_CHECKPOINTS="${BD3LM_NUM_CHECKPOINTS:-8}"

# MDLM is extremely slow at pass@128 + 256 steps. Default to a fast sweep
# configuration intended to finish an 8-GPU node run in ~12 hours for ~10 ckpts.
MDLM_N_SAMPLES="${MDLM_N_SAMPLES:-16}"          # you can set 16/32/64/128
MDLM_STEPS="${MDLM_STEPS:-256}"                 # diffusion steps
MDLM_BATCH_SIZE="${MDLM_BATCH_SIZE:-64}"       # micro-batch per sampling call
MDLM_MAX_NEW_TOKENS="${MDLM_MAX_NEW_TOKENS:-1024}"
MDLM_NUM_CHECKPOINTS="${MDLM_NUM_CHECKPOINTS:-8}"

# Transformer defaults (vLLM-backed evaluation in scripts/eval_checkpoints.py)
TRANSFORMER_SAMPLE_K="${TRANSFORMER_SAMPLE_K:-128}"
TRANSFORMER_MAX_NEW_TOKENS="${TRANSFORMER_MAX_NEW_TOKENS:-1024}"

# Mamba defaults (lingua PackedCausalMambaGenerator)
MAMBA_N_SAMPLES="${MAMBA_N_SAMPLES:-128}"
MAMBA_MAX_GEN_LEN="${MAMBA_MAX_GEN_LEN:-1024}"
MAMBA_MAX_TOKENS="${MAMBA_MAX_TOKENS:-2048}"
MAMBA_NUM_CHECKPOINTS="${MAMBA_NUM_CHECKPOINTS:-10}"

# MTP defaults (lingua PackedCausalTransformerGenerator)
MTP_N_SAMPLES="${MTP_N_SAMPLES:-128}"
MTP_MAX_GEN_LEN="${MTP_MAX_GEN_LEN:-1024}"
MTP_MAX_TOKENS="${MTP_MAX_TOKENS:-2048}"
MTP_NUM_CHECKPOINTS="${MTP_NUM_CHECKPOINTS:-10}"

# Avoid noisy/rare NumExpr thread init failures on big nodes.
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-64}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-64}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

echo "================================================================================"
echo "Run: $RUN_NAME | Type: $RUN_TYPE"
echo "Output: $OUTPUT_BASE"
if [ "$RUN_TYPE" == "mdlm" ]; then
    echo "Mode: PARALLEL on 8 GPUs | pass@${MDLM_N_SAMPLES}, steps=${MDLM_STEPS}, batch=${MDLM_BATCH_SIZE}, max_new_tokens=${MDLM_MAX_NEW_TOKENS} | ~${MDLM_NUM_CHECKPOINTS} ckpts"
elif [ "$RUN_TYPE" == "bd3lm" ]; then
    echo "Mode: PARALLEL on 8 GPUs | pass@${BD3LM_N_SAMPLES}, steps=${BD3LM_STEPS}, batch=${BD3LM_BATCH_SIZE}, max_new_tokens=${BD3LM_MAX_NEW_TOKENS} | ~${BD3LM_NUM_CHECKPOINTS} ckpts"
elif [ "$RUN_TYPE" == "mamba" ]; then
    echo "Mode: PARALLEL on 8 GPUs | pass@${MAMBA_N_SAMPLES}, max_gen_len=${MAMBA_MAX_GEN_LEN} | ~${MAMBA_NUM_CHECKPOINTS} ckpts"
elif [ "$RUN_TYPE" == "mtp" ]; then
    echo "Mode: PARALLEL on 8 GPUs | pass@${MTP_N_SAMPLES}, max_gen_len=${MTP_MAX_GEN_LEN} | ~${MTP_NUM_CHECKPOINTS} ckpts"
else
    echo "Mode: PARALLEL on 8 GPUs | transformer sample_k=${TRANSFORMER_SAMPLE_K}"
fi
echo "================================================================================"

# Sorted checkpoints (newest first)
# Lingua (mamba/mtp) uses checkpoints/<10-digit-number> directories.
# dLLM/transformer uses checkpoint-<number> directories.
if [ "$RUN_TYPE" == "mamba" ] || [ "$RUN_TYPE" == "mtp" ]; then
    CKPT_SEARCH_DIR="$RUN_DIR/checkpoints"
    if [ ! -d "$CKPT_SEARCH_DIR" ]; then
        CKPT_SEARCH_DIR="$RUN_DIR"
    fi
    CHECKPOINTS=$(find "$CKPT_SEARCH_DIR" -maxdepth 1 -type d -regex '.*/[0-9]+' | python3 -c "
import sys, os, re
paths = [l.strip() for l in sys.stdin if l.strip()]
def step(p):
    n = os.path.basename(p)
    m = re.match(r'^(\d+)$', n)
    return int(m.group(1)) if m else -1
paths = [p for p in paths if step(p) > 0]
paths.sort(key=step, reverse=True)
for p in paths: print(p)
")
else
    CHECKPOINTS=$(find "$RUN_DIR" -maxdepth 1 -type d -name "checkpoint-*" | python3 -c "
import sys, os, re
paths = [l.strip() for l in sys.stdin if l.strip()]
def step(p):
    n = os.path.basename(p)
    if 'final' in n: return 999999999
    m = re.search(r'checkpoint-(\d+)', n)
    return int(m.group(1)) if m else -1
paths.sort(key=step, reverse=True)
for p in paths: print(p)
")
fi

ALL_CHECKPOINT_LIST=($CHECKPOINTS)

# For non-transformer runs, only evaluate ~N evenly spaced checkpoints by default.
if [ "$RUN_TYPE" == "mdlm" ]; then
    DLLM_NUM_CKPTS="$MDLM_NUM_CHECKPOINTS"
elif [ "$RUN_TYPE" == "bd3lm" ]; then
    DLLM_NUM_CKPTS="$BD3LM_NUM_CHECKPOINTS"
elif [ "$RUN_TYPE" == "mamba" ]; then
    DLLM_NUM_CKPTS="$MAMBA_NUM_CHECKPOINTS"
elif [ "$RUN_TYPE" == "mtp" ]; then
    DLLM_NUM_CKPTS="$MTP_NUM_CHECKPOINTS"
else
    DLLM_NUM_CKPTS=0
fi

if [ "$DLLM_NUM_CKPTS" -gt 0 ] 2>/dev/null; then
    CHECKPOINTS=$(python3 - "$DLLM_NUM_CKPTS" "${ALL_CHECKPOINT_LIST[@]}" <<'PY'
import os
import re
import sys

target = int(sys.argv[1])
paths = list(sys.argv[2:])

def step(p: str) -> int:
    name = os.path.basename(p)
    if "final" in name:
        return 999_999_999
    m = re.search(r"checkpoint-(\d+)", name)
    return int(m.group(1)) if m else -1

paths.sort(key=step)  # oldest -> newest
n = len(paths)
if target <= 0:
    target = 10

if n <= target:
    selected = paths
else:
    if target == 1:
        idxs = [n - 1]
    else:
        idxs = [int(round(i * (n - 1) / (target - 1))) for i in range(target)]
    seen = set()
    dedup = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            dedup.append(i)
    selected = [paths[i] for i in dedup]

for p in reversed(selected):  # newest first
    print(p)
PY
)
    CHECKPOINT_LIST=($CHECKPOINTS)
else
    CHECKPOINT_LIST=("${ALL_CHECKPOINT_LIST[@]}")
fi

TOTAL=${#CHECKPOINT_LIST[@]}
echo "Found $TOTAL checkpoints."

# Activate environment and set paths
source "$PROJECT_ROOT/gsm_pretrain/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/dllm:$PROJECT_ROOT/lingua:$PYTHONPATH"

# For BD3LM, auto-detect the training block_size from the first checkpoint's
# training_args.bin. Falls back to 32 if detection fails.
if [ "$RUN_TYPE" == "bd3lm" ]; then
    FIRST_CKPT="${CHECKPOINT_LIST[0]}"
    DETECTED_BLOCK_SIZE=$(python3 - "$FIRST_CKPT" <<'PYBS'
import pickle, io, sys, zipfile
path = sys.argv[1] + "/training_args.bin"
try:
    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except:
                return type(name, (), {"__init__": lambda self, *a, **kw: self.__dict__.update(kw)})
    with open(path, "rb") as f:
        zf = zipfile.ZipFile(io.BytesIO(f.read()))
    for n in zf.namelist():
        if n.endswith("data.pkl"):
            obj = Unpickler(io.BytesIO(zf.read(n))).load()
            print(getattr(obj, "block_size", 32))
            break
    else:
        print(32)
except Exception:
    print(32)
PYBS
    )
    BD3LM_BLOCK_SIZE="${BD3LM_BLOCK_SIZE:-$DETECTED_BLOCK_SIZE}"
    echo "BD3LM block_size (from training): $BD3LM_BLOCK_SIZE"
fi

# Distribute checkpoints round-robin to 8 GPUs and launch
PIDS=()
for ((i=0; i<TOTAL; i++)); do
    GPU=$((i % 8))
    CKPT="${CHECKPOINT_LIST[$i]}"
    CKPT_BASENAME=$(basename "$CKPT")
    # Lingua checkpoints are named as 10-digit step numbers; map to checkpoint-<step>
    if [ "$RUN_TYPE" == "mamba" ] || [ "$RUN_TYPE" == "mtp" ]; then
        STEP_NUM=$(echo "$CKPT_BASENAME" | sed 's/^0*//')
        CKPT_NAME="checkpoint-${STEP_NUM:-0}"
    else
        CKPT_NAME="$CKPT_BASENAME"
    fi
    OUT_DIR="$OUTPUT_BASE/$CKPT_NAME"

    if [ -f "$OUT_DIR/metrics.jsonl" ]; then
        echo "[GPU $GPU] SKIP $CKPT_NAME (done)"
        continue
    fi

    mkdir -p "$OUT_DIR"

    if [ "$RUN_TYPE" == "transformer" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/scripts/eval_checkpoints.py" \
            --checkpoints-root "$CKPT" \
            --checkpoints-pattern "none" \
            --data-root "$TEST_DIR" \
            --id-op-threshold 10 \
            --gen-backend vllm \
            --sample-k "$TRANSFORMER_SAMPLE_K" \
            --temperature 0.7 \
            --max-new-tokens "$TRANSFORMER_MAX_NEW_TOKENS" \
            --output-dir "$OUT_DIR" \
            --skip-loss > "$OUT_DIR/eval.log" 2>&1 &
    elif [ "$RUN_TYPE" == "mamba" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python -m apps.mamba.gsm_infinity.eval_pass128 \
            --ckpt_dir "$CKPT" \
            --test_dir "$TEST_DIR" \
            --n_samples "$MAMBA_N_SAMPLES" \
            --max_gen_len "$MAMBA_MAX_GEN_LEN" \
            --max_tokens "$MAMBA_MAX_TOKENS" \
            --temperature 0.7 \
            --output_dir "$OUT_DIR" > "$OUT_DIR/eval.log" 2>&1 &
    elif [ "$RUN_TYPE" == "mtp" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python -m apps.mtp.gsm_infinity.eval_pass128 \
            --ckpt_dir "$CKPT" \
            --test_dir "$TEST_DIR" \
            --n_samples "$MTP_N_SAMPLES" \
            --max_gen_len "$MTP_MAX_GEN_LEN" \
            --max_tokens "$MTP_MAX_TOKENS" \
            --temperature 0.7 \
            --output_dir "$OUT_DIR" > "$OUT_DIR/eval.log" 2>&1 &
    elif [ "$RUN_TYPE" == "mdlm" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/dllm/examples/gsm_infinity/eval_pass128.py" \
            --model_path "$CKPT" \
            --sampler_type "$RUN_TYPE" \
            --test_dir "$TEST_DIR" \
            --n_samples "$MDLM_N_SAMPLES" \
            --batch_size "$MDLM_BATCH_SIZE" \
            --max_new_tokens "$MDLM_MAX_NEW_TOKENS" \
            --steps "$MDLM_STEPS" \
            --temperature 0.7 \
            --output_dir "$OUT_DIR" > "$OUT_DIR/eval.log" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=$GPU python "$PROJECT_ROOT/dllm/examples/gsm_infinity/eval_pass128.py" \
            --model_path "$CKPT" \
            --sampler_type "$RUN_TYPE" \
            --test_dir "$TEST_DIR" \
            --n_samples "$BD3LM_N_SAMPLES" \
            --batch_size "$BD3LM_BATCH_SIZE" \
            --max_new_tokens "$BD3LM_MAX_NEW_TOKENS" \
            --steps "$BD3LM_STEPS" \
            --block_size_bd3lm "$BD3LM_BLOCK_SIZE" \
            --temperature 0.7 \
            --output_dir "$OUT_DIR" > "$OUT_DIR/eval.log" 2>&1 &
    fi

    PID=$!
    PIDS+=($PID)
    echo "[GPU $GPU] PID $PID -> $CKPT_NAME"

    # If 8 jobs running, wait for ALL to finish before launching next batch
    if [ ${#PIDS[@]} -ge 8 ]; then
        echo "--- Waiting for batch of 8 to finish ---"
        for pid in "${PIDS[@]}"; do
            wait $pid
            EC=$?
            echo "  PID $pid exited with code $EC"
        done
        PIDS=()
    fi
done

# Wait for remaining jobs
if [ ${#PIDS[@]} -gt 0 ]; then
    echo "--- Waiting for final batch (${#PIDS[@]} jobs) ---"
    for pid in "${PIDS[@]}"; do
        wait $pid
        EC=$?
        echo "  PID $pid exited with code $EC"
    done
fi

echo "================================================================================"
echo "All evaluations complete! Aggregating..."
echo "================================================================================"

python3 << EOF
import json, os
from pathlib import Path

output_base = "$OUTPUT_BASE"
summary = {"checkpoints": {}}

for ckpt_dir in sorted(Path(output_base).glob("checkpoint-*")):
    try:
        step_str = ckpt_dir.name.replace("checkpoint-", "").replace("final", "999999")
        step = int(step_str)
    except: continue
    metrics = {}
    mf = ckpt_dir / "metrics.jsonl"
    if mf.exists():
        with open(mf) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    metrics.update(data.get("metrics", data))
                except: pass
    if metrics:
        summary["checkpoints"][step] = metrics

sf = os.path.join(output_base, "eval_summary.json")
with open(sf, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary: {sf}")
print(f"Checkpoints evaluated: {len(summary['checkpoints'])}")
EOF
