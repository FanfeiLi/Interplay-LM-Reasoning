#!/bin/bash
# Evaluate all checkpoints in a results directory with pass@128 metrics
# This script:
#   1. Finds all global_step_* checkpoints (evaluates last checkpoint first, descending order)
#   2. Merges FSDP shards to HuggingFace format (if needed)
#   3. Runs val_only evaluation with n=128 to get pass@1 through pass@128
#   4. Aggregates results into eval_summary.json
#
# Usage: ./eval_all_checkpoints.sh <results_dir> [config_file]
# Example: ./eval_all_checkpoints.sh results/gsm_infinity_rl/grpo_mixed

set -e

PROJECT_ROOT="/fast/fli/Interplay-LM-Reasoning"
cd "$PROJECT_ROOT"

RESULTS_DIR=$1
CONFIG_FILE=${2:-"scripts/gsm_infinity_rl/configs/grpo_mixed.yaml"}

if [ -z "$RESULTS_DIR" ]; then
    echo "Error: Results directory required"
    echo "Usage: $0 <results_dir> [config_file]"
    exit 1
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Handle relative config path
if [[ ! "$CONFIG_FILE" = /* ]]; then
    if [[ -f "$PROJECT_ROOT/$CONFIG_FILE" ]]; then
        CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
    fi
fi

echo "================================================================================"
echo "Evaluating all checkpoints in: $RESULTS_DIR"
echo "Config: $CONFIG_FILE"
echo "================================================================================"

# Find all global_step_* directories (sorted descending - last checkpoint first)
CHECKPOINTS=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -t_ -k3 -n -r)

if [ -z "$CHECKPOINTS" ]; then
    echo "No checkpoints found in $RESULTS_DIR"
    exit 1
fi

echo "Found checkpoints:"
for ckpt in $CHECKPOINTS; do
    echo "  - $(basename $ckpt)"
done
echo ""

# Evaluate each checkpoint
for ckpt in $CHECKPOINTS; do
    ckpt_name=$(basename "$ckpt")
    echo "========================================"
    echo "Evaluating: $ckpt_name"
    echo "========================================"

    # Check if already evaluated
    EVAL_MARKER="${ckpt}/eval_pass128/metrics.jsonl"
    if [ -f "$EVAL_MARKER" ]; then
        echo "Skipping (already evaluated): $ckpt_name"
        continue
    fi

    ACTOR_DIR="${ckpt}/actor"
    if [ ! -d "$ACTOR_DIR" ]; then
        echo "Skipping (no actor dir): $ckpt_name"
        continue
    fi

    # Step 1: Merge FSDP shards into HuggingFace format if needed
    HF_DIR="${ACTOR_DIR}/huggingface"
    if [ ! -f "${HF_DIR}/model.safetensors" ] && [ ! -f "${HF_DIR}/pytorch_model.bin" ]; then
        echo "Merging FSDP shards for $ckpt_name ..."
        mkdir -p "$HF_DIR"
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$ACTOR_DIR" \
            --target_dir "$HF_DIR" || {
            echo "Warning: Merge failed for $ckpt_name, skipping..."
            continue
        }
        echo "Merge complete: $HF_DIR"
    fi

    # Step 2: Run evaluation
    EVAL_OUTPUT_DIR="${ckpt}/eval_pass128"
    mkdir -p "$EVAL_OUTPUT_DIR"

    EXPERIMENT_NAME=$(basename "$RESULTS_DIR")

    echo "Running pass@128 evaluation for $ckpt_name ..."
    python3 -m verl.trainer.main_ppo \
        --config-path "$(dirname "$CONFIG_FILE")" \
        --config-name "$(basename "$CONFIG_FILE" .yaml)" \
        actor_rollout_ref.model.path="${HF_DIR}" \
        trainer.val_only=true \
        trainer.val_before_train=true \
        trainer.resume_mode=disable \
        trainer.default_local_dir="${EVAL_OUTPUT_DIR}" \
        trainer.experiment_name="${EXPERIMENT_NAME}_${ckpt_name}_eval" \
        trainer.logger='[console,local_json]' || {
        echo "Warning: Evaluation failed for $ckpt_name, continuing..."
        continue
    }

    echo "Evaluation complete for $ckpt_name"
    echo ""
done

echo ""
echo "================================================================================"
echo "All evaluations complete!"
echo "================================================================================"

# Aggregate results
echo ""
echo "Aggregating results..."

python3 << EOF
import json, os
from pathlib import Path
from collections import defaultdict

results_dir = "$RESULTS_DIR"
summary = {"checkpoints": {}}

for ckpt_dir in sorted(Path(results_dir).glob("global_step_*"), key=lambda p: int(p.name.split("_")[-1])):
    step = int(ckpt_dir.name.replace("global_step_", ""))
    eval_dir = ckpt_dir / "eval_pass128"
    if not eval_dir.exists():
        continue

    metrics = {}
    # Read metrics.jsonl
    for f in eval_dir.glob("*.jsonl"):
        try:
            with open(f) as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if "metrics" in data:
                        metrics.update(data["metrics"])
                    else:
                        metrics.update(data)
        except Exception:
            pass

    # Also check for json files
    for f in eval_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                metrics.update(data)
        except Exception:
            pass

    if metrics:
        summary["checkpoints"][step] = metrics

with open(os.path.join(results_dir, "eval_summary.json"), "w") as fp:
    json.dump(summary, fp, indent=2)

print(f"Summary saved to: {results_dir}/eval_summary.json")
print(f"Total checkpoints evaluated: {len(summary['checkpoints'])}")

# Print pass@k summary table
if summary["checkpoints"]:
    print("\n=== pass@k Summary ===")
    for step in sorted(summary["checkpoints"].keys()):
        m = summary["checkpoints"][step]
        passk = {k: v for k, v in m.items() if "pass@" in k}
        if passk:
            print(f"\nStep {step}:")
            for k in sorted(passk.keys()):
                print(f"  {k}: {passk[k]:.4f}")
EOF

echo ""
echo "Done!"
