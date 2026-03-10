#!/bin/bash
# Master script to run a complete RL experiment with automatic evaluation
# This script:
#   1. Evaluates the base model (before training) on all test sets
#   2. Runs RL training (GRPO, GSPO, etc.)
#   3. Automatically evaluates all checkpoints with pass@1-128 metrics
#   4. Aggregates results for analysis
#
# Usage: ./run_experiment.sh <config_file> [--eval-only] [--skip-eval] [--skip-base-eval]
# Example: ./run_experiment.sh configs/grpo_mixed.yaml
# Example: ./run_experiment.sh configs/grpo_mixed.yaml --eval-only

set -e

# Environment setup
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

PROJECT_ROOT="/fast/fli/Interplay-LM-Reasoning"
cd "$PROJECT_ROOT"

# Parse arguments
CONFIG_FILE=$1
EVAL_ONLY=false
SKIP_EVAL=false
SKIP_BASE_EVAL=false

shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --skip-base-eval)
            SKIP_BASE_EVAL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Config file required"
    echo "Usage: $0 <config_file> [--eval-only] [--skip-eval] [--skip-base-eval]"
    echo ""
    echo "Options:"
    echo "  --eval-only       Skip training, only run evaluation on existing checkpoints"
    echo "  --skip-eval       Skip checkpoint evaluation after training"
    echo "  --skip-base-eval  Skip base model evaluation before training"
    echo ""
    echo "Available configs:"
    ls -1 scripts/gsm_infinity_rl/configs/*.yaml 2>/dev/null | xargs -n1 basename
    exit 1
fi

# Handle relative paths
if [[ ! "$CONFIG_FILE" = /* ]]; then
    if [[ -f "scripts/gsm_infinity_rl/configs/$CONFIG_FILE" ]]; then
        CONFIG_FILE="scripts/gsm_infinity_rl/configs/$CONFIG_FILE"
    elif [[ -f "scripts/gsm_infinity_rl/$CONFIG_FILE" ]]; then
        CONFIG_FILE="scripts/gsm_infinity_rl/$CONFIG_FILE"
    fi
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Extract experiment name from config
EXPERIMENT_NAME=$(grep "experiment_name:" "$CONFIG_FILE" | head -1 | awk '{print $2}')
RESULTS_DIR=$(grep "default_local_dir:" "$CONFIG_FILE" | head -1 | awk '{print $2}')

# If RESULTS_DIR contains variable references, try to resolve them
if [[ "$RESULTS_DIR" == *'$'* ]] || [[ "$RESULTS_DIR" == *'{'* ]]; then
    RESULTS_DIR="results/gsm_infinity_rl/${EXPERIMENT_NAME}"
fi

echo "================================================================================"
echo "GSM-Infinity RL Experiment"
echo "================================================================================"
echo "Config: $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Results: $RESULTS_DIR"
echo "Eval only: $EVAL_ONLY"
echo "Skip eval: $SKIP_EVAL"
echo "Skip base eval: $SKIP_BASE_EVAL"
echo "================================================================================"
echo ""

START_TIME=$(date +%s)

# Step 0: Evaluate base model (unless skipped or eval-only on existing checkpoints)
if [ "$SKIP_BASE_EVAL" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo "=========================================="
    echo "Step 0: Evaluating Base Model (pass@128)"
    echo "=========================================="
    
    # Check if base model eval already exists
    BASE_EVAL_DIR="results/gsm_infinity_rl/base_model_eval_pass128"
    if [ -d "$BASE_EVAL_DIR" ] && [ -f "$BASE_EVAL_DIR/metrics.jsonl" ]; then
        echo "Base model evaluation already exists at: $BASE_EVAL_DIR"
        echo "Skipping base model evaluation..."
    else
        bash scripts/gsm_infinity_rl/eval_base_model.sh "$CONFIG_FILE"
    fi
    echo ""
fi

# Step 1: Training (unless eval-only)
if [ "$EVAL_ONLY" = false ]; then
    echo "=========================================="
    echo "Step 1: Training"
    echo "=========================================="
    
    python3 -m verl.trainer.main_ppo \
        --config-path "$PROJECT_ROOT/$(dirname "$CONFIG_FILE")" \
        --config-name "$(basename "$CONFIG_FILE" .yaml)"
    
    echo ""
    echo "Training complete!"
    echo ""
fi

# Step 2: Evaluation (unless skip-eval)
if [ "$SKIP_EVAL" = false ]; then
    echo "=========================================="
    echo "Step 2: Evaluating all checkpoints (pass@128)"
    echo "=========================================="
    
    # Wait a bit for any pending writes
    sleep 5
    
    if [ -d "$RESULTS_DIR" ]; then
        bash scripts/gsm_infinity_rl/eval_all_checkpoints.sh "$RESULTS_DIR" "$CONFIG_FILE"
    else
        echo "Warning: Results directory not found: $RESULTS_DIR"
        echo "Skipping evaluation"
    fi
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "================================================================================"
echo "Experiment Complete!"
echo "================================================================================"
echo "Config: $CONFIG_FILE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Results: $RESULTS_DIR"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results files:"
echo "  - Base model eval: results/gsm_infinity_rl/base_model_eval_pass128/"
echo "  - Training logs: ${RESULTS_DIR}/logs/"
echo "  - Checkpoints: ${RESULTS_DIR}/global_step_*/"
echo "  - Evaluation summary: ${RESULTS_DIR}/eval_summary.json"
echo "================================================================================"

