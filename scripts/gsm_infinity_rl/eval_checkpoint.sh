#!/bin/bash
# Evaluate a checkpoint on GSM-Infinity test sets with pass@k metrics
# Usage: ./eval_checkpoint.sh <checkpoint_path> [n_samples] [n_gpus]
#   checkpoint_path: Path to the model checkpoint to evaluate
#   n_samples: Number of samples per prompt for pass@k (default: 128)
#   n_gpus: Number of GPUs (default: 6)
#
# This script evaluates on three test sets:
#   - ID (op=2-10): In-distribution test
#   - Edge (op=11-14): Near-OOD test
#   - Hard (op=17-20): Far-OOD test
#
# Computes pass@k for k in {1, 2, 4, 8, 16, 32, 64, 128}

set -e

PROJECT_ROOT="/fast/fli/Interplay-LM-Reasoning"
cd "${PROJECT_ROOT}"

if [ -z "$1" ]; then
    echo "Usage: ./eval_checkpoint.sh <checkpoint_path> [n_samples] [n_gpus]"
    echo "  checkpoint_path: Path to the model checkpoint to evaluate"
    echo "  n_samples: Number of samples per prompt for pass@k (default: 128)"
    echo "  n_gpus: Number of GPUs (default: 6)"
    exit 1
fi

CHECKPOINT_PATH="$1"
N_SAMPLES="${2:-128}"
N_GPUS="${3:-6}"

# Extract experiment name from checkpoint path
EXPERIMENT_NAME=$(basename $(dirname "${CHECKPOINT_PATH}"))
OUTPUT_DIR="results/gsm_infinity_rl/evals/${EXPERIMENT_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Evaluating Checkpoint"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Samples per prompt: ${N_SAMPLES}"
echo "GPUs: ${N_GPUS}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

# Activate virtual environment if exists
if [ -f "${PROJECT_ROOT}/verl_env/bin/activate" ]; then
    source "${PROJECT_ROOT}/verl_env/bin/activate"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"

# Run evaluation with high pass@k samples
python3 -m verl.trainer.main_ppo \
    --config-path "${PROJECT_ROOT}/scripts/gsm_infinity_rl/configs" \
    --config-name "grpo_mixed" \
    actor_rollout_ref.model.path="${CHECKPOINT_PATH}" \
    actor_rollout_ref.rollout.n=${N_SAMPLES} \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.experiment_name="eval_$(basename ${CHECKPOINT_PATH})"

echo "========================================"
echo "Evaluation complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"

