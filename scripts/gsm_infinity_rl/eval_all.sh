#!/bin/bash
# Evaluate all checkpoints in a results directory
# Usage: ./eval_all.sh <results_dir> [n_samples] [n_gpus]
#   results_dir: Directory containing experiment results (e.g., results/gsm_infinity_rl/grpo_mixed)
#   n_samples: Number of samples per prompt for pass@k (default: 128)
#   n_gpus: Number of GPUs (default: 6)

set -e

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
cd "${PROJECT_ROOT}"

if [ -z "$1" ]; then
    echo "Usage: ./eval_all.sh <results_dir> [n_samples] [n_gpus]"
    echo "  results_dir: Directory containing experiment results"
    echo "  n_samples: Number of samples per prompt for pass@k (default: 128)"
    echo "  n_gpus: Number of GPUs (default: 6)"
    exit 1
fi

RESULTS_DIR="$1"
N_SAMPLES="${2:-128}"
N_GPUS="${3:-6}"

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "Error: Results directory not found: ${RESULTS_DIR}"
    exit 1
fi

echo "========================================"
echo "Evaluating All Checkpoints"
echo "========================================"
echo "Results dir: ${RESULTS_DIR}"
echo "Samples per prompt: ${N_SAMPLES}"
echo "GPUs: ${N_GPUS}"
echo "========================================"

# Find all checkpoint directories
CHECKPOINTS=$(find "${RESULTS_DIR}" -type d -name "global_step_*" | sort -V)

if [ -z "${CHECKPOINTS}" ]; then
    echo "No checkpoints found in ${RESULTS_DIR}"
    exit 1
fi

echo "Found checkpoints:"
echo "${CHECKPOINTS}"
echo ""

for CHECKPOINT in ${CHECKPOINTS}; do
    echo "========================================"
    echo "Evaluating: ${CHECKPOINT}"
    echo "========================================"
    
    ./scripts/gsm_infinity_rl/eval_checkpoint.sh "${CHECKPOINT}" "${N_SAMPLES}" "${N_GPUS}"
    
    echo ""
done

echo "========================================"
echo "All evaluations complete!"
echo "========================================"

