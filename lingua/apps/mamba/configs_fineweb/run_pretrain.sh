#!/bin/bash
# =============================================================================
# FineWeb-Edu Pre-training — Mamba-2 (SSM) (lingua)
# =============================================================================
# Usage:
#   bash lingua/apps/mamba/configs_fineweb/run_pretrain.sh 100M
#   bash lingua/apps/mamba/configs_fineweb/run_pretrain.sh 200M
#   bash lingua/apps/mamba/configs_fineweb/run_pretrain.sh 400M
#
# Environment variables:
#   GPU_LIST       GPU IDs (default: 0,1,2,3,4,5,6,7)
#   DATA_DIR       Root data directory (default: /fast/pmayilvahanan/lm_datasets/)
#   WANDB_PROJECT  Wandb project (default: lingua-fineweb)
# =============================================================================

set -e

PROJECT_ROOT="/fast/pmayilvahanan/Interplay-LM-Reasoning"
LINGUA_ROOT="${PROJECT_ROOT}/lingua"
VENV="${PROJECT_ROOT}/gsm_pretrain/bin/activate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SIZE="${1:-400M}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_LIST}"
NPROC="${#GPU_ARRAY[@]}"
DATA_DIR="${DATA_DIR:-/fast/pmayilvahanan/lm_datasets/}"

CONFIG="${SCRIPT_DIR}/mamba_${SIZE}_fineweb.yaml"
RUN_NAME="mamba_${SIZE}_fineweb_$(date +%Y%m%d_%H%M%S)"
DUMP_DIR="${LINGUA_ROOT}/saves/fineweb/${RUN_NAME}"

export WANDB_PROJECT="${WANDB_PROJECT:-lingua-fineweb}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: Config not found: ${CONFIG}"
    echo "Available sizes: 100M, 200M, 400M"
    exit 1
fi

echo "=============================================="
echo "FineWeb-Edu Pre-training — Mamba-2 ${SIZE}"
echo "=============================================="
echo "Run name:   ${RUN_NAME}"
echo "Config:     ${CONFIG}"
echo "GPUs:       ${GPU_LIST} (${NPROC} processes)"
echo "Data dir:   ${DATA_DIR}"
echo "Dump dir:   ${DUMP_DIR}"
echo

module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9.1-cu12.x 2>/dev/null || true

source "${VENV}"
export PYTHONPATH="${PROJECT_ROOT}:${LINGUA_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29600}"

mkdir -p "${DUMP_DIR}"
cd "${LINGUA_ROOT}"

torchrun \
    --nproc_per_node="${NPROC}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m apps.mamba.train \
    config="${CONFIG}" \
    name="${RUN_NAME}" \
    dump_dir="${DUMP_DIR}" \
    data.root_dir="${DATA_DIR}"

echo "Training complete! Output: ${DUMP_DIR}"
