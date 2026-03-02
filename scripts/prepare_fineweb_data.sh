#!/usr/bin/env bash
# Download and prepare FineWeb-Edu 10BT for both lingua (preshuffled JSONL)
# and dllm (HuggingFace streaming -- no preparation needed).
#
# Usage:
#   bash scripts/prepare_fineweb_data.sh [DATA_DIR]
#
# DATA_DIR defaults to /fast/pmayilvahanan/lm_datasets
# The script produces: DATA_DIR/fineweb_edu_10bt_shuffled/
#   ├── fineweb_edu_10bt.chunk.00.jsonl .. chunk.31.jsonl
#   └── fineweb_edu_10bt.val.jsonl

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${1:-/fast/pmayilvahanan/lm_datasets}"

echo "=== FineWeb-Edu 10BT Data Preparation ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Data dir:     ${DATA_DIR}"
echo

# ---- Step 1: lingua format (preshuffled JSONL) ----
echo "--- Downloading + shuffling FineWeb-Edu 10BT for lingua ---"
echo "This downloads ~10BT of data from HuggingFace, converts parquet->jsonl,"
echo "shuffles with terashuf, and splits into 32 chunks."
echo

cd "${PROJECT_ROOT}/lingua"
python setup/download_prepare_hf_data.py fineweb_edu_10bt 8 \
    --data_dir "${DATA_DIR}" \
    --nchunks 32

echo
echo "=== lingua data ready at: ${DATA_DIR}/fineweb_edu_10bt_shuffled/ ==="
echo

# ---- Step 2: dllm streaming ----
echo "--- dllm uses HuggingFace streaming, no local download needed ---"
echo "dllm training scripts use: --dataset_args 'HuggingFaceFW/fineweb-edu'"
echo "with --streaming True. Data is streamed on-the-fly."
echo
echo "If you prefer local data for dllm, you can point it to the same"
echo "lingua data using --dataset_args with a local path."
echo
echo "=== Done ==="
