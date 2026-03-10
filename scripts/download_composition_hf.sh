#!/bin/bash
# =============================================================================
# Download GSM-Infinity composition data directly from HuggingFace
# =============================================================================
# This downloads the pre-generated 1B token shards for ops 2-10
# Total available: ~59B tokens for ops 2-10
# =============================================================================

set -e

# Configuration
OUTPUT_DIR="${1:-/fast/fli/Interplay-LM-Reasoning/data/composition_hf}"
MIN_OP="${2:-2}"
MAX_OP="${3:-10}"
REPO_ID="Interplay-LM-Reasoning/composition"

# Use /tmp for downloads (supports flock), then move to final location
TEMP_DIR="/tmp/hf_download_$$"

echo "=============================================="
echo "Downloading GSM-Infinity Composition Data"
echo "=============================================="
echo "Repository: ${REPO_ID}"
echo "Temp directory: ${TEMP_DIR}"
echo "Final directory: ${OUTPUT_DIR}"
echo "Operation range: ${MIN_OP}-${MAX_OP}"
echo ""

# Create directories
mkdir -p "${OUTPUT_DIR}/train"
mkdir -p "${TEMP_DIR}"

# Set HF cache to temp dir to avoid Lustre flock issues
export HF_HOME="${TEMP_DIR}/.hf_cache"
export HF_HUB_CACHE="${TEMP_DIR}/.hf_cache"
export HUGGINGFACE_HUB_CACHE="${TEMP_DIR}/.hf_cache"

echo "Downloading dataset files to temp location (avoids Lustre flock issues)..."

# Download one op at a time
for op in $(seq ${MIN_OP} ${MAX_OP}); do
    echo ""
    echo "=== Downloading op${op} training data ==="
    
    # Create op directory
    mkdir -p "${OUTPUT_DIR}/train/${op}"
    
    # Download to temp, then copy
    huggingface-cli download ${REPO_ID} \
        --repo-type dataset \
        --include "train/${op}/*" \
        --local-dir "${TEMP_DIR}/data"
    
    # Move files to final location
    if [[ -d "${TEMP_DIR}/data/train/${op}" ]]; then
        cp -v "${TEMP_DIR}/data/train/${op}"/*.jsonl "${OUTPUT_DIR}/train/${op}/" 2>/dev/null || true
        echo "op${op} complete! Files copied to ${OUTPUT_DIR}/train/${op}/"
    fi
    
    # Clean temp to save space
    rm -rf "${TEMP_DIR}/data/train/${op}"
done

# Download heldout and test
echo ""
echo "=== Downloading heldout and test data ==="
mkdir -p "${OUTPUT_DIR}/heldout" "${OUTPUT_DIR}/test"

huggingface-cli download ${REPO_ID} \
    --repo-type dataset \
    --include "heldout/*" "test/*" \
    --local-dir "${TEMP_DIR}/data" 2>/dev/null || true

cp -v "${TEMP_DIR}/data/heldout"/*.jsonl "${OUTPUT_DIR}/heldout/" 2>/dev/null || true
cp -v "${TEMP_DIR}/data/test"/*.jsonl "${OUTPUT_DIR}/test/" 2>/dev/null || true

# Cleanup temp
rm -rf "${TEMP_DIR}"

echo ""
echo "=============================================="
echo "Download complete!"
echo "=============================================="

# Show what was downloaded
echo ""
echo "Downloaded data summary:"
for op in $(seq ${MIN_OP} ${MAX_OP}); do
    if [[ -d "${OUTPUT_DIR}/train/${op}" ]]; then
        count=$(ls -1 "${OUTPUT_DIR}/train/${op}"/*.jsonl 2>/dev/null | wc -l)
        echo "  op${op}: ${count} shards (~${count}B tokens)"
    fi
done

total_size=$(du -sh "${OUTPUT_DIR}" 2>/dev/null | cut -f1)
echo ""
echo "Total size: ${total_size}"
