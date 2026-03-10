#!/bin/bash
# =============================================================================
# GSM-Infinity Pre-training Environment Setup Script
# =============================================================================
# This script sets up a Python virtual environment with all necessary packages
# for pre-training small language models on GSM-Infinity data.
# Hardware: Optimized for 8x H100 GPUs
# =============================================================================

set -e  # Exit on error

# Configuration
ENV_NAME="gsm_pretrain"
PROJECT_ROOT="/fast/fli/Interplay-LM-Reasoning"
PYTHON_VERSION="3.10"

echo "=============================================="
echo "GSM-Infinity Pre-training Environment Setup"
echo "Hardware: 8x H100 GPUs"
echo "=============================================="

# 1. Load required modules
echo ""
echo "[1/7] Loading CUDA and cuDNN modules..."
module load cuda/12.1
module load cudnn/8.9.1-cu12.x

# Verify CUDA is loaded
echo "CUDA version:"
nvcc --version 2>/dev/null || echo "nvcc not found in PATH, but module loaded"

# 2. Create virtual environment
echo ""
echo "[2/7] Creating Python virtual environment..."
cd "$PROJECT_ROOT"

if [ -d "${ENV_NAME}" ]; then
    echo "Environment '${ENV_NAME}' already exists. Removing and recreating..."
    rm -rf "${ENV_NAME}"
fi

python3 -m venv "${ENV_NAME}"
source "${ENV_NAME}/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch with CUDA support (optimized for H100)
echo ""
echo "[3/7] Installing PyTorch with CUDA 12.1 support (H100 optimized)..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# 4. Install LLaMA-Factory and dependencies
echo ""
echo "[4/7] Installing LLaMA-Factory and dependencies..."
cd "$PROJECT_ROOT/LLaMA-Factory"
pip install -e ".[torch,deepspeed,metrics]"

# 5. Install flash-attention (optimized for H100)
echo ""
echo "[5/7] Installing Flash Attention 2 (H100 optimized)..."
pip install flash-attn --no-build-isolation

# 6. Install additional dependencies
echo ""
echo "[6/7] Installing additional packages..."
pip install \
    wandb \
    huggingface_hub \
    termcolor \
    tqdm \
    omegaconf \
    hydra-core \
    ninja  # For faster compilation

# 7. Install gsm_infinite package (for data generation if needed)
echo ""
echo "[7/7] Installing gsm_infinite package..."
cd "$PROJECT_ROOT/gsm_infinite"
pip install -e .

# Create activation script
echo ""
echo "Creating activation helper script..."
cat > "$PROJECT_ROOT/activate_pretrain_env.sh" << 'EOF'
#!/bin/bash
# Quick activation script for GSM pre-training environment (8x H100)
module load cuda/12.1
module load cudnn/8.9.1-cu12.x
source /fast/fli/Interplay-LM-Reasoning/gsm_pretrain/bin/activate
export PYTHONPATH="/fast/fli/Interplay-LM-Reasoning:$PYTHONPATH"
export WANDB_PROJECT="gsm-infinity-pretrain"

# H100 optimizations
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "GSM Pre-training environment activated! (8x H100)"
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
EOF
chmod +x "$PROJECT_ROOT/activate_pretrain_env.sh"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

echo ""
echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  source $PROJECT_ROOT/activate_pretrain_env.sh"
echo ""
echo "Or manually:"
echo "  module load cuda/12.1"
echo "  module load cudnn/8.9.1-cu12.x"
echo "  source $PROJECT_ROOT/${ENV_NAME}/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Prepare data: python scripts/download_gsm_infinity_data.py"
echo "  2. Run training: bash scripts/run_pretrain_gsm_infinity.sh"
echo "  3. Or submit to SLURM: sbatch scripts/submit_pretrain.slurm"
echo ""

