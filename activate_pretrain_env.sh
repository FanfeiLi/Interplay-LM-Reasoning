#!/bin/bash
# Quick activation script for GSM pre-training environment (8x H100)
module load cuda/12.1
module load cudnn/8.9.1-cu12.x
source /fast/pmayilvahanan/Interplay-LM-Reasoning/gsm_pretrain/bin/activate
export PYTHONPATH="/fast/pmayilvahanan/Interplay-LM-Reasoning:$PYTHONPATH"
export WANDB_PROJECT="gsm-infinity-pretrain"

# H100 optimizations
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "GSM Pre-training environment activated! (8x H100)"
echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
