# GSM-Infinity Mamba-2 (SSM) Pre-training & Evaluation

Pre-train and evaluate **Mamba-2** state space models (~400M params) on the GSM-Infinity composition dataset (op 2-10, ~10B tokens), with pass@128 evaluation on the test set.

## Overview

| Property | Value |
|----------|-------|
| Architecture | Mamba-2 SSM (State Space Model) |
| Parameters | ~399M (dim=1024, 150 layers, 16 heads, state_dim=128, conv_size=4) |
| Training | Autoregressive next-token prediction via chunked SSM scan |
| Generation | Incremental state updates (no KV cache, uses SSM state cache) |
| Tokenizer | Qwen2 (same as AR transformer baseline, vocab_size=2200) |

## Files

```
lingua/apps/mamba/gsm_infinity/
├── configs/
│   └── mamba_400M_gsm.yaml   # 400M Mamba-2 training config
├── eval_pass128.py            # Pass@128 evaluation script
├── run_pretrain_400M.sh       # Training launcher
├── run_eval.sh                # Evaluation launcher
└── README.md                  # This file

lingua/apps/gsm_infinity/
└── preprocess_data.py         # Shared data converter (composition_hf -> lingua chunks)
```

## Quick Start

### Prerequisites

```bash
source /fast/pmayilvahanan/Interplay-LM-Reasoning/gsm_pretrain/bin/activate

cd /fast/pmayilvahanan/Interplay-LM-Reasoning/lingua
pip install -e .
```

### Step 1: Preprocess Data

Convert raw composition_hf data to lingua's JSONL chunk format (run once, shared with MTP):

```bash
bash lingua/apps/mamba/gsm_infinity/run_pretrain_400M.sh preprocess
```

### Step 2: Train (8x H100)

```bash
bash lingua/apps/mamba/gsm_infinity/run_pretrain_400M.sh train
```

Or with custom GPUs:

```bash
GPU_LIST=0,1,2,3 bash lingua/apps/mamba/gsm_infinity/run_pretrain_400M.sh train
```

### Step 3: Evaluate (Pass@128)

Evaluate a single checkpoint:

```bash
bash lingua/apps/mamba/gsm_infinity/run_eval.sh \
    lingua/saves/gsm_infinity/mamba_400M_gsm_.../checkpoints/0000010000 \
    results/mamba_eval/mamba_400M_gsm_.../checkpoint-10000
```

Or evaluate all checkpoints in parallel (via the shared eval script):

```bash
./scripts/eval_pretrain_checkpoints.sh mamba lingua/saves/gsm_infinity/mamba_400M_gsm_...
```

## Hyperparameters

Training hyperparameters match the AR transformer baseline for fair comparison:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-4 | |
| LR scheduler | cosine | min_ratio=0.3 |
| Warmup | 500 steps | ~5% of 10K |
| Weight decay | 0.1 | |
| Max grad norm | 1.0 | |
| Sequence length | 2048 | |
| Total steps | 10,000 | ~10B tokens |
| BF16 | True | |
| Batch size (per GPU) | 8 | |
| Gradient accumulation | 8 | |
| Effective batch (8 GPUs) | 8 * 8 * 8 = 512 seqs | ~1M tokens/step |

**Mamba-specific:**

| Parameter | Value |
|-----------|-------|
| State dimension | 128 |
| Conv size | 4 |
| SSM chunk size | 256 |
| n_groups | 1 |
| weight_tying | True |

## Model Sizing

| Architecture | dim | layers | heads | Params |
|---|---|---|---|---|
| AR Transformer (baseline) | 1024 | 26 | 16 | ~398M |
| **Mamba-2 (SSM)** | 1024 | 150 | 16 | ~399M |

Mamba requires more layers for the same parameter count because SSM blocks have fewer parameters per layer than attention+FFN blocks (no Q/K/V/O projections, just in/out projections + SSM parameters).

## Output Format

Evaluation produces `metrics.jsonl` in the same format as the AR transformer and diffusion model evaluations:

```json
{
  "timestamp": 1234567890.0,
  "log_step": 0,
  "metrics": {
    "val-core/difficulty-5B/2/reward/mean@128": 0.97,
    "val-aux/difficulty-5B/2/reward/pass@1": 0.97,
    "val-aux/difficulty-5B/2/reward/pass@128": 1.0,
    ...
  }
}
```
