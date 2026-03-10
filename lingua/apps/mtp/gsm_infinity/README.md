# GSM-Infinity MTP (Multi-Token Prediction) Pre-training & Evaluation

Pre-train and evaluate **Multi-Token Prediction** transformer models (~406M params) on the GSM-Infinity composition dataset (op 2-10, ~10B tokens), with pass@128 evaluation on the test set.

## Overview

| Property | Value |
|----------|-------|
| Architecture | Transformer with 3 extra prediction heads (predicts t+1, t+2, t+3) |
| Parameters | ~406M (dim=1024, 35 layers, 16 heads, 4 KV heads, 3 future heads) |
| Training | Multi-token prediction -- each position predicts the next 4 tokens |
| Generation | Standard autoregressive (uses head 0 only) |
| Tokenizer | Qwen2 (same as AR transformer baseline, vocab_size=2200) |

MTP trains the model to predict multiple future tokens at each position, which has been shown to improve representation learning and sample efficiency. During generation, only the first head (next-token) is used, so inference cost is identical to standard transformers.

## Files

```
lingua/apps/mtp/gsm_infinity/
├── configs/
│   └── mtp_400M_gsm.yaml     # 400M MTP training config
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
source /fast/fli/Interplay-LM-Reasoning/gsm_pretrain/bin/activate

cd /fast/fli/Interplay-LM-Reasoning/lingua
pip install -e .
```

### Step 1: Preprocess Data

Convert raw composition_hf data to lingua's JSONL chunk format (run once, shared with Mamba):

```bash
bash lingua/apps/mtp/gsm_infinity/run_pretrain_400M.sh preprocess
```

### Step 2: Train (8x H100)

```bash
bash lingua/apps/mtp/gsm_infinity/run_pretrain_400M.sh train
```

Or with custom GPUs:

```bash
GPU_LIST=0,1,2,3 bash lingua/apps/mtp/gsm_infinity/run_pretrain_400M.sh train
```

### Step 3: Evaluate (Pass@128)

Evaluate a single checkpoint:

```bash
bash lingua/apps/mtp/gsm_infinity/run_eval.sh \
    lingua/saves/gsm_infinity/mtp_400M_gsm_.../checkpoints/0000010000 \
    results/mtp_eval/mtp_400M_gsm_.../checkpoint-10000
```

Or evaluate all checkpoints in parallel (via the shared eval script):

```bash
./scripts/eval_pretrain_checkpoints.sh mtp lingua/saves/gsm_infinity/mtp_400M_gsm_...
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

**MTP-specific:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| n_future_head | 3 | Predict 3 future tokens |
| n_views | 4 | Must equal n_future_head + 1 |
| n_kv_heads | 4 | Grouped query attention |
| attn_impl | sdpa | Scaled dot product attention |

## Model Sizing

| Architecture | dim | layers | heads | Params |
|---|---|---|---|---|
| AR Transformer (baseline) | 1024 | 26 | 16 | ~398M |
| **MTP Transformer** | 1024 | 35 | 16 | ~406M |

MTP adds 3 extra prediction heads (each is a dim -> vocab_size linear layer). With the small GSM-Infinity vocabulary (2200 tokens), these heads add only ~7M params. The extra layers (35 vs 26) account for most of the difference.

## Data Format

MTP requires `n_views = n_future_head + 1 = 4` shifted copies of each sequence. This is handled automatically by lingua's data loader when `data.n_views: 4` is set in the config. Each training batch has shape `(batch_size, seq_len, 4)` where view 0 is the input and views 1-3 are the targets for each head.

## Output Format

Evaluation produces `metrics.jsonl` in the same format as all other model types:

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
