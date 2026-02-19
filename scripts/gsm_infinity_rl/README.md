# GSM-Infinity RL Finetuning

This directory contains configs and scripts for RL finetuning on GSM-Infinity composition tasks using GRPO, Dr. GSPO, and advanced variants (Clip-Cov, KL-Cov, Ent-Cov, RUP).

**All Dr. GSPO configs use the proven verl_custom settings:**
- `norm_adv_by_std_in_grpo: false` (no std normalization of advantages)
- `loss_agg_mode: seq-mean-token-sum` (token-sum aggregation)

## Quick Start

Everything runs through `run_experiment.sh` which handles: base model eval, training, and checkpoint eval.

```bash
# Run a complete experiment (training + checkpoint eval)
./scripts/gsm_infinity_rl/run_experiment.sh dr_gspo_mixed.yaml

# Any config can be passed directly:
./scripts/gsm_infinity_rl/run_experiment.sh dr_gspo_clip_cov_edge.yaml
./scripts/gsm_infinity_rl/run_experiment.sh dr_gspo_rup_hard.yaml
./scripts/gsm_infinity_rl/run_experiment.sh grpo_mixed.yaml

# Skip base model eval (if already done)
./scripts/gsm_infinity_rl/run_experiment.sh dr_gspo_mixed.yaml --skip-base-eval

# Eval only on existing checkpoints
./scripts/gsm_infinity_rl/run_experiment.sh dr_gspo_mixed.yaml --eval-only
```

## Available Configs

All configs live in `scripts/gsm_infinity_rl/configs/`. Each method has configs for all 4 datasets:

| Method | ID (op=2-10) | Edge (op=11-14) | Hard (op=17-20) | Mixed |
|--------|-------------|-----------------|-----------------|-------|
| **GRPO** | `grpo_id` | `grpo_edge` | `grpo_hard` | `grpo_mixed` |
| **GRPO + RUP** | — | — | — | `grpo_rup_mixed` |
| **Dr. GSPO** | `dr_gspo_id` | `dr_gspo_edge` | `dr_gspo_hard` | `dr_gspo_mixed` |
| **Dr. GSPO + Clip-Cov** | `dr_gspo_clip_cov_id` | `dr_gspo_clip_cov_edge` | `dr_gspo_clip_cov_hard` | `dr_gspo_clip_cov_mixed` |
| **Dr. GSPO + KL-Cov** | `dr_gspo_kl_cov_id` | `dr_gspo_kl_cov_edge` | `dr_gspo_kl_cov_hard` | `dr_gspo_kl_cov_mixed` |
| **Dr. GSPO + Ent-Cov** | `dr_gspo_ent_cov_id` | `dr_gspo_ent_cov_edge` | `dr_gspo_ent_cov_hard` | `dr_gspo_ent_cov_mixed` |
| **Dr. GSPO + RUP** | `dr_gspo_rup_id` | `dr_gspo_rup_edge` | `dr_gspo_rup_hard` | `dr_gspo_rup_mixed` |

Usage: `./scripts/gsm_infinity_rl/run_experiment.sh <config_name>.yaml`

## What the Master Script Does

`run_experiment.sh` runs a complete experiment pipeline:

1. **Base Model Evaluation** (pass@128): Evaluates the pretrained model before RL training. Results saved to `results/gsm_infinity_rl/base_model_eval_pass128/`.
2. **RL Training**: Runs training with periodic validation (every `test_freq` steps) using `rollout.n=8` for quick feedback.
3. **Checkpoint Evaluation** (pass@128): After training, evaluates ALL checkpoints with 128 samples per prompt.
4. **Results Aggregation**: Collects all metrics into `eval_summary.json`.

## Data Preparation

```bash
python scripts/download_rl_data.py --output-dir data/rl_finetune --samples-per-set 200000
```

Creates four training sets:
- `data/rl_finetune/train/id/` — In-distribution (op=2-10), 200K samples
- `data/rl_finetune/train/edge/` — Edge cases (op=11-14), 200K samples
- `data/rl_finetune/train/hard/` — Hard cases (op=17-20), 200K samples
- `data/rl_finetune/train/mixed/` — Mixed (~67K from each), 200K samples

## Directory Structure

```
scripts/gsm_infinity_rl/
├── configs/                              # All YAML configs (see table above)
│   ├── grpo_{id,edge,hard,mixed}.yaml
│   ├── grpo_rup_mixed.yaml
│   ├── dr_gspo_{id,edge,hard,mixed}.yaml
│   ├── dr_gspo_clip_cov_{id,edge,hard,mixed}.yaml
│   ├── dr_gspo_kl_cov_{id,edge,hard,mixed}.yaml
│   ├── dr_gspo_ent_cov_{id,edge,hard,mixed}.yaml
│   └── dr_gspo_rup_{id,edge,hard,mixed}.yaml
├── run_experiment.sh                     # Master script: base eval + training + checkpoint eval
├── eval_base_model.sh                    # Evaluate base model only
├── eval_checkpoint_passk.sh              # Evaluate a single checkpoint with pass@k
├── eval_all_checkpoints.sh               # Evaluate all checkpoints in a directory
└── README.md
```

## Algorithms

### GRPO (Geometric Reinforcement Policy Optimization)
- Standard PPO-style clipping with token-level importance ratio
- `clip_ratio: 0.2`, `loss_mode: vanilla`, `norm_adv_by_std_in_grpo: true`

### Dr. GSPO (Dr. Geometric Sequential Policy Optimization)
- Sequence-level importance ratio for better length normalization
- **No std normalization** of advantages (`norm_adv_by_std_in_grpo: false`)
- Token-sum loss aggregation (`loss_agg_mode: seq-mean-token-sum`)
- Smaller clip ratios: `clip_ratio_low: 0.0003`, `clip_ratio_high: 0.0004`

### Dr. GSPO + Clip-Cov (Covariance-based Token Masking)
- Masks out high-covariance tokens to prevent harmful updates
- `clip_cov_ratio: 0.0002`, `clip_ratio_high: 0.0004`

### Dr. GSPO + KL-Cov (Selective KL Penalty)
- Applies KL penalty selectively to high-covariance tokens
- `kl_cov_ratio: 0.002`, `ppo_kl_coef: 1.0`

### Dr. GSPO + Ent-Cov (Entropy-based Advantage Shaping)
- Alleviates entropy decay during training
- Alpha dampening with cosine schedule: `ent_cov_alpha: 0.15` decaying to `0.0`
- Rollout IS enabled with `rollout_is_threshold: 5.0`

### Dr. GSPO + RUP (Reward Uncertainty Predictor)
- Trains MLP on frozen reference model hidden states to predict rewards
- Adds intrinsic exploration bonus based on prediction error
- `feature_type: ref_hidden`, `hidden_dim: 8172`, `lr: 1e-3`, `scale: 1.0`

## Hyperparameters

| Parameter | GRPO | Dr. GSPO |
|-----------|------|----------|
| `adv_estimator` | grpo | grpo |
| `loss_mode` | vanilla | gspo |
| `loss_agg_mode` | seq-mean-token-mean | seq-mean-token-sum |
| `norm_adv_by_std_in_grpo` | true | false |
| `clip_ratio` | 0.2 | 0.0003 |
| `clip_ratio_low` | 0.2 | 0.0003 |
| `clip_ratio_high` | 0.2 | 0.0004 |
| `use_kl_loss` | true | false |

Common parameters: `train_batch_size: 1024`, `lr: 1e-6`, `ppo_mini_batch_size: 256`, `rollout.n: 8`, `rollout.temperature: 1.0`, `val_kwargs.temperature: 0.7`, `val_kwargs.n: 128`.

## Evaluation

```bash
# Evaluate a single checkpoint
./scripts/gsm_infinity_rl/eval_checkpoint_passk.sh \
    results/gsm_infinity_rl/dr_gspo_mixed/global_step_100 128 \
    scripts/gsm_infinity_rl/configs/dr_gspo_mixed.yaml

# Evaluate all checkpoints in a results directory
./scripts/gsm_infinity_rl/eval_all_checkpoints.sh results/gsm_infinity_rl/dr_gspo_mixed
```

## Base Model

All experiments use:
```
/fast/pmayilvahanan/Interplay-LM-Reasoning/LLaMA-Factory/saves/gsm_infinity/pt_op2-10_10B_alltemps_20260213_233838
```

## Analyzing Results

```python
import json, pandas as pd

with open("results/gsm_infinity_rl/dr_gspo_mixed/eval_summary.json") as f:
    summary = json.load(f)

rows = [{"step": int(s), **m} for s, m in summary["checkpoints"].items()]
df = pd.DataFrame(rows).sort_values("step")
print(df[["step", "val/reward/pass@1", "val/reward/pass@128"]])
```
