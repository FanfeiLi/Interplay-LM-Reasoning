# BD3LM Experiments on GSM-Infinity — Handoff Guide

**Goal**: Train BD3LM (Block Diffusion) models at 400M scale with block sizes 8, 16, 32 on
GSM-Infinity, reaching **90%+ ID pass@1** accuracy (comparable to the Transformer baseline),
and produce ID-vs-OOD plots showing how the scaling law / trade-off changes with block size.

---

## 1. Project Layout

```
/fast/pmayilvahanan/Interplay-LM-Reasoning/
├── dllm/                                 # Diffusion LLM codebase
│   ├── examples/gsm_infinity/            # Training & eval scripts (YOU ARE HERE)
│   │   ├── pt_bd3lm.py                  # BD3LM training entry point
│   │   ├── pt_mdlm.py                   # MDLM training entry point
│   │   ├── eval_pass128.py              # Evaluation script (pass@1 or pass@128)
│   │   ├── run_pretrain_400M.sh         # Main training launch script (400M)
│   │   ├── run_pretrain_200M.sh         # 200M variant
│   │   ├── run_pretrain.sh              # 100M variant
│   │   ├── run_eval.sh                  # Evaluation launch script
│   │   ├── precache_data.py             # Tokenize raw data (run once)
│   │   └── convert_config.py            # Create A2D model configs
│   ├── model_configs/
│   │   ├── a2d_qwen2_100M/             # 100M model config + tokenizer
│   │   ├── a2d_qwen2_200M/             # 200M model config + tokenizer
│   │   └── a2d_qwen2_400M/             # 400M model config + tokenizer
│   ├── saves/gsm_infinity/              # Training checkpoints (see §4)
│   └── scripts/accelerate_configs/      # zero1.yaml, zero2.yaml, zero3.yaml, ddp.yaml
├── data/
│   ├── composition_hf/test_small/       # Test set (op2-200.jsonl ... op20-200.jsonl)
│   ├── composition_hf_dllm_10B/         # Pre-tokenized 10B training data (107 GB)
│   └── composition_hf/train/            # Raw training JSONL (125 GB)
├── results/
│   ├── dllm_eval/<run_name>/            # DLLM eval outputs
│   └── transformer_eval/<run_name>/     # Transformer baseline eval outputs
├── analyze/
│   ├── process_eval_results.py          # Load & aggregate metrics
│   └── results_diffusion.ipynb          # Plotting notebook
├── LLaMA-Factory/saves/gsm_infinity/    # Transformer baseline checkpoints
└── gsm_pretrain/                        # Python virtual environment
```

---

## 2. Environment Setup

```bash
# Activate the virtual environment (always do this first)
source /fast/pmayilvahanan/Interplay-LM-Reasoning/gsm_pretrain/bin/activate

# Set Python path
export PYTHONPATH="/fast/pmayilvahanan/Interplay-LM-Reasoning:/fast/pmayilvahanan/Interplay-LM-Reasoning/dllm:${PYTHONPATH}"

# Verify
cd /fast/pmayilvahanan/Interplay-LM-Reasoning/dllm
python -c "import dllm; print('dllm OK')"
python -c "import transformers; print('transformers OK')"
```

If `dllm` is not installed:
```bash
cd /fast/pmayilvahanan/Interplay-LM-Reasoning/dllm
pip install -e .
```

---

## 3. Data

Pre-tokenized data is already cached at:
```
/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf_dllm_10B/   (107 GB)
```

This contains **10B tokens** of GSM-Infinity composition problems (op 2-10, all templates),
tokenized with max_length=2048. No preprocessing needed — the training scripts auto-detect it.

The test set for evaluation is at:
```
/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/test_small/
```
Contains 200 examples per op level (op 2 through 20), in `opN-200.jsonl` files.

**Important**: Do NOT regenerate the data. The distribution must stay the same across runs.

---

## 4. Existing Checkpoints

### Transformer baselines (for comparison)
| Run | Size | Location |
|-----|------|----------|
| `pt_op2-10_10B_alltemps_20260213_233838` | 100M | `LLaMA-Factory/saves/gsm_infinity/` |
| `pt_200M_ar_20260223_120306` | 200M | `LLaMA-Factory/saves/gsm_infinity/` |
| `pt_400M_ar_20260223_160500` | 400M | `LLaMA-Factory/saves/gsm_infinity/` |

### BD3LM runs (existing, 10K steps each)
| Run | Size | Block Size | Status |
|-----|------|-----------|--------|
| `a2d_bd3lm_100M_20260221_145059` | 100M | 32 | Done, evaluated |
| `a2d_bd3lm_200M_20260223_120716` | 200M | 32 | Done, evaluated |
| `a2d_bd3lm_200M_20260223_221022` | 200M | 128 | Done, evaluated |
| `a2d_bd3lm_400M_20260223_203350` | 400M | 32 | Done, evaluated |
| `a2d_bd3lm_400M_20260227_001819` | 400M | 8 | Done, evaluated |
| `a2d_bd3lm_400M_20260227_084524` | 400M | 16 | Done, evaluated |

### MDLM runs (existing)
| Run | Size | Status |
|-----|------|--------|
| `a2d_mdlm_100M_20260221_144133` | 100M | Done, evaluated |
| `a2d_mdlm_400M_20260227_001637` | 400M | Done, evaluated (optional / low priority) |

All checkpoints are in: `dllm/saves/gsm_infinity/<run_name>/checkpoint-{500,1000,...,10000,final}/`

Eval results are in: `results/dllm_eval/<run_name>/checkpoint-*/metrics.jsonl`

---

## 5. Task List

### Phase 1: BD3LM Training (PRIORITY)

The current 400M BD3LM runs at block_size={8,16} reach ~80% ID pass@1 at 10K steps.
The Transformer baseline reaches 90%+ at 10K steps. We need to close this gap.

**Task 1a**: Train BD3LM 400M with more steps for all three block sizes.

Try training for 20K steps first. If that's not enough, go to 30K or 50K.
Each 10K extra steps costs the same compute as the original run.

```bash
# Block size 8
BLOCK_SIZE=8 bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm --max_steps 20000

# Block size 16
BLOCK_SIZE=16 bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm --max_steps 20000

# Block size 32
BLOCK_SIZE=32 bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm --max_steps 20000
```

The `BLOCK_SIZE` env var controls the block size. The run name will automatically include
the block size (e.g. `a2d_bd3lm_400M_bs16_20260302_...`).

To train even longer (50K steps), increase `--max_steps` and optionally `--save_total_limit`:
```bash
BLOCK_SIZE=16 bash dllm/examples/gsm_infinity/run_pretrain_400M.sh bd3lm \
    --max_steps 50000 --save_steps 2500 --save_total_limit 25
```

> **Note on data**: The dataset has 10B tokens. At ~1M tokens/step, 10K steps = 1 epoch.
> Training for 20-50K steps means 2-5 epochs over the same data. This is fine — the data
> distribution does NOT change, we just see the same data multiple times.

**Task 1b**: Evaluate every 5000th checkpoint to track convergence.

See §6 below for evaluation instructions.

### Phase 2: Evaluation & Analysis

**Task 2a**: Run pass@1 evaluation on the trained checkpoints.

See §6 for the eval command. Evaluate at least checkpoints {5000, 10000, 15000, 20000}
for each block size to see the training trajectory.

**Task 2b**: Study the effect of diffusion steps at eval time.

For a single good checkpoint (e.g. the best one at block_size=16), try different
numbers of diffusion steps:

```bash
for S in 64 128 256 512 1024; do
    STEPS=$S bash dllm/examples/gsm_infinity/run_eval.sh \
        saves/gsm_infinity/<run_name>/checkpoint-final \
        bd3lm \
        results/dllm_eval/<run_name>/checkpoint-final_steps${S}
done
```

**Task 2c**: Generate the ID-vs-OOD plots.

See §7 below. The final deliverable is the `id_vs_ood_pass1.png` plot showing
all three block sizes with distinct trend lines.

### Phase 3: MDLM (OPTIONAL / LOW PRIORITY)

MDLM training seems to plateau and not improve loss even at 400M. Defer this unless
the BD3LM experiments finish early and there is spare compute.

---

## 6. Running Evaluation

The evaluation script computes pass@k for all op levels (2-20) and writes
`metrics.jsonl` to the output directory.

### Pass@1 evaluation (default — fast, ~20 min per checkpoint on 1 GPU)

```bash
# Single checkpoint
bash dllm/examples/gsm_infinity/run_eval.sh \
    saves/gsm_infinity/<run_name>/checkpoint-10000 \
    bd3lm \
    /fast/pmayilvahanan/Interplay-LM-Reasoning/results/dllm_eval/<run_name>/checkpoint-10000

# Important: the block_size for evaluation should match training block_size!
BLOCK_SIZE_BD3LM=8 bash dllm/examples/gsm_infinity/run_eval.sh \
    saves/gsm_infinity/a2d_bd3lm_400M_bs8_.../checkpoint-10000 \
    bd3lm \
    /fast/pmayilvahanan/Interplay-LM-Reasoning/results/dllm_eval/a2d_bd3lm_400M_bs8_.../checkpoint-10000
```

### Evaluate multiple checkpoints (batch loop)

```bash
RUN=a2d_bd3lm_400M_bs16_YYYYMMDD_HHMMSS
BS=16

for CKPT in 5000 10000 15000 20000; do
    BLOCK_SIZE_BD3LM=$BS bash dllm/examples/gsm_infinity/run_eval.sh \
        saves/gsm_infinity/${RUN}/checkpoint-${CKPT} \
        bd3lm \
        /fast/pmayilvahanan/Interplay-LM-Reasoning/results/dllm_eval/${RUN}/checkpoint-${CKPT}
done
```

### Pass@128 evaluation (optional — slow, ~12 hours per checkpoint)

```bash
N_SAMPLES=128 TEMPERATURE=0.7 bash dllm/examples/gsm_infinity/run_eval.sh \
    saves/gsm_infinity/<run_name>/checkpoint-final \
    bd3lm \
    /fast/pmayilvahanan/Interplay-LM-Reasoning/results/dllm_eval/<run_name>/checkpoint-final
```

### Environment variables for run_eval.sh

| Variable | Default | Description |
|----------|---------|-------------|
| `N_SAMPLES` | 1 | Samples per prompt (1 for pass@1, 128 for pass@128) |
| `STEPS` | 256 | Number of diffusion denoising steps |
| `TEMPERATURE` | 0.0 | Sampling temp (0.0 = greedy for pass@1, 0.7 for pass@128) |
| `BLOCK_SIZE_BD3LM` | 16 | BD3LM block size (match training!) |
| `BATCH_SIZE` | 16 | Micro-batch size |
| `MAX_NEW_TOKENS` | 1024 | Max generation length |

---

## 7. Generating Plots

### Register new runs in `process_eval_results.py`

After training and evaluating new runs, you need to register their block sizes in
`analyze/process_eval_results.py`. Find the `_BD3LM_BLOCK_SIZE_HINTS` dict and add
entries:

```python
_BD3LM_BLOCK_SIZE_HINTS: dict[str, int] = {
    # ... existing entries ...
    # Add new runs here:
    "a2d_bd3lm_400M_bs8_YYYYMMDD_HHMMSS": 8,
    "a2d_bd3lm_400M_bs16_YYYYMMDD_HHMMSS": 16,
    "a2d_bd3lm_400M_bs32_YYYYMMDD_HHMMSS": 32,
}
```

> **Note**: The updated training scripts automatically embed the block size in the run
> name (e.g. `a2d_bd3lm_400M_bs16_...`). But you still need to add them to the hints dict
> because the plotting code uses it to assign the family label `BD3LM (bs=16)`.

### Run the plotting notebook

```bash
cd /fast/pmayilvahanan/Interplay-LM-Reasoning/analyze
jupyter notebook results_diffusion.ipynb
```

Or run it from the command line:
```bash
cd /fast/pmayilvahanan/Interplay-LM-Reasoning/analyze
jupyter nbconvert --to notebook --execute results_diffusion.ipynb
```

The notebook will:
1. Auto-discover all runs under `results/dllm_eval/` and `results/transformer_eval/`
2. Load metrics, compute ID (op 2-10) and OOD-hard (op 17-20) averages
3. Plot ID-vs-OOD scatter with probit-scaled axes and linear fits
4. Save to `analyze/figures/id_vs_ood/id_vs_ood_pass1.png`

---

## 8. Key Hyperparameters (do NOT change unless discussed)

| Parameter | Value | Why |
|-----------|-------|-----|
| Learning rate | 1e-4 | Matches Transformer baseline |
| LR scheduler | cosine | Matches Transformer baseline |
| Warmup ratio | 0.05 | Matches Transformer baseline |
| Weight decay | 0.1 | Matches Transformer baseline |
| Max grad norm | 1.0 | Matches Transformer baseline |
| Sequence length | 2048 | Matches Transformer baseline |
| Effective batch | ~1M tokens/step | Matches Transformer baseline |
| Data | composition_hf_dllm_10B | Same 10B token dataset for all |
| BF16 | True | Standard precision |
| Attention | flex_attention | Required for BD3LM block masks |

**What you CAN change**:
- `--max_steps` (10K → 20K, 30K, 50K) — the main knob
- `BLOCK_SIZE` (8, 16, 32) — the experimental variable
- `--save_steps` — adjust for more/fewer checkpoints
- Evaluation: `STEPS`, `TEMPERATURE`, `N_SAMPLES`

**What you should NOT change**:
- Dataset, learning rate, batch size, model architecture
- Anything that affects the data distribution

---

## 9. Compute Estimates

| Training | GPUs | Time (10K steps) | Time (20K steps) |
|----------|------|-------------------|-------------------|
| BD3LM 400M bs=8 | 8x H100 | ~8 hours | ~16 hours |
| BD3LM 400M bs=16 | 8x H100 | ~8 hours | ~16 hours |
| BD3LM 400M bs=32 | 8x H100 | ~8 hours | ~16 hours |

| Evaluation | GPUs | Time per checkpoint |
|-----------|------|---------------------|
| pass@1 (N_SAMPLES=1) | 1x H100 | ~20-30 min |
| pass@128 (N_SAMPLES=128) | 1x H100 | ~10-14 hours |

---

## 10. Troubleshooting

**OOM during training**: Reduce `per_device_train_batch_size` and increase
`gradient_accumulation_steps` proportionally. Or switch to `ACCEL_CONFIG=zero3`.

**Missing `dllm` module**: `pip install -e /fast/pmayilvahanan/Interplay-LM-Reasoning/dllm`

**Tokenized data not found**: The scripts auto-detect
`/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf_dllm_10B/`.
If you need to regenerate (you shouldn't):
```bash
bash dllm/examples/gsm_infinity/run_pretrain_400M.sh precache
```

**WandB login**: `wandb login` or `export WANDB_MODE=offline` to skip.

---

## 11. Further Documentation

- **Diffusion LM training details**: `dllm/examples/gsm_infinity/README.md`
- **RL finetuning (not needed now)**: `scripts/gsm_infinity_rl/README.md`
- **Analysis code**: `analyze/process_eval_results.py` (well-documented module)
