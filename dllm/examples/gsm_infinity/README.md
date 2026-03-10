# GSM-Infinity Diffusion LM Pre-training & Evaluation

Pre-train and evaluate **A2D-MDLM** and **A2D-BD3LM** diffusion language models (~100M params, Qwen2 backbone) on the GSM-Infinity composition dataset (op 2-10, ~10B tokens), with pass@128 evaluation on the test set.

## Overview

| Variant | Architecture | Training | Sampling |
|---------|-------------|----------|----------|
| **A2D-MDLM** | Qwen2 100M with bidirectional attention | Masked diffusion (randomly mask tokens, predict originals) | Start all-mask, iteratively unmask by confidence |
| **A2D-BD3LM** | Qwen2 100M with bidirectional attention | Block diffusion (mask within blocks, attend to clean context) | Generate block-by-block with diffusion within each block |

Both variants share the same Qwen2 backbone and tokenizer from the AR baseline, with an added `<|mask|>` special token.

## Files

```
dllm/examples/gsm_infinity/
├── convert_config.py     # Convert Qwen2 100M config to A2D-Qwen2 with mask token
├── preprocess_data.py    # Convert composition_hf JSONL to HF dataset
├── pt_mdlm.py            # A2D-MDLM pre-training entry point
├── pt_bd3lm.py           # A2D-BD3LM pre-training entry point
├── eval_pass128.py       # Pass@128 evaluation script
├── run_pretrain.sh       # Training launch script
├── run_eval.sh           # Evaluation launch script
└── README.md             # This file

dllm/model_configs/a2d_qwen2_100M/
├── config.json           # A2D-Qwen2 model config (model_type: a2d-qwen2)
├── tokenizer.json        # Tokenizer with <|mask|> token
├── tokenizer_config.json
└── special_tokens_map.json
```

## Quick Start

### Prerequisites

```bash
# Activate the virtual environment
source /fast/fli/Interplay-LM-Reasoning/gsm_pretrain/bin/activate

# Install dLLM (if not already installed)
cd /fast/fli/Interplay-LM-Reasoning/dllm
pip install -e .

# Initialize lm-evaluation-harness submodule
git submodule update --init --recursive
pip install -e "lm-evaluation-harness"
```

### Step 1: Preprocess Data

Convert the raw composition_hf JSONL files into a HuggingFace Arrow dataset:

```bash
cd /fast/fli/Interplay-LM-Reasoning/dllm
bash examples/gsm_infinity/run_pretrain.sh preprocess
```

Or run directly:

```bash
python examples/gsm_infinity/preprocess_data.py \
    --data_dir /fast/fli/Interplay-LM-Reasoning/data/composition_hf/train \
    --output_dir /fast/fli/Interplay-LM-Reasoning/data/composition_hf_dllm \
    --op_min 2 --op_max 10
```

This reads all shards for op 2-10, converts each example to the format:
```
<question> {problem} {question} </question> <solution> {solution_body} </solution> <answer> {answer} </answer>
```
and saves as a HuggingFace dataset at `data/composition_hf_dllm/`.

### Step 2: Pre-train

#### A2D-MDLM (8 GPUs)

```bash
cd /fast/fli/Interplay-LM-Reasoning/dllm
bash examples/gsm_infinity/run_pretrain.sh mdlm
```

Or with custom GPU config:

```bash
GPU_LIST=0,1,2,3 ACCEL_CONFIG=zero2 bash examples/gsm_infinity/run_pretrain.sh mdlm
```

Or launch directly with accelerate:

```bash
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    --num_processes 8 \
    examples/gsm_infinity/pt_mdlm.py \
    --model_name_or_path "model_configs/a2d_qwen2_100M" \
    --dataset_args "/fast/fli/Interplay-LM-Reasoning/data/composition_hf_dllm" \
    --load_preprocessed_data True \
    --max_length 2048 \
    --max_steps 10000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --bf16 True
```

#### A2D-BD3LM (8 GPUs)

```bash
cd /fast/fli/Interplay-LM-Reasoning/dllm
bash examples/gsm_infinity/run_pretrain.sh bd3lm
```

Or launch directly:

```bash
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    --num_processes 8 \
    examples/gsm_infinity/pt_bd3lm.py \
    --model_name_or_path "model_configs/a2d_qwen2_100M" \
    --dataset_args "/fast/fli/Interplay-LM-Reasoning/data/composition_hf_dllm" \
    --load_preprocessed_data True \
    --max_length 2048 \
    --max_steps 10000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --block_size 32 \
    --attn_implementation sdpa \
    --bf16 True
```

#### Training Both (sequentially)

```bash
bash examples/gsm_infinity/run_pretrain.sh all
```

### Step 3: Evaluate (Pass@128)

#### Evaluate A2D-MDLM

```bash
bash examples/gsm_infinity/run_eval.sh \
    saves/gsm_infinity/a2d_mdlm_100M/checkpoint-final \
    mdlm \
    /fast/fli/Interplay-LM-Reasoning/results/dllm_eval/a2d_mdlm_100M
```

#### Evaluate A2D-BD3LM

```bash
bash examples/gsm_infinity/run_eval.sh \
    saves/gsm_infinity/a2d_bd3lm_100M/checkpoint-final \
    bd3lm \
    /fast/fli/Interplay-LM-Reasoning/results/dllm_eval/a2d_bd3lm_100M
```

#### Custom evaluation parameters

```bash
N_SAMPLES=128 BATCH_SIZE=8 TEMPERATURE=0.7 STEPS=256 \
    bash examples/gsm_infinity/run_eval.sh <model_path> <mdlm|bd3lm> <output_dir>
```

Or run the Python script directly:

```bash
python examples/gsm_infinity/eval_pass128.py \
    --model_path <checkpoint_path> \
    --sampler_type mdlm \
    --test_dir /fast/fli/Interplay-LM-Reasoning/data/composition_hf/test_small \
    --n_samples 128 \
    --batch_size 16 \
    --max_new_tokens 1024 \
    --steps 256 \
    --temperature 0.7 \
    --output_dir results/dllm_eval/a2d_mdlm_100M
```

## Hyperparameters

Training hyperparameters match the AR baseline:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 1e-4 | |
| LR scheduler | cosine | |
| Warmup ratio | 0.05 | |
| Weight decay | 0.1 | |
| Max grad norm | 1.0 | |
| Max length | 2048 | Sequence length |
| Total steps | ~10,000 | For ~10B tokens |
| BF16 | True | Mixed precision |

**MDLM-specific:**
| Parameter | Value |
|-----------|-------|
| Batch size (per GPU) | 16 |
| Gradient accumulation | 4 |
| Effective batch (8 GPUs) | 16 * 4 * 8 = 512 sequences |

**BD3LM-specific:**
| Parameter | Value |
|-----------|-------|
| Batch size (per GPU) | 8 (halved due to 2x memory from x_t+x_0 concat) |
| Gradient accumulation | 8 |
| Block size | 32 |
| Attn implementation | SDPA (required for block-diagonal attention) |
| Effective batch (8 GPUs) | 8 * 8 * 8 = 512 sequences |

## Output Format

Evaluation produces `metrics.jsonl` in the same format as the existing AR evaluation:

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

This format is compatible with the existing analysis scripts in `analyze/process_results.py`.


