"""
Pre-train Qwen2ForCausalLM on TinyStories (standard next-token prediction).

Uses the same pre-tokenized training data as pt_mdlm.py so that DLLM and
ARLM results are directly comparable.

Data format: parquet files, each with a 'tokens' column (list of int).
Tokens are already encoded with the paper's SentencePiece 32768 tokenizer.
BOS token = 1.

Usage (1 GPU, testing):
    accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/tinystories/pt_ar.py \
        --training_data_dir /path/to/TinyStories/training_data \
        --max_steps 100 --per_device_train_batch_size 4

Usage (8 GPUs):
    accelerate launch --config_file scripts/accelerate_configs/ddp.yaml \
        examples/tinystories/pt_ar.py \
        --training_data_dir /path/to/TinyStories/training_data
"""

import glob
import json
import os
from dataclasses import dataclass, field

import accelerate
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset

import dllm

logger = dllm.utils.get_default_logger(__name__)

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/fast/fli/Interplay-LM-Reasoning")

BOS_TOKEN_ID = 1


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = os.path.join(
        PROJECT_ROOT, "model_configs/qwen2_tinystories_100M"
    )


@dataclass
class DataArguments:
    training_data_dir: str = field(
        default="",
        metadata={"help": (
            "Path to local directory containing TinyStories training parquets "
            "(downloaded from gs://transformer-ngrams/TinyStories/training_data/). "
            "Each parquet has a 'tokens' column with pre-tokenized sequences."
        )}
    )
    max_length: int = 2048


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = os.path.join(
        PROJECT_ROOT, "saves/tinystories/ar_100M"
    )
    max_steps: int = -1
    num_train_epochs: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    bf16: bool = True
    ddp_timeout: int = 7200
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 50
    eval_strategy: str = "no"
    report_to: str = "wandb"
    run_name: str = "ar_tinystories_100M"
    gradient_checkpointing: bool = False
    seed: int = 42


def _load_tinystories(data_args: DataArguments, seq_length: int):
    """Load pre-tokenized TinyStories parquets and group into fixed-length chunks.

    Identical to pt_mdlm.py data loading for fair comparison.
    Each parquet row has a 'tokens' column with a full story's token IDs.
    We concatenate all stories separated by BOS and chunk into seq_length blocks.

    For causal LM, labels = input_ids (HF Trainer shifts internally).
    BOS positions are masked in labels (set to -100) so they are not predicted.
    """
    parquet_files = sorted(glob.glob(os.path.join(data_args.training_data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {data_args.training_data_dir}. "
            "Download from gs://transformer-ngrams/TinyStories/training_data/"
        )
    logger.info(f"Found {len(parquet_files)} parquet files in {data_args.training_data_dir}")

    all_tokens = []
    for fpath in parquet_files:
        df = pd.read_parquet(fpath)
        for row_tokens in df["tokens"]:
            all_tokens.append(BOS_TOKEN_ID)
            all_tokens.extend(row_tokens)

    logger.info(f"Total tokens (including BOS separators): {len(all_tokens):,}")

    n_chunks = len(all_tokens) // seq_length
    all_tokens = all_tokens[: n_chunks * seq_length]
    chunks = np.array(all_tokens, dtype=np.int32).reshape(n_chunks, seq_length)
    logger.info(f"Chunked into {n_chunks:,} sequences of length {seq_length}")

    input_ids = chunks.tolist()
    labels = []
    for chunk in chunks:
        lbl = chunk.tolist()
        for i, tok in enumerate(lbl):
            if tok == BOS_TOKEN_ID:
                lbl[i] = -100
        labels.append(lbl)

    dataset = Dataset.from_dict({"input_ids": input_ids, "labels": labels})
    return dataset


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Data:  {data_args.training_data_dir}")
    logger.info(f"Output: {training_args.output_dir}")

    # ----- Model ------------------------------------------------------------------
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    with dllm.utils.init_device_context_manager():
        model = transformers.AutoModelForCausalLM.from_config(config)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    logger.info(f"Model dtype: {next(model.parameters()).dtype}")
    logger.info(f"Config torch_dtype: {config.torch_dtype}")

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = _load_tinystories(data_args, seq_length=data_args.max_length)

    split = dataset.train_test_split(test_size=min(5000, len(dataset) // 20), seed=42)

    # ----- Save hyperparams -------------------------------------------------------
    if accelerate.PartialState().is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
        hparams = {
            "model_name_or_path": model_args.model_name_or_path,
            "training_data_dir": data_args.training_data_dir,
            "max_length": data_args.max_length,
            "learning_rate": training_args.learning_rate,
            "weight_decay": training_args.weight_decay,
            "lr_scheduler_type": training_args.lr_scheduler_type,
            "warmup_ratio": training_args.warmup_ratio,
            "max_grad_norm": training_args.max_grad_norm,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "num_train_epochs": training_args.num_train_epochs,
            "max_steps": training_args.max_steps,
            "seed": training_args.seed,
        }
        with open(os.path.join(training_args.output_dir, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=2)

    # ----- Training ---------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()

    logger.info("Start AR pre-training on TinyStories...")

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            padding=True,
        ),
    )

    if trainer.is_world_process_zero():
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update({
                    "model_type": "ar",
                    "n_params": n_params,
                }, allow_val_change=True)
        except ImportError:
            pass

    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
