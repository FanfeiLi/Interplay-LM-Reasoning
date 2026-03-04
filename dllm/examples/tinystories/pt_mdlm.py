"""
Pre-train A2D-Qwen2 100M with Masked Diffusion (MDLM) on TinyStories.

Uses the tokenized training data from the n-gram statistics paper
(arxiv 2407.12034): gs://transformer-ngrams/TinyStories/training_data/

Data format: parquet files, each with a 'tokens' column (list of int).
Tokens are already encoded with the paper's SentencePiece 32768 tokenizer.
BOS token = 1 (do not predict BOS positions, i.e. mask them in labels).

Before running, download the SentencePiece model:
    See dllm/model_configs/a2d_qwen2_tinystories_100M/README.md

Usage (1 GPU, testing):
    accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \\
        examples/tinystories/pt_mdlm.py \\
        --training_data_dir /path/to/TinyStories/training_data \\
        --max_steps 100 --per_device_train_batch_size 4

Usage (8 GPUs, ZeRO-2):
    accelerate launch --config_file scripts/accelerate_configs/zero2.yaml \\
        examples/tinystories/pt_mdlm.py \\
        --training_data_dir /path/to/TinyStories/training_data
"""

import os
from dataclasses import dataclass, field

import accelerate
import torch
import transformers

import dllm

logger = dllm.utils.get_default_logger(__name__)

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/fast/pmayilvahanan/Interplay-LM-Reasoning")


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = os.path.join(
        PROJECT_ROOT, "dllm/model_configs/a2d_qwen2_tinystories_100M"
    )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    training_data_dir: str = field(
        metadata={"help": (
            "Path to local directory containing TinyStories training parquets "
            "(downloaded from gs://transformer-ngrams/TinyStories/training_data/). "
            "Each parquet has a 'tokens' column with pre-tokenized sequences."
        )}
    )
    max_length: int = 2048
    drop_tail: bool = True

    # Unused fields kept for interface compatibility
    dataset_args: str = ""
    streaming: bool = False
    insert_eos: bool = False
    load_preprocessed_data: bool = False
    text_field: str = "tokens"


@dataclass
class TrainingArguments(dllm.core.trainers.MDLMConfig):
    output_dir: str = os.path.join(
        PROJECT_ROOT, "dllm/saves/tinystories/a2d_mdlm_100M"
    )
    # TODO: set max_steps after downloading training data.
    # Compute as: ceil(total_tokens / seq_length) * 4_epochs / (batch_per_step)
    # where batch_per_step = per_device_train_batch_size * num_gpus * gradient_accumulation_steps
    max_steps: int = -1           # -1 means use num_train_epochs instead
    num_train_epochs: int = 4     # match n-gram paper (arxiv 2407.12034)
    # --- identical to colleague's gsm_infinity runs ---
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    bf16: bool = True
    ddp_timeout: int = 7200
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 25
    eval_strategy: str = "no"
    report_to: str = "wandb"
    run_name: str = "a2d_mdlm_tinystories_100M"
    gradient_checkpointing: bool = False


BOS_TOKEN_ID = 1  # matches paper: BOS_TOKEN = 1


def _load_tinystories(data_args: DataArguments, seq_length: int):
    """Load pre-tokenized TinyStories parquets and group into fixed-length chunks.

    Each parquet row has a 'tokens' column with a full story's token IDs
    (already encoded with the 32768 SentencePiece tokenizer from the paper).
    We concatenate all stories separated by BOS and chunk into seq_length blocks.
    BOS positions are masked in labels (set to -100) so they are not targets.
    """
    import glob
    import numpy as np
    import pandas as pd
    from datasets import Dataset

    parquet_files = sorted(glob.glob(os.path.join(data_args.training_data_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {data_args.training_data_dir}. "
            "Download from gs://transformer-ngrams/TinyStories/training_data/"
        )
    logger.info(f"Found {len(parquet_files)} parquet files in {data_args.training_data_dir}")

    # Concatenate all token sequences, inserting BOS between stories
    all_tokens = []
    for fpath in parquet_files:
        df = pd.read_parquet(fpath)
        for row_tokens in df["tokens"]:
            all_tokens.append(BOS_TOKEN_ID)
            all_tokens.extend(row_tokens)

    logger.info(f"Total tokens (including BOS separators): {len(all_tokens):,}")

    # Chunk into seq_length blocks, dropping the last partial chunk
    n_chunks = len(all_tokens) // seq_length
    all_tokens = all_tokens[: n_chunks * seq_length]
    chunks = np.array(all_tokens, dtype=np.int32).reshape(n_chunks, seq_length)
    logger.info(f"Chunked into {n_chunks:,} sequences of length {seq_length}")

    # Build input_ids and labels; mask BOS positions in labels
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
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    with dllm.utils.init_device_context_manager():
        model = transformers.AutoModel.from_config(config, torch_dtype=torch.bfloat16)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # ----- Tokenizer --------------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Optional PEFT: LoRA ----------------------------------------------------
    model = dllm.utils.load_peft(model=model, model_args=model_args)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = _load_tinystories(data_args, seq_length=data_args.max_length)

    split = dataset.train_test_split(test_size=min(5000, len(dataset) // 20), seed=42)

    # ----- Training ---------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start MDLM pre-training on TinyStories...")
    trainer = dllm.core.trainers.MDLMTrainer(
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
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
