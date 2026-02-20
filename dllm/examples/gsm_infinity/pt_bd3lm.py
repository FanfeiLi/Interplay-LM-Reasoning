"""
Pre-train A2D-Qwen2 100M with Block Diffusion (BD3LM) on GSM-Infinity composition data.

This trains the model from scratch on the preprocessed composition_hf_dllm dataset
using the BD3LMTrainer from dLLM.

Local users
------------
- 1 GPU (useful for testing):
    accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/gsm_infinity/pt_bd3lm.py \
        --max_steps 100 --per_device_train_batch_size 2 --attn_implementation sdpa

- 8 GPUs (ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml \
        examples/gsm_infinity/pt_bd3lm.py

- 8 GPUs (FSDP):
    accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/gsm_infinity/pt_bd3lm.py

Slurm users
------------
- 1 Node, 8 GPUs:
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/gsm_infinity/pt_bd3lm.py"
"""

import functools
import os
from dataclasses import dataclass, field

import accelerate
import torch
import transformers

import dllm

logger = dllm.utils.get_default_logger(__name__)

PROJECT_ROOT = "/fast/pmayilvahanan/Interplay-LM-Reasoning"


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = os.path.join(
        PROJECT_ROOT, "dllm/model_configs/a2d_qwen2_100M"
    )
    # BD3LM requires SDPA for the block-diagonal attention mask
    attn_implementation: str = "sdpa"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = os.path.join(
        PROJECT_ROOT, "data/composition_hf_dllm"
    )
    text_field: str = "text"
    max_length: int = 2048
    streaming: bool = False
    drop_tail: bool = True
    insert_eos: bool = True
    load_preprocessed_data: bool = True


@dataclass
class TrainingArguments(dllm.core.trainers.BD3LMConfig):
    output_dir: str = os.path.join(
        PROJECT_ROOT, "dllm/saves/gsm_infinity/a2d_bd3lm_100M"
    )
    # Match AR baseline hyperparameters
    max_steps: int = 10_000
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    # BD3LM uses 2x memory (x_t + x_0 concat), so halve batch size
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    # BD3LM specific
    block_size: int = 32
    # Precision
    bf16: bool = True
    # Logging
    logging_steps: int = 10
    save_steps: int = 1000
    save_total_limit: int = 10
    eval_strategy: str = "no"
    report_to: str = "wandb"
    run_name: str = "a2d_bd3lm_100M_gsm_infinity"
    # Gradient checkpointing to save memory
    gradient_checkpointing: bool = True


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Model ------------------------------------------------------------------
    # Initialize model weights from scratch using the A2D config
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    # BD3LM requires SDPA for block-diagonal attention mask
    if model_args.attn_implementation:
        config._attn_implementation = model_args.attn_implementation
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
        dataset = dllm.data.load_pt_dataset(
            data_args.dataset_args,
            streaming=data_args.streaming,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        # Always tokenize if the dataset has a text column (not yet tokenized)
        has_text_col = data_args.text_field in (
            dataset["train"].column_names
            if hasattr(dataset["train"], "column_names")
            else []
        )
        if has_text_col:
            dataset = dataset.map(
                functools.partial(
                    dllm.utils.tokenize_and_group,
                    tokenizer=tokenizer,
                    text_field=data_args.text_field,
                    seq_length=data_args.max_length,
                    insert_eos=data_args.insert_eos,
                    drop_tail=data_args.drop_tail,
                ),
                batched=True,
                remove_columns=dataset["train"].column_names,
                **({} if data_args.streaming else {"num_proc": data_args.num_proc}),
                **(
                    {}
                    if data_args.streaming
                    else {"desc": "Tokenizing and grouping dataset"}
                ),
            )
        if data_args.streaming:
            dataset = dataset.shuffle(seed=training_args.seed)

    # ----- Training ---------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start BD3LM pre-training...")
    trainer = dllm.core.trainers.BD3LMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
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

