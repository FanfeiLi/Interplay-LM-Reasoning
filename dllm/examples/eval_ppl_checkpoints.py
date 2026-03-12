"""
Evaluate the VLB perplexity of a diffusion LM across multiple checkpoints.

The reported metric is the variational upper bound on perplexity:
    PPL_VLB = exp( E_{t,x_t}[ w(t) * NLL(x_0 | x_t) ] )
where w(t) = -α'(t)/(1-α(t)) is the scheduler weight from the training loss.
This is the standard likelihood-based metric for masked diffusion LMs and is
directly comparable across MDLM-family models (and approximately to AR perplexity).

Supports both MDLM and BD3LM (block diffusion) checkpoints via --model_type.

Usage (single GPU, MDLM):
    python examples/eval_ppl_checkpoints.py \
        --checkpoint_dir /path/to/saves/gsm_infinity/a2d_mdlm_100M \
        --eval_data_path /path/to/eval/data \
        --tokenizer_path /path/to/tokenizer \
        --model_type mdlm \
        --output_file ppl_results.json

Usage (single GPU, BD3LM):
    python examples/eval_ppl_checkpoints.py \
        --checkpoint_dir /path/to/saves/gsm_infinity/a2d_bd3lm_100M \
        --eval_data_path /path/to/eval/data \
        --tokenizer_path /path/to/tokenizer \
        --model_type bd3lm \
        --block_size 32 \
        --output_file ppl_results.json

Usage (multi-GPU via accelerate):
    accelerate launch --num_processes 4 examples/eval_ppl_checkpoints.py ...

Arguments:
    --checkpoint_dir    Root dir containing checkpoint-N subdirectories (auto-sorted by step).
    --checkpoints       Explicit list of checkpoint paths, comma-separated.
    --eval_data_path    HF dataset dir, JSONL file, or text file to evaluate on.
    --tokenizer_path    Path to tokenizer (defaults to first checkpoint).
    --text_field        Dataset column to tokenize (default: "text").
    --max_length        Max sequence length (default: 2048).
    --model_type        "mdlm" or "bd3lm" — must match training (default: mdlm).
    --block_size        Block size for BD3LM (default: 32, ignored for mdlm).
    --scheduler_type    "linear" or "cosine" — must match training (default: linear).
    --time_epsilon      Lower bound for diffusion timestep (default: 1e-3).
    --batch_size        Eval batch size per GPU (default: 16).
    --max_samples       Cap number of eval samples for speed (0 = all, default: 0).
    --output_file       Where to write JSON results (default: ppl_results.json).
"""

import argparse
import json
import os
import re
import sys
from functools import partial

import torch
import torch.nn.functional as F
import transformers

import dllm
from dllm.core.schedulers import make_alpha_scheduler
from dllm.core.trainers.bd3lm import _create_bd3lm_attention_mask
from dllm.core.trainers.utils import NLLMetric, PPLMetric

logger = dllm.utils.get_default_logger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_eval_dataset(eval_data_path: str, text_field: str, tokenizer, max_length: int):
    """Load and tokenize an eval dataset. Returns a HF Dataset with input_ids + labels."""
    from datasets import load_dataset, load_from_disk, Dataset
    import os

    # Try loading as HF dataset directory
    if os.path.isdir(eval_data_path) and os.path.exists(
        os.path.join(eval_data_path, "dataset_info.json")
    ):
        raw = load_from_disk(eval_data_path)
    elif eval_data_path.endswith(".jsonl") or eval_data_path.endswith(".json"):
        raw = load_dataset("json", data_files=eval_data_path, split="train")
    elif eval_data_path.endswith(".txt"):
        raw = load_dataset("text", data_files=eval_data_path, split="train")
        text_field = "text"
    else:
        # Try loading as HF hub id or directory of shards
        try:
            raw = load_from_disk(eval_data_path)
        except Exception:
            raw = load_dataset(eval_data_path, split="train")

    def tokenize(examples):
        tokens = tokenizer(
            examples[text_field],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokens["labels"] = [list(ids) for ids in tokens["input_ids"]]
        return tokens

    logger.info(f"Tokenizing {len(raw)} eval examples ...")
    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    return tokenized


# ---------------------------------------------------------------------------
# Core eval loop (no trainer needed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_checkpoint(
    model_path: str,
    tokenizer,
    eval_dataset,
    scheduler,
    time_epsilon: float,
    batch_size: int,
    max_samples: int,
    device: torch.device,
    mask_token_id: int,
    model_type: str = "mdlm",
    block_size: int = 32,
) -> dict:
    """
    Load one checkpoint and compute VLB NLL + PPL over eval_dataset.

    For model_type="bd3lm" the forward pass mirrors BD3LMTrainer.compute_loss():
    it concatenates [x_t | x_0] (length 2l), uses the specialized block attention
    mask via SDPA, and takes only the first half of the logits.

    Returns a dict with keys: checkpoint, nll, ppl.
    """
    logger.info(f"Loading checkpoint: {model_path}  (model_type={model_type})")

    # BD3LM requires SDPA to honour the 4D block attention mask.
    attn_impl = "sdpa" if model_type == "bd3lm" else None
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation=attn_impl
    )
    model.eval()
    model.to(device)

    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    subset = eval_dataset
    if max_samples > 0:
        subset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    nll_metric = NLLMetric().to(device)
    ppl_metric = PPLMetric().to(device)
    acc_correct = torch.tensor(0, dtype=torch.long, device=device)
    acc_total = torch.tensor(0, dtype=torch.long, device=device)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)        # [b, l]
        labels = batch["labels"].to(device)              # [b, l]
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        b, l = input_ids.shape
        maskable_mask = labels != -100                   # [b, l]

        # === Diffusion timestep + masking (identical for both model types) ===
        t = time_epsilon + (1 - time_epsilon) * torch.rand(b, device=device)
        p_mask = 1.0 - scheduler(t).unsqueeze(1).expand(b, l)   # [b, l]
        w = scheduler.weight(t).unsqueeze(1).expand(b, l)        # [b, l]

        masked_mask = (
            torch.rand((b, l), device=device) < p_mask
        ) & maskable_mask
        noised_input_ids = torch.where(masked_mask, mask_token_id, input_ids)

        # === Forward pass ===
        if model_type == "mdlm":
            outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
            logits = outputs.logits                      # [b, l, V]

        else:  # bd3lm
            # Concatenate [x_t | x_0] — model sees noisy and clean side by side.
            concat_ids = torch.cat([noised_input_ids, input_ids], dim=1)  # [b, 2l]

            # Shared position ids: x_t and x_0 positions are tied token-by-token.
            base_pos = torch.arange(l, device=device).unsqueeze(0).expand(b, l)
            position_ids = torch.cat([base_pos, base_pos], dim=1)         # [b, 2l]

            # 4D boolean block attention mask [1, 1, 2l, 2l] for SDPA.
            # x_t (positions 0..l-1) can attend to x_0 from strictly earlier
            # blocks only — no same-block leakage of clean token identity.
            # Mirrors BD3LMTrainer lines 159-170 exactly.
            block_attn = _create_bd3lm_attention_mask(
                b=None, h=None,
                q_idx=torch.arange(2 * l)[:, None],
                kv_idx=torch.arange(2 * l)[None, :],
                block_size=block_size,
                n=l,
            )
            block_attn = (
                block_attn.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l)
            )
            block_attn = block_attn.to(device)                            # [1, 1, 2l, 2l]

            outputs = model(
                input_ids=concat_ids,
                attention_mask=block_attn,
                position_ids=position_ids,
            )
            # Only the x_t half is used for the loss (mirrors BD3LMTrainer line 200).
            logits = outputs.logits[:, :l]               # [b, l, V]

        # === Weighted per-token NLL (identical for both model types) ===
        token_nll = F.cross_entropy(
            logits.transpose(1, 2),   # [b, V, l]
            input_ids,                # [b, l]
            reduction="none",         # [b, l]
        )
        weighted_nll = token_nll * w * masked_mask.float()       # [b, l]

        nll_metric.update(weighted_nll, maskable_mask.float())
        ppl_metric.update(weighted_nll, maskable_mask.float())

        # === Top-1 accuracy at masked positions ===
        preds = logits.argmax(dim=-1)                    # [b, l]
        acc_correct += ((preds == input_ids) & masked_mask).sum()
        acc_total += masked_mask.sum()

    nll_val = nll_metric.compute().item()
    ppl_val = ppl_metric.compute().item()
    acc_val = (acc_correct.float() / acc_total.clamp(min=1)).item()

    logger.info(f"  NLL={nll_val:.4f}  PPL={ppl_val:.2f}  ACC={acc_val:.4f}")

    del model
    torch.cuda.empty_cache()

    return {"checkpoint": model_path, "nll": nll_val, "ppl": ppl_val, "acc": acc_val}


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _discover_checkpoints(checkpoint_dir: str) -> list[str]:
    """
    Return all checkpoint-N subdirectories inside checkpoint_dir,
    sorted numerically by step number. Also includes checkpoint-final at the end.
    """
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    entries = os.listdir(checkpoint_dir)
    numbered = []
    for e in entries:
        m = re.match(r"^checkpoint-(\d+)$", e)
        if m:
            numbered.append((int(m.group(1)), os.path.join(checkpoint_dir, e)))
    numbered.sort(key=lambda x: x[0])
    paths = [p for _, p in numbered]

    final = os.path.join(checkpoint_dir, "checkpoint-final")
    if os.path.isdir(final):
        paths.append(final)

    if not paths:
        # Maybe checkpoint_dir itself is a single checkpoint
        if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
            paths = [checkpoint_dir]

    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLB perplexity across checkpoints.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Root dir containing checkpoint-N subdirs (auto-discovered).")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated explicit checkpoint paths.")
    parser.add_argument("--eval_data_path", type=str, required=True,
                        help="Path to eval data (HF dataset dir, JSONL, or .txt).")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer. Defaults to first checkpoint.")
    parser.add_argument("--text_field", type=str, default="text",
                        help="Dataset column containing raw text (default: text).")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--model_type", type=str, default="mdlm",
                        choices=["mdlm", "bd3lm"],
                        help="Model type — must match training (default: mdlm).")
    parser.add_argument("--block_size", type=int, default=16,
                        help="Block size for BD3LM — must match training (default: 16, ignored for mdlm).")
    parser.add_argument("--scheduler_type", type=str, default="linear",
                        choices=["linear", "cosine"],
                        help="Noise scheduler used during training (default: linear).")
    parser.add_argument("--time_epsilon", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Cap eval samples (0 = all).")
    parser.add_argument("--output_file", type=str, default="ppl_results.json")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Resolve checkpoints ---
    if args.checkpoints:
        checkpoints = [p.strip() for p in args.checkpoints.split(",") if p.strip()]
    elif args.checkpoint_dir:
        checkpoints = _discover_checkpoints(args.checkpoint_dir)
    else:
        sys.exit("Error: provide --checkpoint_dir or --checkpoints.")

    if not checkpoints:
        sys.exit(f"No checkpoints found in {args.checkpoint_dir}")

    logger.info(f"Found {len(checkpoints)} checkpoint(s) to evaluate.")

    # --- Tokenizer ---
    tokenizer_path = args.tokenizer_path or checkpoints[0]
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    assert tokenizer.mask_token_id is not None, (
        "Tokenizer has no mask_token_id. Check that this is a masked-LM tokenizer."
    )

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Scheduler ---
    scheduler_map = {"linear": "LinearAlphaScheduler", "cosine": "CosineAlphaScheduler"}
    scheduler = make_alpha_scheduler(scheduler_map[args.scheduler_type])
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Noise scheduler: {scheduler_map[args.scheduler_type]}")

    # --- Eval dataset (tokenize once, reuse for all checkpoints) ---
    eval_dataset = _load_eval_dataset(
        args.eval_data_path, args.text_field, tokenizer, args.max_length
    )
    logger.info(f"Eval dataset size: {len(eval_dataset)} examples")

    # --- Eval loop ---
    results = []
    for ckpt in checkpoints:
        result = eval_checkpoint(
            model_path=ckpt,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            scheduler=scheduler,
            time_epsilon=args.time_epsilon,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=device,
            mask_token_id=tokenizer.mask_token_id,
            model_type=args.model_type,
            block_size=args.block_size,
        )
        results.append(result)
        # Print table row
        step = re.search(r"checkpoint-(\w+)", ckpt)
        step_str = step.group(1) if step else os.path.basename(ckpt)
        print(f"step={step_str:>8}  NLL={result['nll']:.4f}  PPL={result['ppl']:.2f}  ACC={result['acc']:.4f}")

    # --- Save results ---
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_file}")

    # --- Summary table ---
    print("\n=== VLB Perplexity across checkpoints ===")
    print(f"{'Checkpoint':<40} {'NLL':>8} {'PPL':>8} {'ACC':>8}")
    print("-" * 62)
    for r in results:
        name = os.path.basename(r["checkpoint"])
        print(f"{name:<40} {r['nll']:>8.4f} {r['ppl']:>8.2f} {r['acc']:>8.4f}")


if __name__ == "__main__":
    main()
