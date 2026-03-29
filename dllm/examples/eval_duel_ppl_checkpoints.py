"""
Evaluate EXACT likelihood (DUEL) of a diffusion LM across multiple checkpoints.

Unlike the VLB estimator in eval_ppl_checkpoints.py (which gives an upper bound
via a single Monte-Carlo timestep sample), this script implements the DUEL
algorithm from "Exact Likelihood for Masked Diffusion via Deterministic
Unmasking" (arXiv:2603.01367).

The idea:
    1. Start with the sequence fully masked.
    2. Run the model forward → softmax → P.
    3. Pick positions to unmask via F (confidence-based or random).
    4. For each position ℓ to unmask: accumulate log P_ℓ[x_ℓ] (ground-truth token).
    5. Reveal those tokens (set z[ℓ] = x_ℓ).
    6. Repeat until no masks remain.
    7. PPL = exp(-ll / num_tokens).

This gives the exact log-likelihood under the model's unmasking policy π_F,
not a variational bound.

Supports both MDLM and BD3LM checkpoints via --model_type.

Usage (single GPU, MDLM):
    python examples/eval_duel_ppl_checkpoints.py \\
        --checkpoint_dir /path/to/saves/mdlm_100M \\
        --eval_data_path /path/to/eval/data \\
        --tokenizer_path /path/to/tokenizer \\
        --model_type mdlm \\
        --steps 128 \\
        --output_file duel_ppl_results.json

Usage (single GPU, BD3LM):
    python examples/eval_duel_ppl_checkpoints.py \\
        --checkpoint_dir /path/to/saves/bd3lm_100M \\
        --eval_data_path /path/to/eval/data \\
        --tokenizer_path /path/to/tokenizer \\
        --model_type bd3lm \\
        --block_size 32 \\
        --steps 128 \\
        --output_file duel_ppl_results.json

Arguments:
    --checkpoint_dir    Root dir containing checkpoint-N subdirectories.
    --checkpoints       Explicit list of checkpoint paths, comma-separated.
    --eval_data_path    HF dataset dir, JSONL file, or text file to evaluate on.
    --tokenizer_path    Path to tokenizer (defaults to first checkpoint).
    --text_field        Dataset column to tokenize (default: "text").
    --max_length        Max sequence length (default: 2048).
    --model_type        "mdlm" or "bd3lm" (default: mdlm).
    --block_size        Block size for BD3LM (default: 32, ignored for mdlm).
    --scheduler_type    "linear" or "cosine" — must match training (default: linear).
    --steps             Number of unmasking steps (default: 0 = unmask one token per step).
    --remasking         Unmasking strategy: "low_confidence" or "random" (default: low_confidence).
    --batch_size        Eval batch size per GPU (default: 4).
    --max_samples       Cap number of eval samples (0 = all, default: 0).
    --output_file       Where to write JSON results (default: duel_ppl_results.json).
"""

import argparse
import json
import math
import os
import re
import sys

import torch
import torch.nn.functional as F
import transformers

import dllm
from dllm.core.schedulers import make_alpha_scheduler
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.core.trainers.bd3lm import _create_bd3lm_attention_mask

logger = dllm.utils.get_default_logger(__name__)


# ---------------------------------------------------------------------------
# Data loading (reused from eval_ppl_checkpoints.py)
# ---------------------------------------------------------------------------

def _load_eval_dataset(eval_data_path: str, text_field: str, tokenizer, max_length: int):
    """Load and tokenize an eval dataset. Returns a HF Dataset with input_ids + labels."""
    from datasets import load_dataset, load_from_disk

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
    tokenized.set_format(type="torch")
    return tokenized


# ---------------------------------------------------------------------------
# DUEL: exact likelihood for a single sequence
# ---------------------------------------------------------------------------

@torch.no_grad()
def _duel_loglikelihood_mdlm(
    model,
    x: torch.Tensor,               # [L] ground-truth token ids
    mask_token_id: int,
    scheduler,
    steps: int,
    remasking: str,
    device: torch.device,
) -> tuple[float, int]:
    """
    DUEL exact log-likelihood for a single MDLM sequence.

    Returns (log_likelihood, num_tokens).
    """
    L = x.shape[0]
    x = x.to(device)

    # Step 1: start fully masked
    z = torch.full((1, L), mask_token_id, dtype=torch.long, device=device)
    ll = 0.0

    # Determine number of tokens to unmask per step
    if steps <= 0 or steps >= L:
        # One token per step (most precise)
        steps = L

    mask_index = torch.ones((1, L), dtype=torch.bool, device=device)
    num_transfer = get_num_transfer_tokens(
        mask_index=mask_index,
        steps=steps,
        scheduler=scheduler,
        stochastic=False,
    )  # [1, effective_steps]
    effective_steps = num_transfer.shape[1]

    num_unmasked = 0
    for i_step in range(effective_steps):
        # Which positions are still masked?
        masked = (z[0] == mask_token_id)
        if not masked.any():
            break

        # Forward pass
        outputs = model(input_ids=z)
        logits = outputs.logits  # [1, L, V]
        log_probs = F.log_softmax(logits[0].float(), dim=-1)  # [L, V]

        # How many tokens to unmask this step
        k = int(num_transfer[0, i_step].item())
        if k <= 0:
            continue
        k = min(k, masked.sum().item())

        # Compute confidence for selection: P(x_true | z) at masked positions
        # We use the ground-truth token's probability as confidence
        true_log_probs = log_probs[torch.arange(L, device=device), x]  # [L]

        if remasking == "low_confidence":
            confidence = true_log_probs.clone()
        elif remasking == "random":
            confidence = torch.rand(L, device=device)
        else:
            raise ValueError(f"Unknown remasking strategy: {remasking}")

        # Only consider masked positions
        confidence[~masked] = -float("inf")

        # Select top-k highest confidence positions
        _, sel = torch.topk(confidence, k=k)

        # Accumulate log P_ℓ[x_ℓ] for selected positions
        for idx in sel:
            ll += true_log_probs[idx].item()

        # Reveal ground-truth tokens at selected positions
        z[0, sel] = x[sel]
        num_unmasked += k

    return ll, L


@torch.no_grad()
def _duel_loglikelihood_bd3lm(
    model,
    x: torch.Tensor,               # [L] ground-truth token ids
    mask_token_id: int,
    scheduler,
    steps: int,
    block_size: int,
    remasking: str,
    device: torch.device,
) -> tuple[float, int]:
    """
    DUEL exact log-likelihood for a single BD3LM sequence.

    Processes block-by-block: for each block, iteratively unmask tokens
    while previous blocks provide clean context — mirroring BD3LM's
    block-causal attention pattern at inference.

    Returns (log_likelihood, num_tokens).
    """
    L = x.shape[0]
    x = x.to(device)

    num_blocks = math.ceil(L / block_size)
    steps_per_block = max(1, math.ceil(steps / num_blocks)) if steps > 0 else block_size

    # Start fully masked
    z = torch.full((1, L), mask_token_id, dtype=torch.long, device=device)
    ll = 0.0

    for b_idx in range(num_blocks):
        blk_start = b_idx * block_size
        blk_end = min(blk_start + block_size, L)
        blk_len = blk_end - blk_start

        # Determine unmasking schedule for this block
        block_mask = torch.ones((1, blk_len), dtype=torch.bool, device=device)
        num_transfer = get_num_transfer_tokens(
            mask_index=block_mask,
            steps=steps_per_block,
            scheduler=scheduler,
            stochastic=False,
        )  # [1, effective_steps]
        effective_steps = num_transfer.shape[1]

        for i_step in range(effective_steps):
            masked_in_block = (z[0, blk_start:blk_end] == mask_token_id)
            if not masked_in_block.any():
                break

            # Build BD3LM attention mask + position ids for the full sequence
            seq_len = L
            q_idx = torch.arange(2 * seq_len, device=device)[:, None]
            kv_idx = torch.arange(2 * seq_len, device=device)[None, :]

            block_attn = _create_bd3lm_attention_mask(
                b=None, h=None,
                q_idx=q_idx,
                kv_idx=kv_idx,
                block_size=block_size,
                n=seq_len,
            )
            block_attn = block_attn.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 2L, 2L]

            # BD3LM forward: concat [z | x_clean_so_far]
            # For DUEL, the "x_0" side contains revealed tokens (already unmasked in z)
            # plus the ground truth for positions in earlier blocks
            concat_ids = torch.cat([z, x.unsqueeze(0)], dim=1)  # [1, 2L]
            base_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            position_ids = torch.cat([base_pos, base_pos], dim=1)  # [1, 2L]

            outputs = model(
                input_ids=concat_ids,
                attention_mask=block_attn,
                position_ids=position_ids,
            )
            logits = outputs.logits[:, :seq_len]  # [1, L, V] — x_t half only
            log_probs = F.log_softmax(logits[0].float(), dim=-1)  # [L, V]

            # How many tokens to unmask this step
            k = int(num_transfer[0, i_step].item())
            if k <= 0:
                continue
            k = min(k, masked_in_block.sum().item())

            # Confidence only over this block's masked positions
            blk_indices = torch.arange(blk_start, blk_end, device=device)
            true_log_probs_block = log_probs[blk_indices, x[blk_start:blk_end]]  # [blk_len]

            if remasking == "low_confidence":
                confidence = true_log_probs_block.clone()
            elif remasking == "random":
                confidence = torch.rand(blk_len, device=device)
            else:
                raise ValueError(f"Unknown remasking strategy: {remasking}")

            confidence[~masked_in_block] = -float("inf")

            _, sel = torch.topk(confidence, k=k)

            # Accumulate log-probs
            for idx in sel:
                ll += true_log_probs_block[idx].item()

            # Reveal ground-truth tokens
            global_sel = sel + blk_start
            z[0, global_sel] = x[global_sel]

    return ll, L


# ---------------------------------------------------------------------------
# Checkpoint evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_checkpoint(
    model_path: str,
    tokenizer,
    eval_dataset,
    scheduler,
    steps: int,
    batch_size: int,
    max_samples: int,
    device: torch.device,
    mask_token_id: int,
    model_type: str = "mdlm",
    block_size: int = 32,
    remasking: str = "low_confidence",
) -> dict:
    """
    Load one checkpoint and compute DUEL exact log-likelihood + PPL.

    Returns a dict with keys: checkpoint, nll, ppl, num_tokens.
    """
    logger.info(f"Loading checkpoint: {model_path}  (model_type={model_type})")

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
        n = min(max_samples, len(eval_dataset))
        indices = torch.randperm(len(eval_dataset))[:n].tolist()
        subset = eval_dataset.select(indices)

    total_ll = 0.0
    total_tokens = 0

    # Process one sequence at a time (DUEL is inherently sequential per sample)
    for i in range(len(subset)):
        input_ids = subset[i]["input_ids"].to(device)  # [L]
        labels = subset[i]["labels"].to(device)         # [L]

        # Use only the non-padding tokens
        valid_mask = labels != -100
        if valid_mask.sum() == 0:
            continue
        x = input_ids[valid_mask]

        if model_type == "mdlm":
            ll, n_tok = _duel_loglikelihood_mdlm(
                model=model,
                x=x,
                mask_token_id=mask_token_id,
                scheduler=scheduler,
                steps=steps,
                remasking=remasking,
                device=device,
            )
        else:  # bd3lm
            ll, n_tok = _duel_loglikelihood_bd3lm(
                model=model,
                x=x,
                mask_token_id=mask_token_id,
                scheduler=scheduler,
                steps=steps,
                block_size=block_size,
                remasking=remasking,
                device=device,
            )

        total_ll += ll
        total_tokens += n_tok

        if (i + 1) % 50 == 0:
            running_nll = -total_ll / total_tokens
            running_ppl = math.exp(running_nll)
            logger.info(
                f"  [{i+1}/{len(subset)}] running NLL={running_nll:.4f} PPL={running_ppl:.2f}"
            )

    nll = -total_ll / total_tokens
    ppl = math.exp(nll)

    logger.info(f"  DUEL NLL={nll:.4f}  PPL={ppl:.2f}  ({total_tokens} tokens)")

    del model
    torch.cuda.empty_cache()

    return {
        "checkpoint": model_path,
        "nll": nll,
        "ppl": ppl,
        "num_tokens": total_tokens,
    }


# ---------------------------------------------------------------------------
# Checkpoint discovery (reused from eval_ppl_checkpoints.py)
# ---------------------------------------------------------------------------

def _discover_checkpoints(checkpoint_dir: str) -> list[str]:
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
        if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
            paths = [checkpoint_dir]

    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DUEL exact likelihood perplexity across checkpoints."
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoints", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--model_type", type=str, default="mdlm",
                        choices=["mdlm", "bd3lm"])
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--scheduler_type", type=str, default="linear",
                        choices=["linear", "cosine"])
    parser.add_argument("--steps", type=int, default=0,
                        help="Unmasking steps (0 = one token per step, most precise).")
    parser.add_argument("--remasking", type=str, default="low_confidence",
                        choices=["low_confidence", "random"],
                        help="Unmasking strategy: high-confidence first or random.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Not used for DUEL (sequential per sample), kept for CLI compat.")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_file", type=str, default="duel_ppl_results.json")
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
    logger.info(f"Unmasking strategy: {args.remasking}")
    logger.info(f"Unmasking steps: {args.steps if args.steps > 0 else 'L (one per token)'}")

    # --- Eval dataset ---
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
            steps=args.steps,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=device,
            mask_token_id=tokenizer.mask_token_id,
            model_type=args.model_type,
            block_size=args.block_size,
            remasking=args.remasking,
        )
        results.append(result)
        step = re.search(r"checkpoint-(\w+)", ckpt)
        step_str = step.group(1) if step else os.path.basename(ckpt)
        print(f"step={step_str:>8}  DUEL_NLL={result['nll']:.4f}  DUEL_PPL={result['ppl']:.2f}")

    # --- Save results ---
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_file}")

    # --- Summary table ---
    print("\n=== DUEL Exact Perplexity across checkpoints ===")
    print(f"{'Checkpoint':<40} {'NLL':>8} {'PPL':>8} {'Tokens':>8}")
    print("-" * 68)
    for r in results:
        name = os.path.basename(r["checkpoint"])
        print(f"{name:<40} {r['nll']:>8.4f} {r['ppl']:>8.2f} {r['num_tokens']:>8}")


if __name__ == "__main__":
    main()
