"""
Evaluate perplexity of an autoregressive LM across checkpoints.

Standard causal LM perplexity: PPL = exp(mean cross-entropy loss).
Output format matches eval_ppl_checkpoints.py for easy comparison.

Data loading uses the same format as eval_ppl_checkpoints.py (MDLM):
accepts HF dataset dirs, JSONL, or text files with raw text that gets
tokenized on the fly. This ensures both AR and MDLM evals use the
exact same validation data for fair comparison.

Usage:
    python examples/eval_ppl_ar.py \
        --checkpoint_dir /path/to/saves/tinystories/ar_tinystories_100M_seed0_... \
        --eval_data_path /path/to/tinystories/training_data/validation \
        --tokenizer_path /path/to/model_configs/qwen2_tinystories_100M \
        --output_file ppl_ar_results.json

    # Or evaluate a single checkpoint:
    python examples/eval_ppl_ar.py \
        --checkpoints /path/to/checkpoint-3000 \
        --eval_data_path /path/to/validation \
        --tokenizer_path /path/to/tokenizer \
        --output_file ppl_ar_results.json
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

logger = dllm.utils.get_default_logger(__name__)


def _load_eval_dataset(eval_data_path: str, text_field: str, tokenizer, max_length: int):
    """Load and tokenize an eval dataset. Returns a HF Dataset with input_ids + labels.

    Same loader as eval_ppl_checkpoints.py so both AR and MDLM evals
    operate on identical validation data.
    """
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
    return tokenized


def _discover_checkpoints(checkpoint_dir: str) -> list:
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


@torch.no_grad()
def eval_checkpoint(
    model_path: str,
    tokenizer,
    eval_dataset,
    batch_size: int,
    max_samples: int,
    device: torch.device,
) -> dict:
    """Load one AR checkpoint and compute cross-entropy NLL + PPL."""
    logger.info(f"Loading checkpoint: {model_path}")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
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

    total_nll = 0.0
    total_tokens = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)    # [b, l]
        labels = batch["labels"].to(device)           # [b, l]
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [b, l, V]

        # Shift: predict token i+1 from position i
        shift_logits = logits[:, :-1, :].contiguous()   # [b, l-1, V]
        shift_labels = labels[:, 1:].contiguous()        # [b, l-1]

        # Per-token cross-entropy (ignore -100)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        n_tokens = (shift_labels != -100).sum().item()
        total_nll += loss.item()
        total_tokens += n_tokens

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(mean_nll)

    logger.info(f"  NLL={mean_nll:.4f}  PPL={ppl:.2f}  tokens={total_tokens:,}")

    del model
    torch.cuda.empty_cache()

    return {"checkpoint": model_path, "nll": mean_nll, "ppl": ppl}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AR perplexity across checkpoints.")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Root dir containing checkpoint-N subdirs.")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated explicit checkpoint paths.")
    parser.add_argument("--eval_data_path", type=str, required=True,
                        help="Path to eval data (HF dataset dir, JSONL, or .txt).")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer. Defaults to first checkpoint.")
    parser.add_argument("--text_field", type=str, default="text",
                        help="Dataset column containing raw text (default: text).")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Cap eval samples (0 = all).")
    parser.add_argument("--output_file", type=str, default="ppl_ar_results.json")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.checkpoints:
        checkpoints = [p.strip() for p in args.checkpoints.split(",") if p.strip()]
    elif args.checkpoint_dir:
        checkpoints = _discover_checkpoints(args.checkpoint_dir)
    else:
        sys.exit("Error: provide --checkpoint_dir or --checkpoints.")

    if not checkpoints:
        sys.exit(f"No checkpoints found.")

    logger.info(f"Found {len(checkpoints)} checkpoint(s) to evaluate.")

    tokenizer_path = args.tokenizer_path or checkpoints[0]
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    eval_dataset = _load_eval_dataset(
        args.eval_data_path, args.text_field, tokenizer, args.max_length
    )
    logger.info(f"Eval dataset size: {len(eval_dataset)} examples")

    results = []
    for ckpt in checkpoints:
        result = eval_checkpoint(
            model_path=ckpt,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            device=device,
        )
        results.append(result)
        step = re.search(r"checkpoint-(\w+)", ckpt)
        step_str = step.group(1) if step else os.path.basename(ckpt)
        print(f"step={step_str:>8}  NLL={result['nll']:.4f}  PPL={result['ppl']:.2f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_file}")

    print("\n=== AR Perplexity across checkpoints ===")
    print(f"{'Checkpoint':<40} {'NLL':>8} {'PPL':>8}")
    print("-" * 58)
    for r in results:
        name = os.path.basename(r["checkpoint"])
        print(f"{name:<40} {r['nll']:>8.4f} {r['ppl']:>8.2f}")


if __name__ == "__main__":
    main()
