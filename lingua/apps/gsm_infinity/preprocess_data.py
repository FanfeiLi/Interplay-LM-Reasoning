"""
Convert GSM-Infinity composition_hf training data into lingua's preshuffled
JSONL chunk format (*.chunk.*.jsonl with {"text": "..."} lines).

The composition_hf data lives at:
    data/composition_hf/train/{op}/shard-*.jsonl
Each line has: {"problem": ..., "question": ..., "solution": ..., ...}

This script reads all shards for ops in [op_min, op_max], converts each
example to the standard training text format:
    <question> {problem} {question} </question> <solution> {body} </solution> <answer> {answer} </answer>
and writes balanced, shuffled JSONL chunks to the output directory.

Output directory layout (compatible with lingua data loader):
    data/composition_lingua/gsm_infinity/gsm_infinity.chunk.00000.jsonl
    data/composition_lingua/gsm_infinity/gsm_infinity.chunk.00001.jsonl
    ...

Usage:
    python -m apps.gsm_infinity.preprocess_data \
        --raw_data_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/train \
        --output_dir   /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_lingua/gsm_infinity \
        --op_min 2 --op_max 10 \
        --token_budget 10B \
        --tokenizer_path /fast/pmayilvahanan/Interplay-LM-Reasoning/model_configs/qwen2_400M \
        --lines_per_chunk 10000
"""

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Tuple


def _split_solution(sol: str) -> Tuple[str, str]:
    if not sol:
        return "", ""
    if "Answer:" not in sol:
        return sol.strip(), ""
    pre, ans = sol.rsplit("Answer:", 1)
    ans = ans.strip().splitlines()[0].strip().rstrip(".")
    return pre.strip(), ans


def format_example(example: dict) -> str:
    problem = (example.get("problem") or "").strip()
    question = (example.get("question") or "").strip()
    solution = (example.get("solution") or "").strip()
    body, answer = _split_solution(solution)
    pq = (problem + " " + question).strip()
    return f"<question> {pq} </question> <solution> {body} </solution> <answer> {answer} </answer>"


def parse_budget(budget_str: str) -> int:
    """Parse token budget strings like '10B', '500M', '1T'."""
    budget_str = budget_str.strip().upper()
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000, "T": 1_000_000_000_000}
    for suffix, mult in multipliers.items():
        if budget_str.endswith(suffix):
            return int(float(budget_str[:-1]) * mult)
    return int(budget_str)


def estimate_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def main():
    parser = argparse.ArgumentParser(description="Convert composition_hf to lingua chunk format")
    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--op_min", type=int, default=2)
    parser.add_argument("--op_max", type=int, default=10)
    parser.add_argument("--token_budget", type=str, default="10B")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="HF tokenizer path for token counting (if None, estimate ~4 chars/token)")
    parser.add_argument("--lines_per_chunk", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    budget = parse_budget(args.token_budget)
    print(f"Token budget: {budget:,} tokens")

    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        print(f"Loaded tokenizer from {args.tokenizer_path} (vocab size: {len(tokenizer)})")

    raw_dir = Path(args.raw_data_dir)
    ops = list(range(args.op_min, args.op_max + 1))
    n_ops = len(ops)
    budget_per_op = budget // n_ops

    all_texts = []
    total_tokens = 0

    for op in ops:
        op_dir = raw_dir / str(op)
        if not op_dir.exists():
            print(f"  WARNING: {op_dir} not found, skipping op={op}")
            continue

        shards = sorted(op_dir.glob("*.jsonl"))
        if not shards:
            print(f"  WARNING: No JSONL files in {op_dir}, skipping op={op}")
            continue

        op_texts = []
        op_tokens = 0

        for shard in shards:
            if op_tokens >= budget_per_op:
                break
            with open(shard) as f:
                for line in f:
                    if op_tokens >= budget_per_op:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    example = json.loads(line)
                    text = format_example(example)
                    if tokenizer:
                        n_tok = estimate_tokens(text, tokenizer)
                    else:
                        n_tok = len(text) // 4
                    op_texts.append(text)
                    op_tokens += n_tok

        total_tokens += op_tokens
        all_texts.extend(op_texts)
        print(f"  op={op}: {len(op_texts):>8,} examples, ~{op_tokens:>12,} tokens "
              f"(from {len(shards)} shards)")

    print(f"\nTotal: {len(all_texts):,} examples, ~{total_tokens:,} tokens")

    # Shuffle
    print("Shuffling...")
    random.shuffle(all_texts)

    # Write chunks
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_idx = 0
    for start in range(0, len(all_texts), args.lines_per_chunk):
        chunk = all_texts[start : start + args.lines_per_chunk]
        chunk_path = out_dir / f"gsm_infinity.chunk.{chunk_idx:05d}.jsonl"
        with open(chunk_path, "w") as f:
            for text in chunk:
                f.write(json.dumps({"text": text}) + "\n")
        chunk_idx += 1

    print(f"Wrote {chunk_idx} chunks to {out_dir}")
    print(f"Chunk pattern: gsm_infinity.chunk.*.jsonl ({args.lines_per_chunk} lines each)")


if __name__ == "__main__":
    main()
