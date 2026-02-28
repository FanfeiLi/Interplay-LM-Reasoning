"""
Convert GSM-Infinity composition_hf training data into lingua's preshuffled
JSONL chunk format (*.chunk.*.jsonl with {"text": "..."} lines).

The composition_hf data lives at:
    data/composition_hf/train/{op}/shard-*.jsonl
Each line has: {"problem": ..., "question": ..., "solution": ..., ...}

This script reads all shards for ops in [op_min, op_max] in parallel across
CPUs, converts each example to the standard training text format:
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
        --lines_per_chunk 10000 \
        --workers 16
"""

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


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


CHARS_PER_TOKEN = 4


def _process_shard(shard_path: str, token_limit: int) -> Tuple[List[str], int]:
    """Read a single shard file, return (texts, approx_token_count)."""
    texts = []
    tokens = 0
    with open(shard_path) as f:
        for line in f:
            if tokens >= token_limit:
                break
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            text = format_example(example)
            n_tok = len(text) // CHARS_PER_TOKEN
            texts.append(text)
            tokens += n_tok
    return texts, tokens


def _process_op(op: int, raw_dir: str, budget_per_op: int) -> Tuple[int, List[str], int]:
    """Process all shards for a single op level. Runs in a worker process."""
    op_dir = Path(raw_dir) / str(op)
    if not op_dir.exists():
        return op, [], 0

    shards = sorted(op_dir.glob("*.jsonl"))
    if not shards:
        return op, [], 0

    op_texts = []
    op_tokens = 0

    for shard in shards:
        if op_tokens >= budget_per_op:
            break
        remaining = budget_per_op - op_tokens
        texts, tokens = _process_shard(str(shard), remaining)
        op_texts.extend(texts)
        op_tokens += tokens

    return op, op_texts, op_tokens


def main():
    parser = argparse.ArgumentParser(description="Convert composition_hf to lingua chunk format")
    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--op_min", type=int, default=2)
    parser.add_argument("--op_max", type=int, default=10)
    parser.add_argument("--token_budget", type=str, default="10B")
    parser.add_argument("--lines_per_chunk", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: min(n_ops, cpu_count))")
    args = parser.parse_args()

    random.seed(args.seed)
    budget = parse_budget(args.token_budget)

    ops = list(range(args.op_min, args.op_max + 1))
    n_ops = len(ops)
    budget_per_op = budget // n_ops
    n_workers = args.workers or min(n_ops, os.cpu_count() or 4)

    print(f"Token budget: {budget:,} (~{budget_per_op:,} per op)")
    print(f"Ops: {ops} | Workers: {n_workers}")
    print(f"Token estimation: ~{CHARS_PER_TOKEN} chars/token (no tokenizer needed)")
    print()

    all_texts = []
    total_tokens = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_op, op, args.raw_data_dir, budget_per_op): op
            for op in ops
        }
        for future in as_completed(futures):
            op, op_texts, op_tokens = future.result()
            total_tokens += op_tokens
            all_texts.extend(op_texts)
            print(f"  op={op}: {len(op_texts):>8,} examples, ~{op_tokens:>12,} tokens")

    print(f"\nTotal: {len(all_texts):,} examples, ~{total_tokens:,} tokens")

    print("Shuffling...")
    random.shuffle(all_texts)

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
