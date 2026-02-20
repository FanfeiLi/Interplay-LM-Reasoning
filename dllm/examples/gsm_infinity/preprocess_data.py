"""
Preprocess GSM-Infinity composition_hf JSONL data into a pre-tokenized
HuggingFace dataset ready for dLLM pre-training.

Pipeline:
  1. Read JSONL shards for op 2-10 in parallel (one process per shard)
  2. Convert each example to text via compose_text()
  3. Tokenize in batches and concatenate into fixed-length (2048) sequences
  4. Merge chunks from all workers and save as Arrow dataset

Usage:
    source /fast/pmayilvahanan/Interplay-LM-Reasoning/gsm_pretrain/bin/activate
    cd /fast/pmayilvahanan/Interplay-LM-Reasoning/dllm

    python examples/gsm_infinity/preprocess_data.py \
        --data_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/train \
        --tokenizer_path /fast/pmayilvahanan/Interplay-LM-Reasoning/dllm/model_configs/a2d_qwen2_100M \
        --output_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf_dllm_tokenized \
        --op_min 2 --op_max 10 --num_workers 16
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Tuple

import transformers
from datasets import Dataset, DatasetDict
from tqdm import tqdm


# ---------- Text composition (from utils/text_preprocess.py) ----------

def _split_solution(sol: str) -> Tuple[str, str]:
    if not sol:
        return "", ""
    if "Answer:" not in sol:
        return sol.strip(), ""
    pre, ans = sol.rsplit("Answer:", 1)
    ans = ans.strip().splitlines()[0].strip().rstrip(".")
    return pre.strip(), ans


def compose_text(obj: dict) -> str:
    problem = (obj.get("problem") or "").strip()
    question = (obj.get("question") or "").strip()
    solution = (obj.get("solution") or "").strip()
    if not (problem or question or solution):
        return ""
    sol_body, answer = _split_solution(solution)
    pq = (problem + " " + question).strip()
    parts = []
    if pq:
        parts.extend(["<question>", pq, "</question>"])
    if sol_body:
        parts.extend(["<solution>", sol_body, "</solution>"])
    if answer:
        parts.extend(["<answer>", answer, "</answer>"])
    return " ".join(parts)


# ---------- Per-shard worker ----------

# Global tokenizer (set once per worker via initializer)
_worker_tokenizer = None
_worker_seq_length = None


def _init_worker(tokenizer_path: str, seq_length: int):
    global _worker_tokenizer, _worker_seq_length
    _worker_tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    _worker_seq_length = seq_length


def _process_one_shard(shard_path: str) -> list[list[int]]:
    """Process a single shard: read JSONL -> compose text -> tokenize -> chunk."""
    tokenizer = _worker_tokenizer
    seq_length = _worker_seq_length
    eos_id = tokenizer.eos_token_id

    token_buffer = []
    chunks = []
    batch_size = 5_000
    texts_batch = []
    n_examples = 0

    with open(shard_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = compose_text(obj)
            if not text:
                continue

            texts_batch.append(text)
            n_examples += 1

            if len(texts_batch) >= batch_size:
                _tokenize_batch(texts_batch, tokenizer, eos_id,
                                token_buffer, chunks, seq_length)
                texts_batch = []

    if texts_batch:
        _tokenize_batch(texts_batch, tokenizer, eos_id,
                        token_buffer, chunks, seq_length)

    shard_name = os.path.basename(shard_path)
    print(f"  [{shard_name}] {n_examples:,} examples -> {len(chunks):,} chunks "
          f"({len(token_buffer):,} leftover tokens)")
    return chunks


def _tokenize_batch(texts, tokenizer, eos_id, token_buffer, chunks, seq_length):
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    for ids in encoded:
        token_buffer.extend(ids)
        if eos_id is not None and (not ids or ids[-1] != eos_id):
            token_buffer.append(eos_id)
    while len(token_buffer) >= seq_length:
        chunks.append(token_buffer[:seq_length])
        del token_buffer[:seq_length]


# ---------- Shard discovery ----------

def find_jsonl_shards(data_dir: str, op_min: int, op_max: int) -> list[str]:
    shard_files = []
    for op in range(op_min, op_max + 1):
        op_dir = os.path.join(data_dir, str(op))
        if not os.path.isdir(op_dir):
            print(f"  Warning: Op dir not found: {op_dir}")
            continue
        for f in sorted(os.listdir(op_dir)):
            if f.endswith(".jsonl"):
                shard_files.append(os.path.join(op_dir, f))
    return shard_files


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess + tokenize GSM-Infinity data for dLLM pre-training"
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/train",
    )
    parser.add_argument(
        "--tokenizer_path", type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/dllm/model_configs/a2d_qwen2_100M",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf_dllm_tokenized",
    )
    parser.add_argument("--op_min", type=int, default=2)
    parser.add_argument("--op_max", type=int, default=10)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--test_split_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers for processing shards")
    parser.add_argument("--num_proc_save", type=int, default=16,
                        help="Processes for saving to disk")
    args = parser.parse_args()

    print(f"Data dir:       {args.data_dir}")
    print(f"Tokenizer:      {args.tokenizer_path}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Op range:       {args.op_min}-{args.op_max}")
    print(f"Seq length:     {args.seq_length}")
    print(f"Workers:        {args.num_workers}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Verify tokenizer loads
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"Tokenizer vocab_size={tokenizer.vocab_size}, "
          f"eos={tokenizer.eos_token}({tokenizer.eos_token_id})\n")

    # --- Find shards ---
    shard_files = find_jsonl_shards(args.data_dir, args.op_min, args.op_max)
    print(f"Found {len(shard_files)} shard files")
    if not shard_files:
        print("ERROR: No JSONL shard files found!")
        sys.exit(1)

    # --- Process shards in parallel ---
    t0 = time.time()
    print(f"\nProcessing {len(shard_files)} shards with {args.num_workers} workers...")

    with Pool(
        processes=args.num_workers,
        initializer=_init_worker,
        initargs=(args.tokenizer_path, args.seq_length),
    ) as pool:
        results = pool.map(_process_one_shard, shard_files)

    # Merge all chunks
    all_chunks = []
    for shard_chunks in results:
        all_chunks.extend(shard_chunks)

    elapsed = time.time() - t0
    print(f"\nTokenization complete in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Total chunks: {len(all_chunks):,} (each {args.seq_length} tokens)")
    total_tokens = len(all_chunks) * args.seq_length
    print(f"Total tokens:  {total_tokens:,} ({total_tokens/1e9:.2f}B)")

    # --- Build HF dataset ---
    print(f"\nBuilding HuggingFace dataset...")
    dataset = Dataset.from_dict({
        "input_ids": all_chunks,
        "labels": [c[:] for c in all_chunks],
    })
    dataset = dataset.shuffle(seed=args.seed)

    if args.test_split_size > 0 and len(dataset) > args.test_split_size:
        ds_dict = dataset.train_test_split(
            test_size=args.test_split_size, seed=args.seed
        )
    else:
        ds_dict = DatasetDict({"train": dataset})

    print(f"Splits: { {k: len(v) for k, v in ds_dict.items()} }")

    # --- Save ---
    print(f"Saving to {args.output_dir}...")
    ds_dict.save_to_disk(args.output_dir, num_proc=args.num_proc_save)

    # --- Verify ---
    print("\n--- Verification ---")
    sample = ds_dict["train"][0]
    print(f"  input_ids length: {len(sample['input_ids'])}")
    decoded = tokenizer.decode(sample["input_ids"][:80])
    print(f"  First 80 tokens: {decoded[:300]}...")
    print(f"\nDone! Pre-tokenized dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
