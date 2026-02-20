"""
Preprocess GSM-Infinity composition_hf JSONL data into a pre-tokenized
HuggingFace dataset ready for dLLM pre-training.

Pipeline:
  1. Stream JSONL shards for op 2-10
  2. Convert each example to text via compose_text()
  3. Tokenize all text and concatenate into fixed-length (2048) sequences
  4. Save as Arrow dataset with input_ids + labels columns

This means the training script can load with load_preprocessed_data=True
and skip the expensive tokenization step entirely.

Usage:
    source /fast/pmayilvahanan/Interplay-LM-Reasoning/gsm_pretrain/bin/activate
    cd /fast/pmayilvahanan/Interplay-LM-Reasoning/dllm

    python examples/gsm_infinity/preprocess_data.py \
        --data_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/train \
        --tokenizer_path /fast/pmayilvahanan/Interplay-LM-Reasoning/dllm/model_configs/a2d_qwen2_100M \
        --output_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf_dllm_tokenized \
        --op_min 2 --op_max 10
"""

import argparse
import json
import os
import sys
from itertools import chain
from pathlib import Path
from typing import Tuple

import transformers
from datasets import Dataset, DatasetDict
from tqdm import tqdm


# ---------- Text composition (from utils/text_preprocess.py) ----------

def _split_solution(sol: str) -> Tuple[str, str]:
    """Split solution text into body and answer parts based on 'Answer:' marker."""
    if not sol:
        return "", ""
    if "Answer:" not in sol:
        return sol.strip(), ""
    pre, ans = sol.rsplit("Answer:", 1)
    ans = ans.strip().splitlines()[0].strip().rstrip(".")
    return pre.strip(), ans


def compose_text(obj: dict, add_special_tokens: bool = True) -> str:
    """Compose a training text string from a raw example."""
    problem = (obj.get("problem") or "").strip()
    question = (obj.get("question") or "").strip()
    solution = (obj.get("solution") or "").strip()

    if add_special_tokens and (problem or question or solution):
        sol_body, answer = _split_solution(solution)
        pq = (problem + " " + question).strip()
        parts = []
        if pq:
            parts.extend(["<question>", pq, "</question>"])
        if sol_body:
            parts.extend(["<solution>", sol_body, "</solution>"])
        if answer:
            parts.extend(["<answer>", answer, "</answer>"])
        text = " ".join([p for p in parts if p]).strip()
        if text:
            return text
    return ""


# ---------- Main preprocessing ----------

def find_jsonl_shards(data_dir: str, op_min: int, op_max: int) -> list[str]:
    """Find all JSONL shard files for op levels in [op_min, op_max]."""
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


def tokenize_and_chunk_shards(
    shard_files: list[str],
    tokenizer: transformers.PreTrainedTokenizer,
    seq_length: int = 2048,
    batch_size: int = 10_000,
) -> list[list[int]]:
    """Read shards, tokenize, concatenate, and chunk into fixed-length sequences.

    Processes in streaming batches to manage memory.
    Returns list of token-id lists, each of length seq_length.
    """
    eos_id = tokenizer.eos_token_id
    all_chunks = []
    token_buffer = []

    total_examples = 0
    total_tokens = 0

    for shard_path in tqdm(shard_files, desc="Processing shards"):
        texts_batch = []
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = compose_text(obj, add_special_tokens=True)
                if not text:
                    continue

                texts_batch.append(text)
                total_examples += 1

                if len(texts_batch) >= batch_size:
                    _flush_batch(texts_batch, tokenizer, eos_id, token_buffer,
                                 all_chunks, seq_length)
                    total_tokens += len(texts_batch) * 50  # rough estimate
                    texts_batch = []

        # Flush remaining in shard
        if texts_batch:
            _flush_batch(texts_batch, tokenizer, eos_id, token_buffer,
                         all_chunks, seq_length)
            texts_batch = []

    # Final partial chunk is dropped (standard PT practice)
    print(f"  Total examples processed: {total_examples:,}")
    print(f"  Total chunks (seq_length={seq_length}): {len(all_chunks):,}")
    print(f"  Leftover tokens dropped: {len(token_buffer):,}")

    return all_chunks


def _flush_batch(
    texts: list[str],
    tokenizer: transformers.PreTrainedTokenizer,
    eos_id: int,
    token_buffer: list[int],
    all_chunks: list[list[int]],
    seq_length: int,
):
    """Tokenize a batch of texts, append to buffer, and extract full chunks."""
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    for ids in encoded:
        token_buffer.extend(ids)
        if eos_id is not None and (not ids or ids[-1] != eos_id):
            token_buffer.append(eos_id)

    # Extract full chunks
    while len(token_buffer) >= seq_length:
        chunk = token_buffer[:seq_length]
        all_chunks.append(chunk)
        del token_buffer[:seq_length]


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
    parser.add_argument("--test_split_size", type=int, default=5000,
                        help="Number of chunks for the test split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=16,
                        help="Processes for saving to disk")
    args = parser.parse_args()

    print(f"Data dir:       {args.data_dir}")
    print(f"Tokenizer:      {args.tokenizer_path}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Op range:       {args.op_min}-{args.op_max}")
    print(f"Seq length:     {args.seq_length}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load tokenizer ---
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"Tokenizer vocab_size={tokenizer.vocab_size}, "
          f"eos={tokenizer.eos_token}({tokenizer.eos_token_id})")

    # --- Find shards ---
    shard_files = find_jsonl_shards(args.data_dir, args.op_min, args.op_max)
    print(f"Found {len(shard_files)} shard files\n")
    if not shard_files:
        print("ERROR: No JSONL shard files found!")
        sys.exit(1)

    # --- Tokenize + chunk ---
    chunks = tokenize_and_chunk_shards(
        shard_files, tokenizer, seq_length=args.seq_length,
    )

    # --- Build HF dataset ---
    print(f"\nBuilding HuggingFace dataset from {len(chunks):,} chunks...")
    dataset = Dataset.from_dict({
        "input_ids": chunks,
        "labels": [c[:] for c in chunks],
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
    ds_dict.save_to_disk(args.output_dir, num_proc=args.num_proc)

    # --- Verify ---
    print("\n--- Verification ---")
    sample = ds_dict["train"][0]
    print(f"  input_ids length: {len(sample['input_ids'])}")
    print(f"  labels length:    {len(sample['labels'])}")
    decoded = tokenizer.decode(sample["input_ids"][:100])
    print(f"  First 100 tokens decoded: {decoded[:200]}...")
    print(f"\nDone! Pre-tokenized dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
