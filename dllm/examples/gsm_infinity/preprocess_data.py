"""
Preprocess GSM-Infinity composition_hf JSONL data into a HuggingFace dataset
with a single "text" column suitable for dLLM pre-training.

Reads all JSONL shards for op 2-10, converts each example to text using
compose_text(), and saves as a HuggingFace Arrow dataset.

Usage:
    source /fast/pmayilvahanan/Interplay-LM-Reasoning/gsm_pretrain/bin/activate
    python dllm/examples/gsm_infinity/preprocess_data.py \
        --data_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/train \
        --output_dir /fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf_dllm \
        --op_min 2 --op_max 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

from datasets import Dataset, DatasetDict
from tqdm import tqdm


# ---------- Text composition (copied from utils/text_preprocess.py) ----------

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
    """Compose a training text string from a raw example.

    Format:
      <question> {problem} {question} </question> <solution> {solution_body} </solution> <answer> {answer} </answer>
    """
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


def process_shards(shard_files: list[str]) -> list[str]:
    """Read all shards and convert each line to a text string."""
    texts = []
    for shard_path in tqdm(shard_files, desc="Processing shards"):
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
                if text:
                    texts.append(text)
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess GSM-Infinity composition_hf data for dLLM pre-training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/train",
        help="Root directory containing op-level subdirs (2/, 3/, ..., 10/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf_dllm",
        help="Output directory for the HuggingFace Arrow dataset",
    )
    parser.add_argument("--op_min", type=int, default=2, help="Minimum op level")
    parser.add_argument("--op_max", type=int, default=10, help="Maximum op level")
    parser.add_argument(
        "--test_split_size",
        type=int,
        default=10000,
        help="Number of examples for the test split (0 to skip)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Op range: {args.op_min}-{args.op_max}")

    # Find shard files
    shard_files = find_jsonl_shards(args.data_dir, args.op_min, args.op_max)
    print(f"Found {len(shard_files)} shard files")

    if not shard_files:
        print("ERROR: No JSONL shard files found!")
        sys.exit(1)

    # Process all shards
    texts = process_shards(shard_files)
    print(f"Total examples: {len(texts):,}")

    # Create HuggingFace dataset
    dataset = Dataset.from_dict({"text": texts})
    dataset = dataset.shuffle(seed=args.seed)

    if args.test_split_size > 0 and len(texts) > args.test_split_size:
        ds_dict = dataset.train_test_split(
            test_size=args.test_split_size, seed=args.seed
        )
    else:
        ds_dict = DatasetDict({"train": dataset})

    print(f"Dataset splits: {dict((k, len(v)) for k, v in ds_dict.items())}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    ds_dict.save_to_disk(args.output_dir)
    print(f"Saved to {args.output_dir}")

    # Print sample
    print("\n--- Sample texts ---")
    for i in range(min(3, len(ds_dict["train"]))):
        text = ds_dict["train"][i]["text"]
        print(f"\n[{i}] ({len(text)} chars): {text[:200]}...")


if __name__ == "__main__":
    main()

