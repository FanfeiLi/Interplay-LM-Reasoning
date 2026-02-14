#!/usr/bin/env python3
"""
Download and prepare GSM-Infinity composition data for pre-training.

This script downloads the composition dataset from HuggingFace
and organizes it into the format expected by LLaMA-Factory.

Dataset: Interplay-LM-Reasoning/composition
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import random

from datasets import load_dataset
from tqdm import tqdm


def download_and_prepare_composition_data(
    output_dir: str,
    op_range: tuple = (2, 10),
    templates: list = None,
    tokens_per_op: int = 1_111_111_111,  # ~1.1B tokens per op for 10B total across 9 ops
    seed: int = 42,
):
    """
    Download composition data and prepare it for pre-training.
    
    Args:
        output_dir: Directory to save the prepared data
        op_range: Tuple of (min_op, max_op) inclusive
        templates: List of templates to include (default: all)
        tokens_per_op: Approximate tokens to include per operation level
        seed: Random seed for shuffling
    """
    random.seed(seed)
    
    if templates is None:
        templates = ["crazy_zootopia", "teachers_in_school", "movie_festival_awards"]
    
    output_path = Path(output_dir)
    train_path = output_path / "composition_2" / "train"
    val_path = output_path / "composition_2" / "val"
    heldout_path = output_path / "composition_2" / "heldout"
    
    # Create directories
    for op in range(op_range[0], op_range[1] + 1):
        (train_path / str(op)).mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    heldout_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading composition data for ops {op_range[0]}-{op_range[1]}...")
    print(f"Templates: {templates}")
    print(f"Target tokens per op: {tokens_per_op:,}")
    
    # Load from HuggingFace: Interplay-LM-Reasoning/composition
    # Use streaming=True to avoid caching the entire dataset to disk
    print("\nLoading dataset from HuggingFace: Interplay-LM-Reasoning/composition (streaming mode)")
    dataset = load_dataset("Interplay-LM-Reasoning/composition", streaming=True)
    
    # Check available splits
    print(f"Available splits: {list(dataset.keys())}")
    
    # Use train split
    if "train" in dataset:
        train_data = dataset["train"]
    else:
        # If no train split, use the first available
        train_data = dataset[list(dataset.keys())[0]]
    
    print("Streaming dataset (no full load into memory/disk)")
    
    # Show sample to understand structure
    print("\nSample data structure:")
    sample = next(iter(train_data))
    for key in sample.keys():
        val = sample[key]
        if isinstance(val, str) and len(val) > 100:
            val = val[:100] + "..."
        print(f"  {key}: {val}")
    
    # Re-create iterator after consuming one item
    train_data = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]
    
    # Group by operation count - stream through the data
    data_by_op = defaultdict(list)
    
    # Calculate max examples needed per op to avoid loading everything
    tokens_per_example = 500
    max_examples_per_op = (tokens_per_op // tokens_per_example) + 60000  # +60k for val/heldout buffer
    
    print(f"\nStreaming and filtering data (max {max_examples_per_op:,} examples per op)...")
    op_counts = defaultdict(int)
    
    for example in tqdm(train_data, desc="Streaming data"):
        op = example.get("op", example.get("num_ops", 2))
        if op_range[0] <= op <= op_range[1]:
            # Skip if we already have enough for this op
            if op_counts[op] >= max_examples_per_op:
                continue
            template = example.get("template", "crazy_zootopia")
            if template in templates:
                data_by_op[op].append(dict(example))
                op_counts[op] += 1
        
        # Early exit if we have enough for all ops
        if all(op_counts.get(op, 0) >= max_examples_per_op for op in range(op_range[0], op_range[1] + 1)):
            print("\nCollected enough examples for all ops, stopping early.")
            break
    
    print(f"\nData distribution by op:")
    for op in sorted(data_by_op.keys()):
        print(f"  op={op}: {len(data_by_op[op])} examples")
    
    # Save to files
    for op in range(op_range[0], op_range[1] + 1):
        examples = data_by_op.get(op, [])
        if not examples:
            print(f"Warning: No examples found for op={op}")
            continue
        
        random.shuffle(examples)
        
        # Estimate tokens (rough: ~500 tokens per example on average)
        tokens_per_example = 500
        max_examples = tokens_per_op // tokens_per_example
        
        # Split: 90% train, 5% val, 5% heldout
        n_train = min(len(examples), int(max_examples * 0.9))
        n_val = min(200, len(examples) - n_train)
        n_heldout = min(50000, len(examples) - n_train - n_val)
        
        train_examples = examples[:n_train]
        val_examples = examples[n_train:n_train + n_val]
        heldout_examples = examples[n_train + n_val:n_train + n_val + n_heldout]
        
        # Save training data in chunks
        chunk_size = 100_000_000  # ~100M tokens per file
        examples_per_chunk = chunk_size // tokens_per_example
        
        for i, chunk_start in enumerate(range(0, len(train_examples), examples_per_chunk)):
            chunk = train_examples[chunk_start:chunk_start + examples_per_chunk]
            chunk_tokens = len(chunk) * tokens_per_example
            filename = f"op{op}_{chunk_tokens // 1_000_000}M.jsonl"
            filepath = train_path / str(op) / filename
            
            with open(filepath, "w") as f:
                for ex in chunk:
                    f.write(json.dumps(ex) + "\n")
            print(f"Saved {len(chunk)} examples to {filepath}")
        
        # Save validation data
        if val_examples:
            val_file = val_path / f"op{op}-{len(val_examples)}.jsonl"
            with open(val_file, "w") as f:
                for ex in val_examples:
                    f.write(json.dumps(ex) + "\n")
            print(f"Saved {len(val_examples)} val examples to {val_file}")
        
        # Save heldout data
        if heldout_examples:
            heldout_file = heldout_path / f"op{op}-{len(heldout_examples)}.jsonl"
            with open(heldout_file, "w") as f:
                for ex in heldout_examples:
                    f.write(json.dumps(ex) + "\n")
            print(f"Saved {len(heldout_examples)} heldout examples to {heldout_file}")




def main():
    parser = argparse.ArgumentParser(description="Download and prepare GSM-Infinity data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/data",
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--min-op",
        type=int,
        default=2,
        help="Minimum operation count",
    )
    parser.add_argument(
        "--max-op",
        type=int,
        default=10,
        help="Maximum operation count",
    )
    parser.add_argument(
        "--tokens-total",
        type=int,
        default=10_000_000_000,
        help="Total tokens to generate (default: 10B)",
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        default=["crazy_zootopia", "teachers_in_school", "movie_festival_awards"],
        help="Templates to include",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    n_ops = args.max_op - args.min_op + 1
    tokens_per_op = args.tokens_total // n_ops
    
    print("=" * 60)
    print("GSM-Infinity Data Preparation")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Operation range: {args.min_op}-{args.max_op}")
    print(f"Total tokens: {args.tokens_total:,}")
    print(f"Tokens per op: {tokens_per_op:,}")
    print(f"Templates: {args.templates}")
    print("=" * 60)
    
    download_and_prepare_composition_data(
        output_dir=args.output_dir,
        op_range=(args.min_op, args.max_op),
        templates=args.templates,
        tokens_per_op=tokens_per_op,
        seed=args.seed,
    )
    
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()

