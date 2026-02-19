#!/usr/bin/env python3
"""
Download and prepare GSM-Infinity composition data for RL finetuning.

This script downloads data from HuggingFace (Interplay-LM-Reasoning/composition)
and organizes it into 4 training sets:
1. ID (In-Distribution): op=2-10, 200K samples
2. Edge: op=11-14, 200K samples  
3. Hard: op=17-20, 200K samples
4. Mixed: ~67K samples from each of the above

Only downloads a small subset of shards (not the entire dataset) since we only need 200K samples.
Uses /tmp for HF cache to avoid Lustre flock issues.
"""

import os
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from collections import defaultdict
import random
from tqdm import tqdm


def get_shard_list(repo_id: str, op: int, temp_dir: str) -> list:
    """Get list of available shards for an op using huggingface-cli."""
    from huggingface_hub import HfApi
    api = HfApi()
    
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        shards = [f for f in files if f.startswith(f"train/{op}/") and f.endswith(".jsonl")]
        return sorted(shards)
    except Exception as e:
        print(f"  Warning: Could not list files for op={op}: {e}")
        return []


def download_and_prepare_rl_data(
    output_dir: str,
    samples_per_set: int = 200_000,
    seed: int = 42,
    templates: list = None,
    shards_per_op: int = 2,  # Only download this many shards per op
):
    """
    Download composition data and prepare it for RL finetuning.
    
    Args:
        output_dir: Directory to save the prepared data
        samples_per_set: Number of samples per training set (default: 200K)
        seed: Random seed for shuffling
        templates: List of templates to include (default: all)
        shards_per_op: Number of shards to download per op (default: 2)
    """
    random.seed(seed)
    
    if templates is None:
        templates = ["crazy_zootopia", "teachers_in_school", "movie_festival_awards"]
    
    output_path = Path(output_dir)
    
    # Define the 4 training sets
    train_sets = {
        "id": list(range(2, 11)),      # op=2-10 (ID)
        "edge": list(range(11, 15)),   # op=11-14 (edge)
        "hard": list(range(17, 21)),   # op=17-20 (hard)
    }
    
    # All ops we need to download
    all_ops = set()
    for ops in train_sets.values():
        all_ops.update(ops)
    
    # Create temp directory for downloads (avoids Lustre flock issues)
    temp_dir = tempfile.mkdtemp(prefix="hf_rl_download_")
    hf_cache = os.path.join(temp_dir, ".hf_cache")
    
    # Set HF environment variables to use temp dir
    os.environ["HF_HOME"] = hf_cache
    os.environ["HF_HUB_CACHE"] = hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
    
    print("=" * 60)
    print("GSM-Infinity RL Data Preparation (Minimal Download)")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Temp directory: {temp_dir}")
    print(f"Samples per set: {samples_per_set:,}")
    print(f"Shards per op: {shards_per_op} (downloading only what's needed)")
    print(f"Templates: {templates}")
    print(f"Ops to download: {sorted(all_ops)}")
    print("=" * 60)
    
    # Create output directories
    for set_name in list(train_sets.keys()) + ["mixed"]:
        (output_path / "train" / set_name).mkdir(parents=True, exist_ok=True)
    
    # Calculate how many samples we need per op
    # For main sets: samples_per_set / num_ops
    # For mixed: additional samples_per_set / 3 / num_ops
    # Total per op = main + mixed buffer
    samples_needed_per_op = {}
    for set_name, ops in train_sets.items():
        main_per_op = samples_per_set // len(ops)
        mixed_per_op = (samples_per_set // 3) // len(ops)
        for op in ops:
            samples_needed_per_op[op] = main_per_op + mixed_per_op + 5000  # buffer
    
    print(f"\nSamples needed per op: {samples_needed_per_op}")
    
    # Download data for each op - only a few shards
    repo_id = "Interplay-LM-Reasoning/composition"
    data_dir = os.path.join(temp_dir, "data")
    
    print("\nDownloading minimal data (only needed shards)...")
    
    for op in sorted(all_ops):
        print(f"\n=== Downloading op={op} (up to {shards_per_op} shards) ===")
        
        # Get list of available shards
        shards = get_shard_list(repo_id, op, temp_dir)
        if not shards:
            print(f"  No shards found, trying to download all...")
            # Fallback: try downloading with pattern
            shards_to_download = [f"train/{op}/*"]
        else:
            # Only download first N shards
            shards_to_download = shards[:shards_per_op]
            print(f"  Found {len(shards)} shards, downloading first {len(shards_to_download)}")
        
        for shard in shards_to_download:
            cmd = [
                "huggingface-cli", "download", repo_id,
                "--repo-type", "dataset",
                "--include", shard,
                "--local-dir", data_dir
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"  Warning: Failed to download {shard}: {e}")
        
        # Check what we got
        op_dir = Path(data_dir) / "train" / str(op)
        if op_dir.exists():
            files = list(op_dir.glob("*.jsonl"))
            print(f"  Downloaded {len(files)} files for op={op}")
    
    # Now process the downloaded files
    print("\n" + "=" * 60)
    print("Processing downloaded data...")
    print("=" * 60)
    
    # Collect data by op from downloaded files
    data_by_op = defaultdict(list)
    
    for op in sorted(all_ops):
        op_dir = Path(data_dir) / "train" / str(op)
        if not op_dir.exists():
            print(f"  Warning: No data found for op={op}")
            continue
        
        jsonl_files = sorted(op_dir.glob("*.jsonl"))
        print(f"\n  Processing op={op}: {len(jsonl_files)} files")
        
        needed = samples_needed_per_op.get(op, 50000)
        collected = 0
        
        for jsonl_file in jsonl_files:
            if collected >= needed:
                break
                
            with open(jsonl_file, "r") as f:
                for line in f:
                    if collected >= needed:
                        break
                    try:
                        example = json.loads(line.strip())
                        template = example.get("template", "crazy_zootopia")
                        if template in templates:
                            data_by_op[op].append(example)
                            collected += 1
                    except json.JSONDecodeError:
                        continue
        
        print(f"    Collected {len(data_by_op[op]):,} examples for op={op}")
    
    # Shuffle all data
    print("\nShuffling data...")
    for op in data_by_op:
        random.shuffle(data_by_op[op])
    
    # Create the 3 main training sets
    used_per_op = defaultdict(int)  # Track how many samples used per op
    
    for set_name, ops in train_sets.items():
        print(f"\nCreating {set_name} training set...")
        samples_per_op = samples_per_set // len(ops)
        
        all_examples = []
        for op in ops:
            available = len(data_by_op[op])
            to_take = min(samples_per_op, available)
            examples = data_by_op[op][:to_take]
            all_examples.extend(examples)
            used_per_op[op] = to_take
            print(f"  op={op}: {len(examples):,} examples (available: {available:,})")
        
        random.shuffle(all_examples)
        
        # Save to file
        output_file = output_path / "train" / set_name / f"{set_name}_200k.jsonl"
        with open(output_file, "w") as f:
            for ex in all_examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  Saved {len(all_examples):,} examples to {output_file}")
    
    # Create mixed training set (from remaining samples)
    print(f"\nCreating mixed training set...")
    mixed_examples = []
    samples_per_category = samples_per_set // 3
    
    for set_name, ops in train_sets.items():
        samples_per_op_mixed = samples_per_category // len(ops)
        for op in ops:
            start_idx = used_per_op[op]
            available = len(data_by_op[op]) - start_idx
            to_take = min(samples_per_op_mixed, available)
            examples = data_by_op[op][start_idx:start_idx + to_take]
            mixed_examples.extend(examples)
            print(f"  {set_name}/op={op}: {len(examples):,} examples")
    
    random.shuffle(mixed_examples)
    
    # Save mixed set
    output_file = output_path / "train" / "mixed" / "mixed_200k.jsonl"
    with open(output_file, "w") as f:
        for ex in mixed_examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved {len(mixed_examples):,} examples to {output_file}")
    
    # Cleanup temp directory
    print(f"\nCleaning up temp directory: {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\nTraining sets created in {output_path / 'train'}:")
    for set_name in ["id", "edge", "hard", "mixed"]:
        set_dir = output_path / "train" / set_name
        files = list(set_dir.glob("*.jsonl"))
        total = sum(sum(1 for _ in open(f)) for f in files)
        print(f"  {set_name}: {total:,} samples")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare GSM-Infinity RL data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/fast/pmayilvahanan/Interplay-LM-Reasoning/data/rl_finetune",
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--samples-per-set",
        type=int,
        default=200_000,
        help="Number of samples per training set (default: 200K)",
    )
    parser.add_argument(
        "--shards-per-op",
        type=int,
        default=2,
        help="Number of shards to download per op (default: 2)",
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
    
    download_and_prepare_rl_data(
        output_dir=args.output_dir,
        samples_per_set=args.samples_per_set,
        templates=args.templates,
        seed=args.seed,
        shards_per_op=args.shards_per_op,
    )


if __name__ == "__main__":
    main()
