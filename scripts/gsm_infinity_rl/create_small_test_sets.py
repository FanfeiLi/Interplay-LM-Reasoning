#!/usr/bin/env python3
"""Create smaller test sets (200 samples per op level) for fast pass@128 evaluation.

The full test sets have 1000 samples per op level.
With pass@128, that's 1000 x 128 = 128K generations per op level.
200 samples per op x 128 rollouts = 25.6K generations per op level -- ~5x faster.
"""

import json
import os
import random

random.seed(42)

SRC_DIR = "/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/test"
DST_DIR = "/fast/pmayilvahanan/Interplay-LM-Reasoning/data/composition_hf/test_small"
SAMPLES_PER_OP = 200

os.makedirs(DST_DIR, exist_ok=True)

for fname in sorted(os.listdir(SRC_DIR)):
    if not fname.endswith(".jsonl"):
        continue
    src_path = os.path.join(SRC_DIR, fname)
    with open(src_path) as f:
        lines = f.readlines()
    random.shuffle(lines)
    selected = lines[:SAMPLES_PER_OP]
    out_name = fname.replace("-1k.jsonl", f"-{SAMPLES_PER_OP}.jsonl")
    dst_path = os.path.join(DST_DIR, out_name)
    with open(dst_path, "w") as f:
        f.writelines(selected)
    print(f"{fname}: {len(lines)} -> {len(selected)} samples -> {out_name}")

print(f"\nDone! Small test sets saved to {DST_DIR}")
print(f"Total: {len(os.listdir(DST_DIR))} files x {SAMPLES_PER_OP} samples = {len(os.listdir(DST_DIR)) * SAMPLES_PER_OP} prompts")
print(f"With pass@128: {len(os.listdir(DST_DIR)) * SAMPLES_PER_OP * 128:,} total generations")

