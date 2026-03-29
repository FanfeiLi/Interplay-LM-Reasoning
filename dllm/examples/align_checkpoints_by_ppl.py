"""
Align MDLM and AR checkpoints by perplexity for fair comparison.

Reads PPL result JSONs produced by:
  - eval_ppl_checkpoints.py  (MDLM VLB — Round 1)
  - eval_duel_ppl_checkpoints.py  (MDLM DUEL exact — Round 2)
  - eval_ppl_ar.py  (AR cross-entropy — exact)

Finds best-matching checkpoint pairs at each model size (100M, 200M, 400M).

Usage:
    # Round 1: align using VLB (fast sweep)
    python examples/align_checkpoints_by_ppl.py \
        --results_dir /path/to/ppl_results

    # Round 2: align using exact DUEL numbers (prefer over VLB when available)
    python examples/align_checkpoints_by_ppl.py \
        --results_dir /path/to/ppl_results --use_duel

    # Explicit files:
    python examples/align_checkpoints_by_ppl.py \
        --mdlm_files mdlm_100M.json mdlm_200M.json \
        --ar_files ar_100M.json ar_200M.json
"""

import argparse
import glob
import json
import os
import re
import sys


def load_results(json_path: str) -> list[dict]:
    with open(json_path) as f:
        return json.load(f)


def extract_step(checkpoint_path: str) -> int | str:
    m = re.search(r"checkpoint-(\d+)", checkpoint_path)
    if m:
        return int(m.group(1))
    if "checkpoint-final" in checkpoint_path:
        return "final"
    return os.path.basename(checkpoint_path)


def extract_size(filename: str) -> str | None:
    """Extract model size like '100M', '200M', '400M' from filename."""
    m = re.search(r"(\d+M)", filename)
    return m.group(1) if m else None


def merge_duel_into_vlb(vlb_results: list[dict], duel_results: list[dict]) -> list[dict]:
    """Replace VLB entries with DUEL entries where available (matched by checkpoint path)."""
    duel_by_ckpt = {r["checkpoint"]: r for r in duel_results}
    merged = []
    for r in vlb_results:
        if r["checkpoint"] in duel_by_ckpt:
            entry = dict(duel_by_ckpt[r["checkpoint"]])
            entry["source"] = "duel"
            merged.append(entry)
        else:
            entry = dict(r)
            entry["source"] = "vlb"
            merged.append(entry)
    # Add any DUEL-only checkpoints not in VLB
    vlb_ckpts = {r["checkpoint"] for r in vlb_results}
    for r in duel_results:
        if r["checkpoint"] not in vlb_ckpts:
            entry = dict(r)
            entry["source"] = "duel"
            merged.append(entry)
    return merged


def find_best_alignment(
    mdlm_results: list[dict],
    ar_results: list[dict],
) -> list[dict]:
    """
    For each MDLM checkpoint, find the AR checkpoint with closest PPL.
    Returns list of aligned pairs sorted by MDLM step.
    """
    alignments = []

    for md in mdlm_results:
        md_ppl = md["ppl"]
        md_step = extract_step(md["checkpoint"])

        best_ar = None
        best_diff = float("inf")
        for ar in ar_results:
            diff = abs(ar["ppl"] - md_ppl)
            if diff < best_diff:
                best_diff = diff
                best_ar = ar

        if best_ar is not None:
            alignments.append({
                "mdlm_checkpoint": md["checkpoint"],
                "mdlm_step": md_step,
                "mdlm_ppl": md_ppl,
                "mdlm_nll": md["nll"],
                "mdlm_source": md.get("source", "vlb"),
                "ar_checkpoint": best_ar["checkpoint"],
                "ar_step": extract_step(best_ar["checkpoint"]),
                "ar_ppl": best_ar["ppl"],
                "ar_nll": best_ar["nll"],
                "ppl_diff": best_diff,
                "ppl_diff_pct": 100 * best_diff / md_ppl if md_ppl > 0 else 0,
            })

    alignments.sort(key=lambda x: x["mdlm_step"] if isinstance(x["mdlm_step"], int) else 999999)
    return alignments


def print_run_summary(name: str, results: list[dict]):
    """Print PPL curve for a single run."""
    print(f"\n  {name}")
    print(f"  {'Step':>8}  {'NLL':>8}  {'PPL':>10}")
    print(f"  {'-'*30}")
    sorted_results = sorted(
        results,
        key=lambda r: extract_step(r["checkpoint"]) if isinstance(extract_step(r["checkpoint"]), int) else 999999,
    )
    for r in sorted_results:
        step = extract_step(r["checkpoint"])
        src = f" ({r['source']})" if "source" in r else ""
        print(f"  {str(step):>8}  {r['nll']:>8.4f}  {r['ppl']:>10.2f}{src}")
    best = min(results, key=lambda r: r["ppl"])
    print(f"  Best: step={extract_step(best['checkpoint'])} PPL={best['ppl']:.2f}")


def print_alignment_table(size: str, alignments: list[dict]):
    """Print alignment table for one model size."""
    print(f"\n{'='*80}")
    print(f"  Alignment: {size}")
    print(f"{'='*80}")
    print(
        f"  {'MDLM step':>10}  {'MDLM PPL':>10}  {'src':>5}  "
        f"{'AR step':>10}  {'AR PPL':>10}  {'Δ PPL':>8}  {'Δ%':>6}"
    )
    print(f"  {'-'*70}")
    for a in alignments:
        print(
            f"  {str(a['mdlm_step']):>10}  {a['mdlm_ppl']:>10.2f}  "
            f"{a['mdlm_source']:>5}  "
            f"{str(a['ar_step']):>10}  {a['ar_ppl']:>10.2f}  "
            f"{a['ppl_diff']:>8.2f}  {a['ppl_diff_pct']:>5.1f}%"
        )


def find_best_pair(alignments: list[dict]) -> dict | None:
    """Find the alignment pair with smallest PPL difference (best match)."""
    if not alignments:
        return None
    return min(alignments, key=lambda a: a["ppl_diff"])


def main():
    parser = argparse.ArgumentParser(
        description="Align MDLM and AR checkpoints by perplexity."
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Directory containing *_vlb_ppl.json, *_duel_ppl.json, and *_ar_ppl.json files.",
    )
    parser.add_argument(
        "--mdlm_files", nargs="*", default=None,
        help="Explicit MDLM result JSON files (VLB or DUEL).",
    )
    parser.add_argument(
        "--ar_files", nargs="*", default=None,
        help="Explicit AR result JSON files.",
    )
    parser.add_argument(
        "--use_duel", action="store_true",
        help="Prefer DUEL exact PPL over VLB where available.",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Save alignment results to JSON.",
    )
    parser.add_argument(
        "--print_round2_env", type=str, default=None,
        help="Write ROUND2_* env var exports to this file (for automated Round 2).",
    )
    args = parser.parse_args()

    # Discover result files
    mdlm_vlb_files = args.mdlm_files or []
    mdlm_duel_files = []
    ar_files = args.ar_files or []

    if args.results_dir:
        if not mdlm_vlb_files:
            mdlm_vlb_files = sorted(glob.glob(os.path.join(args.results_dir, "*mdlm*_vlb_ppl.json")))
        if args.use_duel:
            mdlm_duel_files = sorted(glob.glob(os.path.join(args.results_dir, "*mdlm*_duel_ppl.json")))
        if not ar_files:
            ar_files = sorted(glob.glob(os.path.join(args.results_dir, "*ar_*_ar_ppl.json")))

    if not mdlm_vlb_files and not ar_files:
        sys.exit("No result files found. Provide --results_dir or --mdlm_files/--ar_files.")

    ppl_source = "DUEL (exact) where available, VLB otherwise" if args.use_duel else "VLB (fast)"
    print("=" * 80)
    print(f"  MDLM vs AR Perplexity Comparison  [{ppl_source}]")
    print("=" * 80)

    # Group VLB results by size
    mdlm_vlb_by_size: dict[str, list[dict]] = {}
    for f in mdlm_vlb_files:
        size = extract_size(os.path.basename(f))
        if size:
            mdlm_vlb_by_size[size] = load_results(f)

    # Group DUEL results by size
    mdlm_duel_by_size: dict[str, list[dict]] = {}
    for f in mdlm_duel_files:
        size = extract_size(os.path.basename(f))
        if size:
            mdlm_duel_by_size[size] = load_results(f)

    # Merge: prefer DUEL over VLB when --use_duel
    mdlm_by_size: dict[str, list[dict]] = {}
    for size, vlb_results in mdlm_vlb_by_size.items():
        if args.use_duel and size in mdlm_duel_by_size:
            mdlm_by_size[size] = merge_duel_into_vlb(vlb_results, mdlm_duel_by_size[size])
        else:
            mdlm_by_size[size] = [dict(r, source="vlb") for r in vlb_results]

    for size in sorted(mdlm_by_size):
        duel_count = sum(1 for r in mdlm_by_size[size] if r.get("source") == "duel")
        vlb_count = len(mdlm_by_size[size]) - duel_count
        label = f"MDLM {size}"
        if duel_count > 0:
            label += f" ({duel_count} duel, {vlb_count} vlb)"
        print_run_summary(label, mdlm_by_size[size])

    ar_by_size: dict[str, list[dict]] = {}
    for f in ar_files:
        size = extract_size(os.path.basename(f))
        if size:
            ar_by_size[size] = load_results(f)
            print_run_summary(f"AR {size}", ar_by_size[size])

    # Align per size
    sizes = sorted(set(mdlm_by_size.keys()) & set(ar_by_size.keys()))
    all_best_pairs = []

    for size in sizes:
        alignments = find_best_alignment(mdlm_by_size[size], ar_by_size[size])
        print_alignment_table(size, alignments)

        best = find_best_pair(alignments)
        if best:
            all_best_pairs.append({"size": size, **best})

    # Summary of best pairs
    print(f"\n{'='*80}")
    print("  BEST MATCHING PAIRS (closest PPL)")
    print(f"{'='*80}")
    print(
        f"  {'Size':>6}  {'MDLM step':>10}  {'MDLM PPL':>10}  {'src':>5}  "
        f"{'AR step':>10}  {'AR PPL':>10}  {'Δ%':>6}"
    )
    print(f"  {'-'*66}")
    for p in all_best_pairs:
        print(
            f"  {p['size']:>6}  {str(p['mdlm_step']):>10}  {p['mdlm_ppl']:>10.2f}  "
            f"{p['mdlm_source']:>5}  "
            f"  {str(p['ar_step']):>10}  {p['ar_ppl']:>10.2f}  {p['ppl_diff_pct']:>5.1f}%"
        )

    # Print checkpoint paths for easy copy-paste
    print(f"\n  Checkpoint paths for best pairs:")
    for p in all_best_pairs:
        print(f"  {p['size']}:")
        print(f"    MDLM: {p['mdlm_checkpoint']}")
        print(f"    AR:   {p['ar_checkpoint']}")

    # Build ROUND2 env vars (best match + neighbors)
    round2_envs = {}
    for p in all_best_pairs:
        step = p["mdlm_step"]
        ckpt_dir = os.path.dirname(p["mdlm_checkpoint"])
        if isinstance(step, int):
            neighbors = []
            for delta in [-100, 0, 100]:
                candidate = os.path.join(ckpt_dir, f"checkpoint-{step + delta}")
                if os.path.isdir(candidate):
                    neighbors.append(candidate)
            if not neighbors:
                neighbors = [p["mdlm_checkpoint"]]
            round2_envs[p["size"]] = ",".join(neighbors)
        else:
            round2_envs[p["size"]] = p["mdlm_checkpoint"]

    if not args.use_duel:
        print(f"\n  For Round 2 DUEL, export these (best match + neighbors):")
        for size_key, val in round2_envs.items():
            print(f"  export ROUND2_{size_key}=\"{val}\"")

    # Write env file for automated Round 2
    if args.print_round2_env:
        with open(args.print_round2_env, "w") as ef:
            for size_key, val in round2_envs.items():
                ef.write(f"export ROUND2_{size_key}=\"{val}\"\n")
        print(f"\n  Round 2 env vars written to {args.print_round2_env}")

    # Save if requested
    if args.output_file:
        output = {
            "ppl_source": "duel+vlb" if args.use_duel else "vlb",
            "per_size": {},
            "best_pairs": all_best_pairs,
        }
        for size in sizes:
            output["per_size"][size] = find_best_alignment(
                mdlm_by_size[size], ar_by_size[size]
            )
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
