"""
Process GSM-Infinity RL experiment results for pass@k analysis.

Loads metrics from root metrics.jsonl for each run at the final training step,
computes average pass@k across difficulty ranges for three evaluation regimes:
  - ID (op=2-10)
  - OOD-mid / Edge (op=11-14)
  - OOD-hard (op=17-20)

Note: ops 15-16 are not evaluated; OOD-hard uses ops 17-20 only.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "gsm_infinity_rl"

PASS_K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]

# Difficulty groupings (ops actually evaluated)
DIFFICULTY_GROUPS = {
    "ID (op=2-10)": list(range(2, 11)),       # 2,3,...,10
    "OOD-mid (op=11-14)": list(range(11, 15)),  # 11,12,13,14
    "OOD-hard (op=17-20)": list(range(17, 21)), # 17,18,19,20
}

# All evaluated ops
ALL_OPS = sorted(
    DIFFICULTY_GROUPS["ID (op=2-10)"]
    + DIFFICULTY_GROUPS["OOD-mid (op=11-14)"]
    + DIFFICULTY_GROUPS["OOD-hard (op=17-20)"]
)

# Run directories
ALL_RUNS = [
    "base_model_eval_pass128",
    # DR-GSPO standard
    "dr_gspo_id",
    "dr_gspo_edge",
    "dr_gspo_mixed",
    "dr_gspo_hard",
    # DR-GSPO ClipCov
    "dr_gspo_clip_cov_id",
    "dr_gspo_clip_cov_edge",
    "dr_gspo_clip_cov_mixed",
    "dr_gspo_clip_cov_hard",
    # DR-GSPO MGPO
    "dr_gspo_mgpo_id",
    "dr_gspo_mgpo_edge",
    "dr_gspo_mgpo_mixed",
    "dr_gspo_mgpo_hard",
    # GRPO
    "grpo_id",
    "grpo_edge",
    "grpo_mixed",
    "grpo_hard",
]

# Nice display names
DISPLAY_NAMES = {
    "base_model_eval_pass128": "Base",
    # DR-GSPO standard
    "dr_gspo_id": "DR-GSPO (ID)",
    "dr_gspo_edge": "DR-GSPO (Edge)",
    "dr_gspo_mixed": "DR-GSPO (Mixed)",
    "dr_gspo_hard": "DR-GSPO (Hard)",
    # DR-GSPO ClipCov
    "dr_gspo_clip_cov_id": "DR-GSPO-ClipCov (ID)",
    "dr_gspo_clip_cov_edge": "DR-GSPO-ClipCov (Edge)",
    "dr_gspo_clip_cov_mixed": "DR-GSPO-ClipCov (Mixed)",
    "dr_gspo_clip_cov_hard": "DR-GSPO-ClipCov (Hard)",
    # DR-GSPO MGPO
    "dr_gspo_mgpo_id": "DR-GSPO-MGPO (ID)",
    "dr_gspo_mgpo_edge": "DR-GSPO-MGPO (Edge)",
    "dr_gspo_mgpo_mixed": "DR-GSPO-MGPO (Mixed)",
    "dr_gspo_mgpo_hard": "DR-GSPO-MGPO (Hard)",
    # GRPO
    "grpo_id": "GRPO (ID)",
    "grpo_edge": "GRPO (Edge)",
    "grpo_mixed": "GRPO (Mixed)",
    "grpo_hard": "GRPO (Hard)",
}

# Training data regimes
DATA_REGIMES = {
    "dr_gspo_id": "id",
    "dr_gspo_edge": "edge",
    "dr_gspo_mixed": "mixed",
    "dr_gspo_hard": "hard",
    "dr_gspo_clip_cov_id": "id",
    "dr_gspo_clip_cov_edge": "edge",
    "dr_gspo_clip_cov_mixed": "mixed",
    "dr_gspo_clip_cov_hard": "hard",
    "dr_gspo_mgpo_id": "id",
    "dr_gspo_mgpo_edge": "edge",
    "dr_gspo_mgpo_mixed": "mixed",
    "dr_gspo_mgpo_hard": "hard",
    "grpo_id": "id",
    "grpo_edge": "edge",
    "grpo_mixed": "mixed",
    "grpo_hard": "hard",
}

# Training data op ranges for labels
DATA_REGIME_OPS = {
    "id": "op=2-10",
    "edge": "op=11-14",
    "hard": "op=17-20",
    "mixed": "op=2-20 (mixed)",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_metrics_at_step(run_name: str, step: Optional[int] = None) -> dict:
    """
    Load validation metrics for a run at a given step.

    For base_model_eval_pass128, reads the first line of its metrics.jsonl.
    For RL runs, reads root metrics.jsonl and finds the entry at `step`.
    If step is None, uses the value from latest_checkpointed_iteration.txt.
    """
    run_dir = RESULTS_DIR / run_name

    if run_name == "base_model_eval_pass128":
        metrics_path = run_dir / "metrics.jsonl"
        with open(metrics_path) as f:
            data = json.loads(f.readline())
        return data["metrics"]

    # Determine step
    if step is None:
        iter_file = run_dir / "latest_checkpointed_iteration.txt"
        step = int(iter_file.read_text().strip())

    metrics_path = run_dir / "metrics.jsonl"
    with open(metrics_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("log_step") == step:
                return entry["metrics"]

    raise ValueError(f"Step {step} not found in {metrics_path}")


def _extract_pass_at_k(metrics: dict, op: int, k: int) -> float:
    """Extract pass@k value for a given op from metrics dict."""
    key = f"val-aux/difficulty-5B/{op}/reward/pass@{k}"
    if key not in metrics:
        raise KeyError(f"Key {key} not found in metrics")
    return metrics[key]


def get_pass_at_k_for_run(
    run_name: str,
    step: Optional[int] = None,
) -> dict[str, dict[int, float]]:
    """
    Get average pass@k for each difficulty group for a single run.

    Returns:
        dict mapping group_name -> {k: avg_pass_at_k} for each k in PASS_K_VALUES
    """
    metrics = _load_metrics_at_step(run_name, step)

    result = {}
    for group_name, ops in DIFFICULTY_GROUPS.items():
        pass_k_avg = {}
        for k in PASS_K_VALUES:
            values = [_extract_pass_at_k(metrics, op, k) for op in ops]
            pass_k_avg[k] = np.mean(values)
        result[group_name] = pass_k_avg

    return result


def get_all_runs_data(
    run_names: Optional[list[str]] = None,
    step: Optional[int] = None,
) -> dict[str, dict[str, dict[int, float]]]:
    """
    Load pass@k data for all specified runs.

    Returns:
        dict mapping run_name -> group_name -> {k: avg_pass_at_k}
    """
    if run_names is None:
        run_names = ALL_RUNS

    all_data = {}
    for run_name in run_names:
        try:
            all_data[run_name] = get_pass_at_k_for_run(run_name, step)
        except Exception as e:
            print(f"Warning: Failed to load {run_name}: {e}")

    return all_data


def data_to_dataframe(all_data: dict) -> pd.DataFrame:
    """
    Convert the nested dict into a tidy DataFrame for easier plotting.

    Columns: run, display_name, group, k, pass_at_k
    """
    rows = []
    for run_name, groups in all_data.items():
        for group_name, pass_k_dict in groups.items():
            for k, val in pass_k_dict.items():
                rows.append({
                    "run": run_name,
                    "display_name": DISPLAY_NAMES.get(run_name, run_name),
                    "group": group_name,
                    "k": k,
                    "pass_at_k": val * 100,  # convert to percentage
                })
    return pd.DataFrame(rows)


# ── Per-op pass@k (for detailed analysis) ─────────────────────────────────────

def get_per_op_pass_at_k(
    run_name: str,
    step: Optional[int] = None,
) -> pd.DataFrame:
    """
    Get per-op pass@k values (not averaged) for a single run.

    Returns DataFrame with columns: run, op, k, pass_at_k
    """
    metrics = _load_metrics_at_step(run_name, step)
    rows = []
    for op in ALL_OPS:
        for k in PASS_K_VALUES:
            val = _extract_pass_at_k(metrics, op, k)
            rows.append({
                "run": run_name,
                "display_name": DISPLAY_NAMES.get(run_name, run_name),
                "op": op,
                "k": k,
                "pass_at_k": val * 100,
            })
    return pd.DataFrame(rows)


# ── Convenience: quick summary table ──────────────────────────────────────────

def summary_table(
    run_names: Optional[list[str]] = None,
    ks: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Print a summary table of pass@1 and pass@128 for all groups and runs.
    """
    if ks is None:
        ks = [1, 128]
    data = get_all_runs_data(run_names)
    rows = []
    for run_name, groups in data.items():
        row = {"Run": DISPLAY_NAMES.get(run_name, run_name)}
        for group_name, pass_k_dict in groups.items():
            for k in ks:
                col = f"{group_name} pass@{k}"
                row[col] = f"{pass_k_dict[k]*100:.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick sanity check
    df = summary_table()
    print(df.to_string(index=False))




