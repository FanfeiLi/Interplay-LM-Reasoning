"""
Load GSM-Infinity *evaluation* results (transformer + diffusion) and plot
ID vs OOD pass@k trajectories across checkpoints.

This module is intended for outputs written under:
  - results/transformer_eval/<run_name>/checkpoint-*/checkpoint-*_metrics.json
  - results/dllm_eval/<run_name>/checkpoint-*/metrics.jsonl

We summarize pass@k by averaging per-op pass@k over these op ranges:
  - ID:       op=2..10
  - OOD-mid:  op=11..14
  - OOD-hard: op=17..20
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

try:  # optional, used only for nicer defaults
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

PASS_K_VALUES: list[int] = [1, 2, 4, 8, 16, 32, 64, 128]

DIFFICULTY_GROUPS: dict[str, list[int]] = {
    "ID (op=2-10)": list(range(2, 11)),
    "OOD-mid (op=11-14)": list(range(11, 15)),
    "OOD-hard (op=17-20)": list(range(17, 21)),
}


@dataclass(frozen=True)
class RunSpec:
    run_type: str  # "transformer" | "dllm"
    run_name: str
    label: Optional[str] = None

    @property
    def display_label(self) -> str:
        return self.label or self.run_name


def checkpoint_step(checkpoint_name: str) -> int:
    """Parse an integer step from 'checkpoint-XXXX' or map 'checkpoint-final' high."""
    name = (checkpoint_name or "").strip()
    if not name:
        return -1
    if "final" in name:
        return 999_999_999
    m = re.search(r"checkpoint-(\d+)", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1
    return -1


def _iter_checkpoint_dirs(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    ckpts = [p for p in run_dir.glob("checkpoint-*") if p.is_dir()]
    ckpts.sort(key=lambda p: checkpoint_step(p.name))
    return ckpts


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data)}")
    return data


def _load_metrics_jsonl(path: Path) -> dict[str, Any]:
    """Merge all JSONL records into a single metrics dict."""
    metrics: dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and isinstance(obj.get("metrics"), dict):
                metrics.update(obj["metrics"])
            elif isinstance(obj, dict):
                metrics.update(obj)
    return metrics


def _safe_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _extract_transformer_per_op_pass_at_k(
    metrics_json: dict[str, Any],
    *,
    op: int,
    k: int,
    split: str = "total",
) -> float:
    per_op = metrics_json.get(split, {}).get("per_op_pass_at_k", {})
    op_dict = per_op.get(str(op), {})
    key = f"pass@{k}"
    val = op_dict.get(key, None)
    if val is None:
        return float("nan")
    return float(val)


def _extract_dllm_per_op_pass_at_k(
    metrics: dict[str, Any],
    *,
    op: int,
    k: int,
) -> float:
    key = f"val-aux/difficulty-5B/{op}/reward/pass@{k}"
    val = metrics.get(key, None)
    if val is None:
        return float("nan")
    return float(val)


def load_eval_run(
    spec: RunSpec,
    *,
    k_values: Optional[list[int]] = None,
    difficulty_groups: Optional[dict[str, list[int]]] = None,
    to_percent: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a single eval run and return a tidy DataFrame with columns:
      run_type, run_name, model, checkpoint, step, group, k, pass_at_k
    """
    if k_values is None:
        k_values = PASS_K_VALUES
    if difficulty_groups is None:
        difficulty_groups = DIFFICULTY_GROUPS

    if spec.run_type not in {"transformer", "dllm"}:
        raise ValueError(f"Unknown run_type={spec.run_type!r} (expected 'transformer' or 'dllm')")

    base_dir = RESULTS_ROOT / ("transformer_eval" if spec.run_type == "transformer" else "dllm_eval") / spec.run_name
    ckpt_dirs = _iter_checkpoint_dirs(base_dir)
    if not ckpt_dirs and verbose:
        print(f"[warn] No checkpoints found under {base_dir}")

    rows: list[dict[str, Any]] = []
    for ckpt_dir in ckpt_dirs:
        ckpt = ckpt_dir.name
        step = checkpoint_step(ckpt)

        if spec.run_type == "transformer":
            metrics_path = ckpt_dir / f"{ckpt}_metrics.json"
            if not metrics_path.exists():
                # fallback: first *_metrics.json in the directory
                candidates = sorted(ckpt_dir.glob("*_metrics.json"))
                if candidates:
                    metrics_path = candidates[0]
            if not metrics_path.exists():
                if verbose:
                    print(f"[warn] Missing transformer metrics for {ckpt_dir}")
                continue
            metrics_json = _load_json(metrics_path)

            for group_name, ops in difficulty_groups.items():
                for k in k_values:
                    per_op_vals = [
                        _extract_transformer_per_op_pass_at_k(metrics_json, op=op, k=k, split="total")
                        for op in ops
                    ]
                    val = _safe_mean(per_op_vals)
                    if np.isnan(val):
                        continue
                    rows.append(
                        {
                            "run_type": spec.run_type,
                            "run_name": spec.run_name,
                            "model": spec.display_label,
                            "checkpoint": ckpt,
                            "step": step,
                            "group": group_name,
                            "k": int(k),
                            "pass_at_k": float(val * (100.0 if to_percent else 1.0)),
                        }
                    )
        else:
            metrics_path = ckpt_dir / "metrics.jsonl"
            if metrics_path.exists():
                metrics = _load_metrics_jsonl(metrics_path)
            else:
                # Incomplete evals sometimes have details_op*.json but no metrics.jsonl.
                # We keep this loader lightweight and simply skip those checkpoints.
                if verbose:
                    print(f"[warn] Missing dllm metrics.jsonl for {ckpt_dir} (skipping)")
                continue

            for group_name, ops in difficulty_groups.items():
                for k in k_values:
                    per_op_vals = [
                        _extract_dllm_per_op_pass_at_k(metrics, op=op, k=k)
                        for op in ops
                    ]
                    val = _safe_mean(per_op_vals)
                    if np.isnan(val):
                        continue
                    rows.append(
                        {
                            "run_type": spec.run_type,
                            "run_name": spec.run_name,
                            "model": spec.display_label,
                            "checkpoint": ckpt,
                            "step": step,
                            "group": group_name,
                            "k": int(k),
                            "pass_at_k": float(val * (100.0 if to_percent else 1.0)),
                        }
                    )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["model", "step", "k", "group"], kind="stable").reset_index(drop=True)
    return df


def load_eval_runs(
    specs: list[RunSpec] | list[dict[str, Any]],
    *,
    k_values: Optional[list[int]] = None,
    difficulty_groups: Optional[dict[str, list[int]]] = None,
    to_percent: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load multiple runs and concatenate."""
    run_specs: list[RunSpec] = []
    for s in specs:
        if isinstance(s, RunSpec):
            run_specs.append(s)
        elif isinstance(s, dict):
            run_specs.append(
                RunSpec(
                    run_type=str(s.get("run_type")),
                    run_name=str(s.get("run_name")),
                    label=(None if s.get("label") is None else str(s.get("label"))),
                )
            )
        else:
            raise TypeError(f"Unsupported spec type: {type(s)}")

    dfs = [
        load_eval_run(
            spec,
            k_values=k_values,
            difficulty_groups=difficulty_groups,
            to_percent=to_percent,
            verbose=verbose,
        )
        for spec in run_specs
    ]
    if not dfs:
        return pd.DataFrame()
    out = pd.concat([d for d in dfs if not d.empty], ignore_index=True) if any(not d.empty for d in dfs) else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["model", "step", "k", "group"], kind="stable").reset_index(drop=True)
    return out


def plot_id_vs_ood_scatter(
    df: pd.DataFrame,
    *,
    k_values: list[int] = [1, 16, 32, 64, 128],
    id_group: str = "ID (op=2-10)",
    ood_groups: list[str] = ["OOD-mid (op=11-14)", "OOD-hard (op=17-20)"],
    title_prefix: str = "",
    save_dir: Optional[Path] = None,
    dpi: int = 200,
    loglog: bool = True,
    value_floor: float = 1e-3,
    value_ceiling: Optional[float] = 100.0,
    fit_loglog: bool = True,
    fit_min_points: int = 3,
):
    """
    For each k in k_values, create a 1xN figure (one subplot per OOD group) showing:
      x = ID pass@k, y = OOD pass@k
    with one color per model:
      - scatter: checkpoints
      - line:    log-log linear fit (if enabled)
    """
    if df.empty:
        raise ValueError("Empty dataframe: nothing to plot")

    if sns is not None:
        sns.set_theme(style="ticks", context="talk")

    # Pivot to wide form: one row per (model, checkpoint, k)
    wide = (
        df.pivot_table(
            index=["model", "run_type", "run_name", "checkpoint", "step", "k"],
            columns="group",
            values="pass_at_k",
            aggfunc="mean",
        )
        .reset_index()
    )

    models = list(dict.fromkeys(wide["model"].tolist()))
    if sns is not None:
        palette_list = sns.color_palette("tab10", n_colors=max(3, len(models)))
    else:
        palette_list = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
    color_map = {m: palette_list[i % len(palette_list)] for i, m in enumerate(models)}

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    def _clip_positive(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").astype(float)
        if loglog:
            s = s.where(s > 0.0, np.nan)
            if value_floor is not None and value_floor > 0:
                s = s.clip(lower=float(value_floor))
        return s

    def _fit_loglog_line(xs: np.ndarray, ys: np.ndarray):
        if xs.size < fit_min_points or ys.size < fit_min_points:
            return None
        if not loglog:
            return None
        ok = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0) & (ys > 0.0)
        xs = xs[ok]
        ys = ys[ok]
        if xs.size < fit_min_points:
            return None
        lx = np.log10(xs)
        ly = np.log10(ys)
        m, b = np.polyfit(lx, ly, deg=1)
        return float(m), float(b)

    for k in k_values:
        wk = wide[wide["k"] == k].copy()
        if wk.empty:
            continue

        fig, axes = plt.subplots(1, len(ood_groups), figsize=(7 * len(ood_groups), 6), constrained_layout=True)
        if len(ood_groups) == 1:
            axes = [axes]

        short_id = id_group.split(" (", 1)[0].strip() if id_group else "ID"

        for ax, og in zip(axes, ood_groups):
            short_og = og.split(" (", 1)[0].strip() if og else "OOD"
            if loglog:
                ax.set_xscale("log")
                ax.set_yscale("log")

                # Log ticks: label 1-2-5 per decade, keep the rest as unlabeled minor ticks.
                ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
                ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
                ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0)))
                ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0)))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

            # Track bounds based on actually plotted values (per subplot)
            all_x: list[float] = []
            all_y: list[float] = []

            for m in models:
                sub = wk[wk["model"] == m].sort_values("step")
                if id_group not in sub.columns or og not in sub.columns:
                    continue
                xs = _clip_positive(sub[id_group])
                ys = _clip_positive(sub[og])
                ok = xs.notna() & ys.notna()
                xs = xs[ok].to_numpy(dtype=float)
                ys = ys[ok].to_numpy(dtype=float)
                if xs.size == 0:
                    continue

                all_x.extend(xs.tolist())
                all_y.extend(ys.tolist())

                # Scatter checkpoints
                ax.scatter(
                    xs,
                    ys,
                    color=color_map[m],
                    s=55,
                    alpha=0.9,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                    label=m,
                )

                # Best-fit line in log-log space: log10(y) = m*log10(x) + b
                if fit_loglog:
                    fit = _fit_loglog_line(xs, ys)
                    if fit is not None:
                        slope, intercept = fit
                        x_lo = float(np.nanmin(xs))
                        x_hi = float(np.nanmax(xs))
                        if loglog and x_lo > 0.0 and x_hi > 0.0 and x_hi > x_lo:
                            x_line = np.logspace(np.log10(x_lo), np.log10(x_hi), 100)
                        else:
                            x_line = np.linspace(x_lo, x_hi, 100)
                        y_line = 10.0 ** (slope * np.log10(x_line) + intercept)
                        ax.plot(
                            x_line,
                            y_line,
                            color=color_map[m],
                            linewidth=2.5,
                            alpha=0.9,
                            zorder=2,
                        )

            ax.set_xlabel(f"{short_id} pass@{k}")
            ax.set_ylabel(f"{short_og} pass@{k}")
            ax.set_title(f"{og}")

            # Set independent x/y ranges from plotted values
            if all_x and all_y:
                x_min = float(np.nanmin(all_x))
                x_max = float(np.nanmax(all_x))
                y_min = float(np.nanmin(all_y))
                y_max = float(np.nanmax(all_y))

                if loglog:
                    pad = 1.25
                    x_lo = max(value_floor, x_min / pad)
                    y_lo = max(value_floor, y_min / pad)
                    x_hi = x_max * pad
                    y_hi = y_max * pad
                    if value_ceiling is not None and value_ceiling > 0:
                        x_hi = min(float(value_ceiling), x_hi)
                        y_hi = min(float(value_ceiling), y_hi)
                    ax.set_xlim(x_lo, x_hi)
                    ax.set_ylim(y_lo, y_hi)
                else:
                    pad = 0.05
                    dx = max(1e-12, x_max - x_min)
                    dy = max(1e-12, y_max - y_min)
                    ax.set_xlim(x_min - pad * dx, x_max + pad * dx)
                    ax.set_ylim(y_min - pad * dy, y_max + pad * dy)

            ax.grid(True, which="both", alpha=0.25, linewidth=0.8)

        # Single shared legend (deduplicate)
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=min(4, len(by_label)))

        if title_prefix:
            fig.suptitle(f"{title_prefix} (pass@{k})", y=1.02)

        if save_dir is not None:
            out_path = save_dir / f"id_vs_ood_pass{k}.png"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

        plt.show()

