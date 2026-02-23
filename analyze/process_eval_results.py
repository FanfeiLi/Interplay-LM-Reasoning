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
import matplotlib.ticker as mticker
from scipy.special import ndtri as _probit_raw  # Φ⁻¹
from scipy.special import ndtr as _probit_cdf    # Φ


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


def _probit(p: np.ndarray | float) -> np.ndarray | float:
    """Probit transform Φ⁻¹(p) with safe clipping away from 0 and 1."""
    EPS = 1e-6
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, EPS, 1.0 - EPS)
    return _probit_raw(arr)


def _probit_inv(z: np.ndarray | float) -> np.ndarray | float:
    """Inverse probit: Φ(z), maps real line back to (0,1)."""
    return _probit_cdf(np.asarray(z, dtype=float))


# ── Probit-scale axis helpers ────────────────────────────────────────────────

class _ProbitScale:
    """Lightweight pair of (forward, inverse) for matplotlib FuncScale."""
    @staticmethod
    def forward(a: np.ndarray) -> np.ndarray:
        return _probit(a)

    @staticmethod
    def inverse(a: np.ndarray) -> np.ndarray:
        return _probit_inv(a)


def _setup_probit_axis(
    ax: plt.Axes,
    *,
    which: str = "both",
    pct: bool = True,
):
    """
    Apply probit (Φ⁻¹) scale to the given axes.

    If *pct* is True the data is in [0, 100] (percentages); we divide by 100
    before Φ⁻¹ and label ticks as percentages.  Otherwise data is in [0, 1].
    """
    divisor = 100.0 if pct else 1.0

    def _fwd(a):
        return _ProbitScale.forward(np.asarray(a, dtype=float) / divisor)

    def _inv(a):
        return _ProbitScale.inverse(np.asarray(a, dtype=float)) * divisor

    if which in ("both", "x"):
        ax.set_xscale("function", functions=(_fwd, _inv))
    if which in ("both", "y"):
        ax.set_yscale("function", functions=(_fwd, _inv))


def _probit_ticks(
    lo_pct: float,
    hi_pct: float,
    *,
    max_ticks: int = 12,
) -> np.ndarray:
    """Choose nice percentage tick values that span [lo_pct, hi_pct]."""
    candidates = np.array(
        [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 98, 99],
        dtype=float,
    )
    inside = candidates[(candidates >= lo_pct) & (candidates <= hi_pct)]
    if inside.size == 0:
        return np.array([lo_pct, hi_pct])
    if inside.size > max_ticks:
        step = max(1, inside.size // max_ticks)
        inside = inside[::step]
    return inside


# ── Main plotting function ───────────────────────────────────────────────────

def plot_id_vs_ood_scatter(
    df: pd.DataFrame,
    *,
    k_values: list[int] = [1, 16, 32, 64, 128],
    id_group: str = "ID (op=2-10)",
    ood_groups: list[str] = ["OOD-mid (op=11-14)", "OOD-hard (op=17-20)"],
    title_prefix: str = "",
    save_dir: Optional[Path] = None,
    dpi: int = 200,
    probit: bool = True,
    fit: bool = True,
    fit_min_points: int = 3,
    pct: bool = True,
):
    """
    For each k in k_values, create a 1xN figure (one subplot per OOD group):
      x = ID pass@k, y = OOD pass@k

    When ``probit=True`` both axes use the probit transform Φ⁻¹ so that a
    universal linear relation (Miller et al., "Accuracy on the Line") appears
    as a straight line.  The fit line is an OLS in probit space.

    Parameters
    ----------
    pct : bool
        True if pass_at_k values are percentages (0-100); False if fractions (0-1).
    """
    if df.empty:
        raise ValueError("Empty dataframe: nothing to plot")

    if sns is not None:
        sns.set_theme(style="ticks", context="talk")

    divisor = 100.0 if pct else 1.0
    EPS_PCT = 0.01  # smallest plottable percentage (~0.01%)

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

    def _clean(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").astype(float)
        s = s.clip(lower=EPS_PCT, upper=divisor * (1.0 - 1e-6))
        return s

    def _fit_probit_line(xs_pct: np.ndarray, ys_pct: np.ndarray):
        ok = np.isfinite(xs_pct) & np.isfinite(ys_pct)
        xs_pct = xs_pct[ok]
        ys_pct = ys_pct[ok]
        if xs_pct.size < fit_min_points:
            return None
        px = _probit(xs_pct / divisor)
        py = _probit(ys_pct / divisor)
        ok2 = np.isfinite(px) & np.isfinite(py)
        px = px[ok2]
        py = py[ok2]
        if px.size < fit_min_points:
            return None
        m, b = np.polyfit(px, py, deg=1)
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

            if probit:
                _setup_probit_axis(ax, which="both", pct=pct)

            all_x: list[float] = []
            all_y: list[float] = []

            for m in models:
                sub = wk[wk["model"] == m].sort_values("step")
                if id_group not in sub.columns or og not in sub.columns:
                    continue
                xs = _clean(sub[id_group])
                ys = _clean(sub[og])
                ok = xs.notna() & ys.notna()
                xs = xs[ok].to_numpy(dtype=float)
                ys = ys[ok].to_numpy(dtype=float)
                if xs.size == 0:
                    continue

                all_x.extend(xs.tolist())
                all_y.extend(ys.tolist())

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

                if fit:
                    result = _fit_probit_line(xs, ys)
                    if result is not None:
                        slope, intercept = result
                        px_lo = _probit(float(np.nanmin(xs)) / divisor)
                        px_hi = _probit(float(np.nanmax(xs)) / divisor)
                        if not (np.isfinite(px_lo) and np.isfinite(px_hi)):
                            continue
                        pz = np.linspace(px_lo, px_hi, 200)
                        x_line_pct = _probit_inv(pz) * divisor
                        y_line_pct = _probit_inv(slope * pz + intercept) * divisor
                        ax.plot(
                            x_line_pct,
                            y_line_pct,
                            color=color_map[m],
                            linewidth=2.5,
                            alpha=0.9,
                            zorder=2,
                        )

            ax.set_xlabel(f"{short_id} pass@{k} (%)")
            ax.set_ylabel(f"{short_og} pass@{k} (%)")
            ax.set_title(og)

            if all_x and all_y:
                x_min, x_max = float(np.nanmin(all_x)), float(np.nanmax(all_x))
                y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))
                if probit:
                    pad_pct = 2.0
                    ax.set_xlim(max(EPS_PCT, x_min - pad_pct), min(divisor - EPS_PCT, x_max + pad_pct))
                    ax.set_ylim(max(EPS_PCT, y_min - pad_pct), min(divisor - EPS_PCT, y_max + pad_pct))

                    xticks = _probit_ticks(x_min - pad_pct, x_max + pad_pct)
                    yticks = _probit_ticks(y_min - pad_pct, y_max + pad_pct)
                    ax.set_xticks(xticks)
                    ax.set_yticks(yticks)
                    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
                else:
                    pad = 0.05
                    dx, dy = max(1e-12, x_max - x_min), max(1e-12, y_max - y_min)
                    ax.set_xlim(x_min - pad * dx, x_max + pad * dx)
                    ax.set_ylim(y_min - pad * dy, y_max + pad * dy)

            ax.grid(True, which="major", alpha=0.3, linewidth=0.8)

        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=min(4, len(by_label)))

        if title_prefix:
            fig.suptitle(f"{title_prefix} (pass@{k})", y=1.02)

        if save_dir is not None:
            out_path = save_dir / f"id_vs_ood_pass{k}.png"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

        plt.show()

