"""Shared helpers for retrieval experiments (EXP-3a, EXP-3b, EXP-3c).

Each retrieval experiment ranks a pool of candidates against one or more
queries by ascending distance, then evaluates how early in the ranking the
"active" molecules appear. They share metric definitions, the raw-CSV
schema, and most plot styles; what differs is query/pool construction and
which metrics are reported.

This module owns the shared bits. Experiments compose them in their own
`run()` and `make_plots()` methods.

Raw retrieval CSV schema (`RETRIEVAL_COLS`):
    seed, query_id, active_id, rank, distance, n_total, n_actives_in_pool
"seed" is interpreted by the experiment (scaffold seed, random query draw,
target index, ...); it just has to be a stable group key for aggregation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import result_dir


RETRIEVAL_COLS = (
    "seed", "query_id", "active_id", "rank", "distance",
    "n_total", "n_actives_in_pool",
)


# ---------------------------------------------------------------------------
# Metric helpers — all operate on the compact active-rank representation.
# active_ranks: list/array of 1-indexed ranks of the active molecules in the
#               sorted pool (lower = closer to query).
# n_total:      pool size (excluding the query itself).
# n_actives:    number of actives in the pool (excluding the query).
# ---------------------------------------------------------------------------

def logauc(active_ranks, n_total: int, n_actives: int, min_fpr: float = 1e-3) -> float:
    """Log-AUC over [min_fpr, 1.0] (Truchon & Bayly 2007)."""
    ni = n_total - n_actives
    if n_actives == 0 or ni <= 0:
        return float("nan")

    ranks = np.sort(np.asarray(active_ranks, dtype=float))
    k = len(ranks)

    # After the i-th active (0-indexed): TP = i+1, FP = ranks[i] - (i+1).
    tp = np.arange(1, k + 1, dtype=float)
    fpr_after = (ranks - tp) / ni
    tpr_after = tp / n_actives

    fpr_pts = np.concatenate([[0.0], fpr_after, [1.0]])
    tpr_pts = np.concatenate([[0.0], tpr_after, [1.0]])

    log_range = math.log10(1.0 / min_fpr)
    total = 0.0
    for i in range(k + 1):
        t = float(tpr_pts[i])
        if t == 0.0:
            continue
        fl = max(float(fpr_pts[i]), min_fpr)
        fr = min(float(fpr_pts[i + 1]), 1.0)
        if fr <= fl:
            continue
        total += t * (math.log10(fr) - math.log10(fl)) / log_range
    return total


def bedroc(active_ranks, n_total: int, n_actives: int, alpha: float = 20.0) -> float:
    """BEDROC_α (Truchon & Bayly 2007)."""
    if n_actives == 0 or n_total == 0:
        return float("nan")
    N = n_total
    Ra = n_actives / N
    ranks = np.asarray(active_ranks, dtype=float)

    S = float(np.sum(np.exp(-alpha * ranks / N)))
    C = (math.exp(alpha / N) - 1.0) / (Ra * (1.0 - math.exp(-alpha)))
    rie = S * C
    rie_max = (1.0 - math.exp(-alpha * Ra)) / (Ra * (1.0 - math.exp(-alpha)))
    rie_min = math.exp(-alpha * (1.0 - Ra)) * rie_max
    if rie_max == rie_min:
        return float("nan")
    return (rie - rie_min) / (rie_max - rie_min)


def ef_at_percent(active_ranks, n_total: int, n_actives: int, percent: float = 1.0) -> float:
    """Enrichment factor at `percent`% of the ranked list."""
    if n_actives == 0:
        return float("nan")
    cutoff = math.ceil(n_total * percent / 100.0)
    tp = sum(1 for r in active_ranks if r <= cutoff)
    return tp / (n_actives * percent / 100.0)


def dcg_at_k(active_ranks, k: int = 100) -> float:
    """Discounted Cumulative Gain at rank k (binary relevance)."""
    return sum(1.0 / math.log2(r + 1) for r in active_ranks if r <= k)


# ---------------------------------------------------------------------------
# Aggregation helpers — silent over NaN so test-mode "no actives in this
# split" cases don't drown the log in numpy warnings.
# ---------------------------------------------------------------------------

def nanmean(values: list[float]) -> float:
    finite = [x for x in values if not math.isnan(x)]
    return float(np.mean(finite)) if finite else float("nan")


def mean_se(values: list[float]) -> tuple[float, float]:
    v = [x for x in values if not math.isnan(x)]
    if not v:
        return float("nan"), float("nan")
    m = float(np.mean(v))
    se = float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    return m, se


def read_csv_or_empty(p: Path) -> pd.DataFrame:
    """Read a CSV, returning an empty DataFrame on zero-byte files."""
    if p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


def metric_entry(key: str, values: list[float]) -> dict:
    """Pack a mean ± SE pair into a dict suitable for metrics.json."""
    m, se = mean_se(values)
    return {key: m, f"{key}_se": se}


def metrics_to_json(metrics: dict) -> str:
    """Render a metrics dict, converting NaN floats to None for valid JSON."""
    return json.dumps(
        {k: (None if isinstance(v, float) and math.isnan(v) else v)
         for k, v in metrics.items()},
        indent=2,
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# Plotting helpers. Each takes the experiment id (for output paths and
# titles) plus the list of dataset ids it spans; they read raw CSVs from
# `result_dir(exp_id, method_id, dataset_id)` and write a single PNG.
# ---------------------------------------------------------------------------

def _grid_shape(n: int) -> tuple[int, int]:
    """Roughly square subplot grid for n datasets."""
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)
    return n_rows, n_cols


def _per_dataset_grid(n_datasets: int, cell_size: tuple[float, float] = (5.0, 4.0)):
    n_rows, n_cols = _grid_shape(n_datasets)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(cell_size[0] * n_cols, cell_size[1] * n_rows),
    )
    axes_flat = np.atleast_1d(axes).flatten()
    return fig, axes_flat


# Shared log-FPR grid for enrichment-curve summaries. Cached npz files are
# keyed by this grid, so changing it invalidates every cached summary —
# rebuild by deleting the npz files (or just letting them be regenerated
# from the source CSV by `_load_or_build_curve`).
LOG_FPR_GRID = np.linspace(-3, 0, 200)
ENRICHMENT_CURVE_NPZ = "enrichment_curve.npz"


def _build_curve_arrays(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Aggregate the per-(seed, query) interp into the shared LOG_FPR_GRID.

    Returns (tpr_avg, n_curves). When n_curves == 0 the TPR array is zeros
    and callers should treat the summary as empty.
    """
    tpr_grid = np.zeros(LOG_FPR_GRID.size)
    n_curves = 0
    fprs = 10 ** LOG_FPR_GRID
    for (_seed, _q), grp in df.groupby(["seed", "query_id"]):
        n_total = grp["n_total"].iloc[0]
        n_actives = grp["n_actives_in_pool"].iloc[0]
        ni = n_total - n_actives
        if ni <= 0 or n_actives == 0:
            continue
        ranks = np.sort(grp["rank"].values.astype(float))
        tp = np.arange(1, len(ranks) + 1, dtype=float)
        fpr_pts = np.concatenate([[0.0], (ranks - tp) / ni, [1.0]])
        tpr_pts = np.concatenate([[0.0], tp / n_actives, [1.0]])
        tpr_grid += np.interp(fprs, fpr_pts, tpr_pts)
        n_curves += 1
    if n_curves > 0:
        tpr_grid /= n_curves
    return tpr_grid, n_curves


def write_enrichment_summary(out_dir: Path, retrieval_csv: Path) -> None:
    """Persist the 200-point TPR curve next to `retrieval_csv` as npz.

    Call from each experiment's `run()` immediately after writing the
    retrieval CSV. Plotting then loads tiny npz files instead of
    re-grouping millions of CSV rows. An empty CSV writes no npz —
    `_load_or_build_curve` treats absence as "skip this method".
    """
    df = read_csv_or_empty(retrieval_csv)
    if df.empty:
        return
    tpr, n_curves = _build_curve_arrays(df)
    if n_curves == 0:
        return
    np.savez(
        out_dir / ENRICHMENT_CURVE_NPZ,
        log_fprs=LOG_FPR_GRID,
        tpr=tpr,
        n_curves=np.int64(n_curves),
    )


def _load_or_build_curve(method_dir: Path) -> np.ndarray | None:
    """Return the cached TPR array, building it from the CSV if missing.

    The npz is considered fresh when it exists and is at least as new as
    the source CSV. Stale npz files are rebuilt in place so older runs
    (from before this cache existed) self-heal on first plot.
    """
    csv_path = method_dir / "retrieval_rankings.csv"
    npz_path = method_dir / ENRICHMENT_CURVE_NPZ
    if not csv_path.exists():
        return None
    if npz_path.exists() and npz_path.stat().st_mtime >= csv_path.stat().st_mtime:
        with np.load(npz_path) as data:
            return data["tpr"].copy()
    write_enrichment_summary(method_dir, csv_path)
    if npz_path.exists():
        with np.load(npz_path) as data:
            return data["tpr"].copy()
    return None


def plot_enrichment_curves(
    exp_id: str,
    plots_dir: Path,
    method_ids: list[str],
    datasets: list[str],
) -> None:
    """Log-scale ROC per dataset, one line per method (avg over seeds & queries).

    Reads cached `enrichment_curve.npz` per (method, dataset) — built by
    `write_enrichment_summary` during `run()`. Falls back to recomputing
    + caching on the fly for older runs that predate the cache.
    """
    fig, axes_flat = _per_dataset_grid(len(datasets))

    for ax_idx, ds_id in enumerate(datasets):
        ax = axes_flat[ax_idx]
        ax.set_title(ds_id, fontsize=9)
        ax.set_xlabel("log₁₀(FPR)")
        ax.set_ylabel("TPR")
        ax.set_xlim(-3, 0)
        ax.set_ylim(0, 1)
        ax.plot(LOG_FPR_GRID, 10 ** LOG_FPR_GRID, color="grey", linewidth=0.8,
                linestyle="--", label="random")

        for m_id in method_ids:
            tpr = _load_or_build_curve(result_dir(exp_id, m_id, ds_id))
            if tpr is None:
                continue
            ax.plot(LOG_FPR_GRID, tpr, linewidth=1.2, label=m_id)

        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    for ax_idx in range(len(datasets), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(f"{exp_id}: Log-scale enrichment curves (avg over queries & seeds)")
    fig.tight_layout()
    fig.savefig(plots_dir / "enrichment_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(
    exp_id: str,
    plots_dir: Path,
    method_ids: list[str],
    datasets: list[str],
    metrics: list[str],
) -> None:
    """One PNG per scalar metric: methods × datasets heatmap."""
    for metric in metrics:
        data_grid = np.full((len(method_ids), len(datasets)), np.nan)
        for i, m_id in enumerate(method_ids):
            for j, ds_id in enumerate(datasets):
                p = result_dir(exp_id, m_id, ds_id) / "metrics.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                if d.get(metric) is not None:
                    data_grid[i, j] = d[metric]

        if np.all(np.isnan(data_grid)):
            continue

        fig, ax = plt.subplots(
            figsize=(max(6, len(datasets) * 0.9),
                     max(3, len(method_ids) * 0.6)),
        )
        im = ax.imshow(data_grid, aspect="auto", cmap="YlGn",
                       vmin=np.nanmin(data_grid), vmax=np.nanmax(data_grid))
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(method_ids)))
        ax.set_yticklabels(method_ids, fontsize=8)
        for i in range(len(method_ids)):
            for j in range(len(datasets)):
                v = data_grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{exp_id}: {metric} — methods × datasets")
        fig.tight_layout()
        fig.savefig(plots_dir / f"heatmap_{metric.lower().replace('-', '_')}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


