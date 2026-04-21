"""EXP-1: Homologous Series Smoothness (PROJECT_PLAN.md §5.1).

Metrics (§6.1):
  M-MONO   — Spearman ρ between |i−j| and d(i,j) over all pairs. Range [-1,1]; 1 = perfect.
  M-SMOOTH — Std dev of consecutive-distance ratios d(k,k+1)/d(k-1,k). Lower is smoother.
  M-LIN    — R² of linear regression of d(mol_1, mol_k) vs k.

Plots (§8):
  distance_from_first.png  — x=chain length, y=d(mol_1,mol_k); methods overlaid per series.
  mono_scatter.png         — x=|i−j|, y=d(i,j) for all pairs; methods overlaid per series.
  consecutive_distances.png — x=k, y=d(mol_k,mol_{k+1}); methods overlaid per series.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

from ..cache import hash_file, is_stale, write_manifest
from ..data.base import Stage
from .base import result_dir


SERIES_LABELS = {
    "S1": "n-Alkanes",
    "S2": "n-Alkanols",
    "S3": "n-Alkanoic acids",
    "S4": "n-Alkylamines",
    "S5": "Polyethylene glycols",
}


def _m_smooth(consecutive_distances: list[float]) -> float:
    """Std dev of ratios d(k,k+1)/d(k-1,k) for k=2..n-1.

    Undefined (nan) when any denominator is zero or fewer than 3 molecules.
    """
    if len(consecutive_distances) < 3:
        return float("nan")
    ratios = []
    for i in range(1, len(consecutive_distances)):
        denom = consecutive_distances[i - 1]
        if denom == 0.0:
            return float("nan")
        ratios.append(consecutive_distances[i] / denom)
    return float(np.std(ratios))


def _m_lin(positions: list[int], distances_from_first: list[float]) -> tuple[float, float]:
    """R² and slope p-value of OLS regression of d(mol_1, mol_k) on k."""
    _, _, r, p, _ = linregress(positions, distances_from_first)
    return float(r ** 2), float(p)


def _make_grid(n_series: int, ncols: int = 3):
    nrows = (n_series + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = axes.flatten() if n_series > 1 else [axes]
    return fig, axes_flat


def _hide_unused(axes_flat, n_used: int) -> None:
    for i in range(n_used, len(axes_flat)):
        axes_flat[i].set_visible(False)


class Exp1Homologous:
    id = "EXP-1"
    version = "v6-more-plots"
    datasets = ["S1", "S2", "S3", "S4", "S5"]

    def run(self, method, stage_data, embeddings, dataset_id, out):
        df: pd.DataFrame = (
            stage_data[Stage.SMILES].sort_values("position").reset_index(drop=True)
        )
        mol_ids = [str(row.id) for row in df.itertuples(index=False)]
        positions = [int(row.position) for row in df.itertuples(index=False)]

        # All pairs — used for M-MONO and mono_scatter plot
        rows = []
        for a, b in combinations(df.itertuples(index=False), 2):
            d = method.distance(embeddings[str(a.id)], embeddings[str(b.id)])
            rows.append(
                {
                    "a": a.id,
                    "b": b.id,
                    "gap": abs(int(a.position) - int(b.position)),
                    "distance": float(d),
                }
            )
        pairs = pd.DataFrame(rows)
        out.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(out / "pairs.csv", index=False)

        rho, p = spearmanr(pairs["gap"], pairs["distance"])

        # Consecutive distances d(k, k+1) — used for M-SMOOTH and consecutive plot
        consec = [
            method.distance(embeddings[mol_ids[i]], embeddings[mol_ids[i + 1]])
            for i in range(len(mol_ids) - 1)
        ]
        pd.DataFrame({"k": positions[:-1], "distance": consec}).to_csv(
            out / "consecutive.csv", index=False
        )

        # Distances from first molecule d(mol_1, mol_k) — used for M-LIN and from_first plot
        from_first = [
            method.distance(embeddings[mol_ids[0]], embeddings[mol_ids[k]])
            for k in range(1, len(mol_ids))
        ]
        pd.DataFrame({"position": positions[1:], "distance": from_first}).to_csv(
            out / "from_first.csv", index=False
        )

        m_lin, m_lin_p = _m_lin(positions[1:], from_first)
        metrics = {
            "M-MONO": float(rho),
            "M-MONO_pvalue": float(p),
            "M-SMOOTH": _m_smooth(consec),
            "M-LIN": m_lin,
            "M-LIN_pvalue": m_lin_p,
        }
        (out / "metrics.json").write_text(
            json.dumps(
                {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()},
                indent=2,
                sort_keys=True,
            )
        )
        return metrics

    def make_plots(self, exp_results_dir: Path, method_ids: list[str]) -> None:
        plots_dir = exp_results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Collect input hashes across all (method, dataset) result files
        input_hashes: dict[str, str] = {}
        for m_id in method_ids:
            for ds_id in self.datasets:
                for fname in ("from_first.csv", "pairs.csv", "consecutive.csv"):
                    p = result_dir(self.id, m_id, ds_id) / fname
                    if p.exists():
                        input_hashes[f"{m_id}/{ds_id}/{fname}"] = hash_file(p)

        if not input_hashes:
            return

        # Use distance_from_first as the sentinel for all three plots
        sentinel = plots_dir / "distance_from_first.png"
        if not is_stale(sentinel, self.version, input_hashes):
            print(f"  [{self.id}] plots fresh")
            return

        print(f"  [{self.id}] generating plots...")
        self._plot_distance_from_first(plots_dir, method_ids)
        self._plot_mono_scatter(plots_dir, method_ids)
        self._plot_consecutive_distances(plots_dir, method_ids)
        write_manifest(sentinel, version=self.version, inputs=input_hashes, compute_time=0.0, dataset_size=0)

    # Direction: +1 = higher is better, -1 = lower is better
    metric_direction = {"M-MONO": 1, "M-SMOOTH": -1, "M-LIN": 1}

    def collect_results(self, method_ids: list[str]) -> pd.DataFrame:
        """Return tidy DataFrame: method, dataset, metric, value."""
        rows = []
        for m_id in method_ids:
            for ds_id in self.datasets:
                p = result_dir(self.id, m_id, ds_id) / "metrics.json"
                if not p.exists():
                    continue
                data = json.loads(p.read_text())
                for metric in self.metric_direction:
                    rows.append({
                        "method": m_id,
                        "dataset": ds_id,
                        "metric": metric,
                        "value": data.get(metric),
                    })
        return pd.DataFrame(rows)

    def _plot_distance_from_first(self, plots_dir: Path, method_ids: list[str]) -> None:
        fig, axes_flat = _make_grid(len(self.datasets))
        for ax_idx, ds_id in enumerate(self.datasets):
            ax = axes_flat[ax_idx]
            for m_id in method_ids:
                csv = result_dir(self.id, m_id, ds_id) / "from_first.csv"
                if not csv.exists():
                    continue
                df = pd.read_csv(csv)
                ax.plot(df["position"], df["distance"], marker="o", markersize=4, label=m_id)
            ax.set_title(SERIES_LABELS.get(ds_id, ds_id))
            ax.set_xlabel("Chain length k")
            ax.set_ylabel("d(mol₁, molₖ)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        _hide_unused(axes_flat, len(self.datasets))
        fig.suptitle("EXP-1: Distance from first molecule in homologous series", y=1.01)
        fig.tight_layout()
        fig.savefig(plots_dir / "distance_from_first.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_mono_scatter(self, plots_dir: Path, method_ids: list[str]) -> None:
        fig, axes_flat = _make_grid(len(self.datasets))
        for ax_idx, ds_id in enumerate(self.datasets):
            ax = axes_flat[ax_idx]
            for m_id in method_ids:
                csv = result_dir(self.id, m_id, ds_id) / "pairs.csv"
                if not csv.exists():
                    continue
                df = pd.read_csv(csv)
                ax.scatter(df["gap"], df["distance"], s=10, alpha=0.5, label=m_id)
            ax.set_title(SERIES_LABELS.get(ds_id, ds_id))
            ax.set_xlabel("|i − j|")
            ax.set_ylabel("d(molᵢ, molⱼ)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        _hide_unused(axes_flat, len(self.datasets))
        fig.suptitle("EXP-1: All-pairs distance vs chain-length gap (M-MONO)", y=1.01)
        fig.tight_layout()
        fig.savefig(plots_dir / "mono_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_consecutive_distances(self, plots_dir: Path, method_ids: list[str]) -> None:
        fig, axes_flat = _make_grid(len(self.datasets))
        for ax_idx, ds_id in enumerate(self.datasets):
            ax = axes_flat[ax_idx]
            for m_id in method_ids:
                csv = result_dir(self.id, m_id, ds_id) / "consecutive.csv"
                if not csv.exists():
                    continue
                df = pd.read_csv(csv)
                ax.plot(df["k"], df["distance"], marker="o", markersize=4, label=m_id)
            ax.set_title(SERIES_LABELS.get(ds_id, ds_id))
            ax.set_xlabel("Chain length k")
            ax.set_ylabel("d(molₖ, molₖ₊₁)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        _hide_unused(axes_flat, len(self.datasets))
        fig.suptitle("EXP-1: Consecutive step distances (M-SMOOTH)", y=1.01)
        fig.tight_layout()
        fig.savefig(plots_dir / "consecutive_distances.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
