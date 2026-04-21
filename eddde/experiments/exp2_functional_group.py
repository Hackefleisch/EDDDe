"""EXP-2: Functional Group Substitution Sensitivity (PROJECT_PLAN.md §5.2).

Metrics (§6.1):
  M-HAMMETT     — Spearman ρ between Hammett σ_para and d(X-compound, H-compound). S8 only.
  M-SILHOUETTE  — Silhouette score of 2D MDS projection of distance matrix, using
                  donor/acceptor/neutral labels. S6 and S7.

Note: M-HALO-ORDER was dropped because ElektroNN's basis set does not include Br
(supported = {H, C, N, O, F, S, Cl}), so the F/Cl/Br ordering cannot be computed.
See CLAUDE.md §"SMILES-stage element filter".

Plots (§8):
  mds_s6.png         — 2D MDS of S6 coloured by donor/acceptor/neutral.
  hammett_scatter.png — Hammett σ vs d(X, H-compound) per method, S8.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..cache import hash_file, is_stale, write_manifest
from ..data.base import Stage
from .base import result_dir

LABEL_COLORS = {"donor": "#2196F3", "acceptor": "#F44336", "neutral": "#9E9E9E"}


def _pairwise_distances(method, embeddings: dict, mol_ids: list[str]) -> np.ndarray:
    n = len(mol_ids)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(method.distance(embeddings[mol_ids[i]], embeddings[mol_ids[j]]))
            D[i, j] = d
            D[j, i] = d
    return D


def _mds_2d(D: np.ndarray) -> np.ndarray:
    from sklearn.manifold import MDS

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    return mds.fit_transform(D)


def _m_silhouette(coords_2d: np.ndarray, labels: list[str]) -> float:
    from sklearn.metrics import silhouette_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(silhouette_score(coords_2d, labels))


class Exp2FunctionalGroup:
    id = "EXP-2"
    version = "v2-drop-halo"
    datasets = ["S6", "S7", "S8"]

    metric_direction = {"M-HAMMETT": 1, "M-SILHOUETTE": 1}

    def run(self, method, stage_data, embeddings, dataset_id, out):
        df: pd.DataFrame = stage_data[Stage.SMILES]
        mol_ids = list(df["id"].astype(str))
        out.mkdir(parents=True, exist_ok=True)

        D = _pairwise_distances(method, embeddings, mol_ids)

        dist_df = pd.DataFrame(D, index=mol_ids, columns=mol_ids)
        dist_df.to_csv(out / "pairwise_distances.csv")

        metrics: dict = {}

        if dataset_id in ("S6", "S7"):
            labels = list(df["label"])
            coords = _mds_2d(D)
            np.save(out / "mds_coords.npy", coords)
            pd.DataFrame({"mol_id": mol_ids, "x": coords[:, 0], "y": coords[:, 1], "label": labels}).to_csv(
                out / "mds_coords.csv", index=False
            )
            metrics["M-SILHOUETTE"] = _m_silhouette(coords, labels)

        elif dataset_id == "S8":
            sigma = list(df["sigma_para"].astype(float))
            h_ids = [m for m in mol_ids if m.endswith("_H")]
            if h_ids:
                h_idx = mol_ids.index(h_ids[0])
                dist_from_h = [float(D[i, h_idx]) for i in range(len(mol_ids))]
                rho, p = spearmanr(sigma, dist_from_h)
                pd.DataFrame({"mol_id": mol_ids, "sigma_para": sigma, "dist_from_H": dist_from_h}).to_csv(
                    out / "hammett_data.csv", index=False
                )
                metrics["M-HAMMETT"] = float(rho)
                metrics["M-HAMMETT_pvalue"] = float(p)

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

        input_hashes: dict[str, str] = {}
        for m_id in method_ids:
            for ds_id in self.datasets:
                for fname in ("pairwise_distances.csv", "mds_coords.csv", "hammett_data.csv"):
                    p = result_dir(self.id, m_id, ds_id) / fname
                    if p.exists():
                        input_hashes[f"{m_id}/{ds_id}/{fname}"] = hash_file(p)

        if not input_hashes:
            return

        sentinel = plots_dir / "mds_s6.png"
        if not is_stale(sentinel, self.version, input_hashes):
            print(f"  [{self.id}] plots fresh")
            return

        print(f"  [{self.id}] generating plots...")
        self._plot_mds(plots_dir, method_ids, "S6")
        self._plot_mds(plots_dir, method_ids, "S7")
        self._plot_hammett(plots_dir, method_ids)
        write_manifest(sentinel, version=self.version, inputs=input_hashes, compute_time=0.0, dataset_size=0)

    def collect_results(self, method_ids: list[str]) -> pd.DataFrame:
        rows = []
        for m_id in method_ids:
            for ds_id in self.datasets:
                p = result_dir(self.id, m_id, ds_id) / "metrics.json"
                if not p.exists():
                    continue
                data = json.loads(p.read_text())
                for metric in self.metric_direction:
                    if metric in data:
                        rows.append({"method": m_id, "dataset": ds_id, "metric": metric, "value": data[metric]})
        return pd.DataFrame(rows)

    def _plot_mds(self, plots_dir: Path, method_ids: list[str], ds_id: str) -> None:
        n_methods = len(method_ids)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4), squeeze=False)
        for col, m_id in enumerate(method_ids):
            ax = axes[0, col]
            csv = result_dir(self.id, m_id, ds_id) / "mds_coords.csv"
            if not csv.exists():
                ax.set_visible(False)
                continue
            df = pd.read_csv(csv)
            for label, grp in df.groupby("label"):
                ax.scatter(grp["x"], grp["y"], c=LABEL_COLORS.get(label, "#000"), label=label, s=60, edgecolors="k", linewidths=0.4)
                for _, row in grp.iterrows():
                    sub = str(row["mol_id"]).split("_", 1)[-1]
                    ax.annotate(sub, (row["x"], row["y"]), fontsize=7, textcoords="offset points", xytext=(4, 4))
            ax.set_title(m_id, fontsize=9)
            ax.set_xlabel("MDS dim 1")
            ax.set_ylabel("MDS dim 2")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"EXP-2: MDS of {ds_id} (donor/acceptor/neutral)", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / f"mds_{ds_id.lower()}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_hammett(self, plots_dir: Path, method_ids: list[str]) -> None:
        n_methods = len(method_ids)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4), squeeze=False)
        for col, m_id in enumerate(method_ids):
            ax = axes[0, col]
            csv = result_dir(self.id, m_id, "S8") / "hammett_data.csv"
            if not csv.exists():
                ax.set_visible(False)
                continue
            df = pd.read_csv(csv)
            ax.scatter(df["sigma_para"], df["dist_from_H"], s=50, edgecolors="k", linewidths=0.4)
            for _, row in df.iterrows():
                sub = str(row["mol_id"]).split("_", 1)[-1]
                ax.annotate(sub, (row["sigma_para"], row["dist_from_H"]), fontsize=7, textcoords="offset points", xytext=(4, 4))
            ax.set_xlabel("Hammett σ_para")
            ax.set_ylabel("d(X-compound, H-compound)")
            ax.set_title(m_id, fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.suptitle("EXP-2: Hammett σ_para vs distance from H-compound (S8)", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "hammett_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
