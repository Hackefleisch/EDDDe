"""EXP-4: Activity Cliff Analysis (PROJECT_PLAN.md §5.6).

Per ChEMBL target, the similar-pair pool is the set of pairs (i, j) with
similarity >= 0.9 by ANY of:
  - ECFP4-Tanimoto (Morgan radius=2, 1024 bits)
  - graph-framework ECFP-Tanimoto (MakeScaffoldGeneric on the whole molecule)
  - 1 - normalized Levenshtein on canonical SMILES
A similar pair is a CLIFF if |ΔpY| >= 1 (i.e. >= 10x difference in Ki/EC50),
otherwise NON-CLIFF. Metrics are computed on the similar-pair pool only.

The cliff-defining ECFP uses radius=2, fpSize=1024. Baseline B1 (ECFP4) uses
2048 bits independently.

Metrics (§6.1):
  M-DIST-POTENCY-RHO  Spearman rho between d(A,B) and |ΔpY|. Higher is better.
  M-CLIFF-AUC         ROC-AUC for cliff vs non-cliff using SALI as score.
                      Higher is better.
  M-SALI-DIST         KS statistic between cliff and non-cliff SALI
                      distributions. Higher is better separation.

SALI(A,B) = |ΔpY| / max(eps, d_method(A,B)). PROJECT_PLAN §6.1 writes this
as |ΔpKi|/(1 - sim); for similarity-based methods d = 1 - sim and the two
forms coincide, so substituting d_method generalises to native-distance
methods (CLAUDE.md "Distance convention").

Restricting to similar pairs (typically a few hundred per target) instead of
all O(N^2) pairs keeps the per-target cost bounded across 30 targets and
fits the activity-cliff framing (cliffs are similar pairs by definition).

Output files written to `out/`:
  pairs.csv     a, b, sim_ecfp, sim_scaffold, sim_lev, delta_pY_abs, cliff, distance
  metrics.json  metric values, p-values, and pool counts

Plots (§8):
  cliff_violin.png  d(A,B) split by cliff/non-cliff, one column per method
  dist_vs_dpy.png   d(A,B) vs |ΔpY| scatter, one column per method
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import roc_auc_score

from ..cache import hash_file, is_stale, write_manifest
from ..data.base import Stage
from ..data.sources.molecule_ace import MOLECULE_ACE_DATASET_IDS
from .base import result_dir


SIM_THRESHOLD = 0.9
CLIFF_DPY_THRESHOLD = 1.0  # |ΔpY| >= 1 means >= 10x potency
EPS = 1e-9
ECFP_RADIUS = 2
ECFP_BITS = 1024


# ---------------------------------------------------------------------------
# Similarity helpers. Each returns a symmetric (n, n) float matrix in [0, 1]
# with diagonal = 1.
# ---------------------------------------------------------------------------

def _morgan_fps(mols: list[Chem.Mol]) -> list:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)
    return [gen.GetFingerprint(m) for m in mols]


def _tanimoto_matrix(fps: list) -> np.ndarray:
    n = len(fps)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i:])
        M[i, i:] = sims
        M[i:, i] = sims
    return M


def _scaffold_mol(mol: Chem.Mol) -> Chem.Mol:
    """Graph framework: all atoms to C, all bond orders to single, applied to
    the whole molecule. Falls back to the plain Murcko scaffold on the rare
    inputs where MakeScaffoldGeneric raises."""
    try:
        return MurckoScaffold.MakeScaffoldGeneric(mol)
    except Exception:
        return MurckoScaffold.GetScaffoldForMol(mol)


def _scaled_levenshtein_matrix(smiles: list[str]) -> np.ndarray:
    """sim(s, t) = 1 - Levenshtein(s, t) / max(len(s), len(t))."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz import process

    dist = process.cdist(smiles, smiles, scorer=Levenshtein.distance, dtype=np.int32)
    lengths = np.array([len(s) for s in smiles], dtype=np.float32)
    max_len = np.maximum(lengths[:, None], lengths[None, :])
    max_len = np.maximum(max_len, 1.0)  # avoid /0 on empty SMILES
    sim = 1.0 - dist.astype(np.float32) / max_len
    np.fill_diagonal(sim, 1.0)
    return sim


# ---------------------------------------------------------------------------

class Exp4ActivityCliffs:
    id = "EXP-4"
    version = "v1-molecule-ace"
    datasets = MOLECULE_ACE_DATASET_IDS

    metric_direction = {
        "M-DIST-POTENCY-RHO": +1,
        "M-CLIFF-AUC":        +1,
        "M-SALI-DIST":        +1,
    }
    metric_datasets = {m: MOLECULE_ACE_DATASET_IDS for m in metric_direction}

    def run(self, method, stage_data, embeddings, dataset_id, out):
        df: pd.DataFrame = stage_data[Stage.SMILES]
        mol_ids = df["id"].astype(str).tolist()
        smiles = df["smiles"].astype(str).tolist()
        pY = df["pY"].astype(float).to_numpy()
        n = len(mol_ids)

        out.mkdir(parents=True, exist_ok=True)

        mols = [Chem.MolFromSmiles(s) for s in smiles]
        if any(m is None for m in mols):
            bad = [mol_ids[i] for i, m in enumerate(mols) if m is None]
            raise RuntimeError(f"[{dataset_id}] {len(bad)} unparseable SMILES: {bad[:5]}...")

        fps_full = _morgan_fps(mols)
        fps_scaf = _morgan_fps([_scaffold_mol(m) for m in mols])
        sim_ecfp = _tanimoto_matrix(fps_full)
        sim_scaf = _tanimoto_matrix(fps_scaf)
        sim_lev = _scaled_levenshtein_matrix(smiles)

        similar = (
            (sim_ecfp >= SIM_THRESHOLD)
            | (sim_scaf >= SIM_THRESHOLD)
            | (sim_lev  >= SIM_THRESHOLD)
        )
        triu = np.triu(np.ones_like(similar, dtype=bool), k=1)
        i_idx, j_idx = np.where(similar & triu)

        n_pairs = len(i_idx)
        metrics: dict = {
            "n_molecules": int(n),
            "n_pairs": int(n_pairs),
            "n_cliff_pairs": 0,
            "n_noncliff_pairs": 0,
            "M-DIST-POTENCY-RHO": float("nan"),
            "M-CLIFF-AUC":        float("nan"),
            "M-SALI-DIST":        float("nan"),
        }

        if n_pairs == 0:
            pd.DataFrame(columns=[
                "a", "b", "sim_ecfp", "sim_scaffold", "sim_lev",
                "delta_pY_abs", "cliff", "distance",
            ]).to_csv(out / "pairs.csv", index=False)
            (out / "metrics.json").write_text(_json_metrics(metrics))
            return metrics

        d_pY = np.abs(pY[i_idx] - pY[j_idx])
        cliff = (d_pY >= CLIFF_DPY_THRESHOLD).astype(np.int8)

        dist = np.empty(n_pairs, dtype=np.float64)
        for k, (i, j) in enumerate(zip(i_idx, j_idx)):
            dist[k] = float(method.distance(embeddings[mol_ids[i]], embeddings[mol_ids[j]]))

        pairs_df = pd.DataFrame({
            "a": [mol_ids[i] for i in i_idx],
            "b": [mol_ids[j] for j in j_idx],
            "sim_ecfp":     sim_ecfp[i_idx, j_idx],
            "sim_scaffold": sim_scaf[i_idx, j_idx],
            "sim_lev":      sim_lev[i_idx, j_idx],
            "delta_pY_abs": d_pY,
            "cliff":        cliff,
            "distance":     dist,
        })
        pairs_df.to_csv(out / "pairs.csv", index=False)

        if n_pairs >= 2:
            rho, p_rho = spearmanr(dist, d_pY)
            metrics["M-DIST-POTENCY-RHO"] = float(rho) if not np.isnan(rho) else float("nan")
            metrics["M-DIST-POTENCY-RHO_pvalue"] = float(p_rho) if not np.isnan(p_rho) else float("nan")

        n_cliff = int(cliff.sum())
        n_noncliff = int((1 - cliff).sum())
        metrics["n_cliff_pairs"] = n_cliff
        metrics["n_noncliff_pairs"] = n_noncliff

        sali = d_pY / np.maximum(EPS, dist)
        if n_cliff > 0 and n_noncliff > 0:
            metrics["M-CLIFF-AUC"] = float(roc_auc_score(cliff, sali))
            ks = ks_2samp(sali[cliff == 1], sali[cliff == 0])
            metrics["M-SALI-DIST"] = float(ks.statistic)
            metrics["M-SALI-DIST_pvalue"] = float(ks.pvalue)

        (out / "metrics.json").write_text(_json_metrics(metrics))
        return metrics


    def collect_results(self, method_ids: list[str]) -> pd.DataFrame:
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

    def make_plots(self, exp_results_dir: Path, method_ids: list[str]) -> None:
        plots_dir = exp_results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        input_hashes: dict[str, str] = {}
        for m_id in method_ids:
            for ds_id in self.datasets:
                p = result_dir(self.id, m_id, ds_id) / "pairs.csv"
                if p.exists():
                    input_hashes[f"{m_id}/{ds_id}/pairs.csv"] = hash_file(p)

        if not input_hashes:
            return

        sentinel = plots_dir / "cliff_violin.png"
        if not is_stale(sentinel, self.version, input_hashes):
            print(f"  [{self.id}] plots fresh")
            return

        print(f"  [{self.id}] generating plots...")
        self._plot_cliff_violin(plots_dir, method_ids)
        self._plot_dist_vs_dpy(plots_dir, method_ids)
        write_manifest(sentinel, version=self.version, inputs=input_hashes, compute_time=0.0, dataset_size=0)

    def _load_pooled(self, method_id: str) -> pd.DataFrame:
        """Concatenate pairs.csv across all targets for one method."""
        frames = []
        for ds_id in self.datasets:
            p = result_dir(self.id, method_id, ds_id) / "pairs.csv"
            if p.exists():
                frame = pd.read_csv(p)
                if not frame.empty:
                    frame["dataset"] = ds_id
                    frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _plot_cliff_violin(self, plots_dir: Path, method_ids: list[str]) -> None:
        n = len(method_ids)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False, sharey=False)
        for col, m_id in enumerate(method_ids):
            ax = axes[0, col]
            df = self._load_pooled(m_id)
            if df.empty:
                ax.set_visible(False)
                continue
            data = [
                df.loc[df["cliff"] == 0, "distance"].to_numpy(),
                df.loc[df["cliff"] == 1, "distance"].to_numpy(),
            ]
            labels = [f"non-cliff (n={len(data[0])})", f"cliff (n={len(data[1])})"]
            # Drop empty groups (violinplot raises on empty arrays).
            keep = [(d, lbl) for d, lbl in zip(data, labels) if len(d) > 0]
            if not keep:
                ax.set_visible(False)
                continue
            parts = ax.violinplot([d for d, _ in keep], showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(keep) + 1))
            ax.set_xticklabels([lbl for _, lbl in keep], fontsize=8)
            ax.set_ylabel("d(A, B)")
            ax.set_title(m_id, fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")
            for i, body in enumerate(parts["bodies"]):
                body.set_facecolor("#F44336" if "cliff (" in keep[i][1] and not keep[i][1].startswith("non") else "#2196F3")
                body.set_alpha(0.6)
        fig.suptitle("EXP-4: Distance distribution, cliff vs non-cliff similar pairs (pooled across 30 targets)", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "cliff_violin.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_dist_vs_dpy(self, plots_dir: Path, method_ids: list[str]) -> None:
        n = len(method_ids)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False, sharey=False)
        rng = np.random.default_rng(42)
        for col, m_id in enumerate(method_ids):
            ax = axes[0, col]
            df = self._load_pooled(m_id)
            if df.empty:
                ax.set_visible(False)
                continue
            # Subsample for plot legibility (metrics use full data).
            if len(df) > 5000:
                df = df.iloc[rng.choice(len(df), 5000, replace=False)]
            colors = np.where(df["cliff"] == 1, "#F44336", "#2196F3")
            ax.scatter(df["delta_pY_abs"], df["distance"], c=colors, s=8, alpha=0.4, edgecolors="none")
            # OLS line on the (sub)sampled data.
            if len(df) >= 2:
                slope, intercept = np.polyfit(df["delta_pY_abs"], df["distance"], 1)
                xs = np.linspace(df["delta_pY_abs"].min(), df["delta_pY_abs"].max(), 50)
                ax.plot(xs, slope * xs + intercept, color="black", linewidth=1)
            ax.axvline(CLIFF_DPY_THRESHOLD, color="grey", linestyle="--", linewidth=0.5)
            ax.set_xlabel("|ΔpY|")
            ax.set_ylabel("d(A, B)")
            ax.set_title(m_id, fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.suptitle("EXP-4: d(A,B) vs |ΔpY| (red = cliff, blue = non-cliff; subsampled to 5k)", y=1.02)
        fig.tight_layout()
        fig.savefig(plots_dir / "dist_vs_dpy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _json_metrics(metrics: dict) -> str:
    return json.dumps(
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in metrics.items()},
        indent=2,
        sort_keys=True,
    )
