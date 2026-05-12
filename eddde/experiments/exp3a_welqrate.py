"""EXP-3a: WelQrate Retrieval (PROJECT_PLAN.md §5.3).

Protocol:
  For each of the 5 scaffold seeds provided by WelQrate:
    Task 1 — retrieval:
      For each test-split active as query, rank all valid+test molecules
      (i.e. everything except train) by ascending distance.

      Pool rationale:
        - Train is excluded for computational reasons (it is 60 % of the
          data and contributes no unique information for non-learned
          methods).
        - Valid and test are both retained. Valid molecules carry different
          scaffolds from the query (guaranteed by the scaffold split).
          Test molecules include scaffold-mates of the query, so
          fingerprint methods get no free scaffold-similarity advantage
          that would not also exist in real virtual screening. The
          scaffold diversity of retrieved actives is measured separately
          in EXP-6, which is the right experiment for that question.
        - For any future method trained on WelQrate data: if the valid
          set was used for early stopping, its molecules are present in
          the pool and the method has an advantage there. That is an
          acceptable trade-off; the test split remains uncontaminated
          regardless.

      Metrics computed per query then averaged across queries, then
      across seeds (mean ± SE over 5 seeds): LogAUC, BEDROC(α=20),
      EF1%, DCG100.

    Task 2 — k-NN classification:
      For each test-split molecule (active or inactive), find its
      K_MAX nearest neighbors within the same valid+test pool (minus
      the molecule itself). Consistent with Task 1.
      Metric: AUC-ROC at k = 5, 10, 20, averaged across seeds.

Output files written to `out/`:
  retrieval_rankings.csv   (schema: retrieval_common.RETRIEVAL_COLS)
  knn_neighbors.csv
    seed, test_id, test_activity, neighbor_rank, neighbor_id,
    neighbor_activity, neighbor_distance
  metrics.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..cache import hash_file, is_stale, write_manifest
from ..data.base import Stage
from ..data.sources.welqrate import N_SCAFFOLD_SEEDS
from . import retrieval_common as rc
from .base import result_dir

WELQRATE_DATASET_IDS = [
    "AID1798", "AID1843", "AID2258", "AID2689",
    "AID435008", "AID435034", "AID463087", "AID485290", "AID488997",
]

K_VALUES = (5, 10, 20)
K_MAX = max(K_VALUES)

_KNN_COLS = (
    "seed", "test_id", "test_activity", "neighbor_rank",
    "neighbor_id", "neighbor_activity", "neighbor_distance",
)


class Exp3aWelQrate:
    id = "EXP-3a"
    version = "v3"
    datasets = WELQRATE_DATASET_IDS

    metric_direction = {
        "M-LOGAUC":     +1,
        "M-BEDROC20":   +1,
        "M-EF1":        +1,
        "M-DCG100":     +1,
        "M-AUCROC_k5":  +1,
        "M-AUCROC_k10": +1,
        "M-AUCROC_k20": +1,
    }

    def run(self, method, stage_data, embeddings, dataset_id, out):
        df: pd.DataFrame = stage_data[Stage.SMILES].copy()
        df["id"] = df["id"].astype(str)
        out.mkdir(parents=True, exist_ok=True)

        mol_ids: list[str] = df["id"].tolist()
        activity: dict[str, int] = dict(zip(df["id"], df["activity"].astype(int)))

        retrieval_rows: list[dict] = []
        knn_rows: list[dict] = []

        seed_logauc:   list[float] = []
        seed_bedroc:   list[float] = []
        seed_ef1:      list[float] = []
        seed_dcg100:   list[float] = []
        seed_aucroc: dict[int, list[float]] = {k: [] for k in K_VALUES}

        for seed in range(1, N_SCAFFOLD_SEEDS + 1):
            split_col = f"scaffold_seed{seed}"
            split: dict[str, str] = df.set_index("id")[split_col].to_dict()

            test_ids        = [m for m in mol_ids if split[m] == "test"]
            test_active_ids = [m for m in test_ids if activity[m] == 1]
            pool_ids        = [m for m in mol_ids if split[m] in ("valid", "test")]
            n_actives_in_pool_base = sum(activity[c] for c in pool_ids)

            # ------------------------------------------------------------------
            # Task 1: retrieval
            # ------------------------------------------------------------------
            q_logauc = []
            q_bedroc = []
            q_ef1    = []
            q_dcg100 = []

            for query_id in test_active_ids:
                candidates = [m for m in pool_ids if m != query_id]
                distances = np.array([
                    method.distance(embeddings[query_id], embeddings[cid])
                    for cid in candidates
                ])
                order = np.argsort(distances, kind="stable")

                n_total = len(candidates)
                n_actives = n_actives_in_pool_base - 1  # query removed

                active_ranks: list[int] = []
                for rank_0, idx in enumerate(order):
                    cid = candidates[idx]
                    if activity[cid] == 1:
                        rank = rank_0 + 1
                        active_ranks.append(rank)
                        retrieval_rows.append({
                            "seed": seed,
                            "query_id": query_id,
                            "active_id": cid,
                            "rank": rank,
                            "distance": float(distances[idx]),
                            "n_total": n_total,
                            "n_actives_in_pool": n_actives,
                        })

                q_logauc.append(rc.logauc(active_ranks, n_total, n_actives))
                q_bedroc.append(rc.bedroc(active_ranks, n_total, n_actives))
                q_ef1.append(rc.ef_at_percent(active_ranks, n_total, n_actives, percent=1.0))
                q_dcg100.append(rc.dcg_at_k(active_ranks, k=100))

            seed_logauc.append(rc.nanmean(q_logauc))
            seed_bedroc.append(rc.nanmean(q_bedroc))
            seed_ef1.append(rc.nanmean(q_ef1))
            seed_dcg100.append(rc.nanmean(q_dcg100))

            # ------------------------------------------------------------------
            # Task 2: k-NN classification
            # ------------------------------------------------------------------
            knn_by_test: dict[str, list[int]] = {}

            for test_id in test_ids:
                candidates = [m for m in pool_ids if m != test_id]
                distances = np.array([
                    method.distance(embeddings[test_id], embeddings[cid])
                    for cid in candidates
                ])
                order = np.argsort(distances, kind="stable")
                neighbor_acts = []
                for rank_0 in range(min(K_MAX, len(candidates))):
                    neighbor_id = candidates[order[rank_0]]
                    neighbor_act = activity[neighbor_id]
                    neighbor_acts.append(neighbor_act)
                    knn_rows.append({
                        "seed": seed,
                        "test_id": test_id,
                        "test_activity": activity[test_id],
                        "neighbor_rank": rank_0 + 1,
                        "neighbor_id": neighbor_id,
                        "neighbor_activity": neighbor_act,
                        "neighbor_distance": float(distances[order[rank_0]]),
                    })
                knn_by_test[test_id] = neighbor_acts

            test_labels = [activity[t] for t in test_ids]
            for k in K_VALUES:
                pred_probs = [
                    float(np.mean(knn_by_test[t][:k])) if knn_by_test.get(t) else 0.0
                    for t in test_ids
                ]
                if len(set(test_labels)) < 2:
                    seed_aucroc[k].append(float("nan"))
                else:
                    seed_aucroc[k].append(float(roc_auc_score(test_labels, pred_probs)))

        pd.DataFrame(retrieval_rows, columns=list(rc.RETRIEVAL_COLS)).to_csv(
            out / "retrieval_rankings.csv", index=False)
        pd.DataFrame(knn_rows, columns=list(_KNN_COLS)).to_csv(
            out / "knn_neighbors.csv", index=False)

        metrics: dict = {}
        metrics.update(rc.metric_entry("M-LOGAUC",   seed_logauc))
        metrics.update(rc.metric_entry("M-BEDROC20", seed_bedroc))
        metrics.update(rc.metric_entry("M-EF1",      seed_ef1))
        metrics.update(rc.metric_entry("M-DCG100",   seed_dcg100))
        for k in K_VALUES:
            metrics.update(rc.metric_entry(f"M-AUCROC_k{k}", seed_aucroc[k]))

        (out / "metrics.json").write_text(rc.metrics_to_json(metrics))

        # Triage: silent nanmean swallows per-seed empties; this single line
        # surfaces the unusual case where every scaffold seed's test split was
        # empty (e.g. dataset is too small under test-mode for an unlucky seed,
        # or a future dataset ships with no test actives).
        if all(math.isnan(v) for v in seed_logauc):
            print(
                f"    [{self.id} x {method.id} on {dataset_id}] "
                "no test-active queries across any scaffold seed — "
                "all retrieval metrics NaN"
            )

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
                    if metric in data:
                        rows.append({"method": m_id, "dataset": ds_id, "metric": metric, "value": data[metric]})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def make_plots(self, exp_results_dir: Path, method_ids: list[str]) -> None:
        plots_dir = exp_results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        input_hashes: dict[str, str] = {}
        for m_id in method_ids:
            for ds_id in self.datasets:
                for fname in ("retrieval_rankings.csv", "knn_neighbors.csv", "metrics.json"):
                    p = result_dir(self.id, m_id, ds_id) / fname
                    if p.exists():
                        input_hashes[f"{m_id}/{ds_id}/{fname}"] = hash_file(p)

        if not input_hashes:
            return

        sentinel = plots_dir / "enrichment_curves.png"
        if not is_stale(sentinel, self.version, input_hashes):
            print(f"  [{self.id}] plots fresh")
            return

        print(f"  [{self.id}] generating plots...")
        rc.plot_enrichment_curves(self.id, plots_dir, method_ids, self.datasets)
        rc.plot_metric_heatmap(self.id, plots_dir, method_ids, self.datasets,
                               metrics=["M-LOGAUC", "M-BEDROC20", "M-EF1", "M-DCG100"])
        rc.plot_cumulative_recall(self.id, plots_dir, method_ids, self.datasets)
        rc.plot_rank_distributions(self.id, plots_dir, method_ids, self.datasets)
        self._plot_knn_aucroc_vs_k(plots_dir, method_ids)
        write_manifest(sentinel, version=self.version, inputs=input_hashes, compute_time=0.0, dataset_size=0)

    def _plot_knn_aucroc_vs_k(self, plots_dir: Path, method_ids: list[str]) -> None:
        """kNN AUC-ROC vs k per dataset, one line per method. EXP-3a only."""
        fig, axes_flat = rc._per_dataset_grid(len(self.datasets))

        for ax_idx, ds_id in enumerate(self.datasets):
            ax = axes_flat[ax_idx]
            ax.set_title(ds_id, fontsize=9)
            ax.set_xlabel("k")
            ax.set_ylabel("AUC-ROC")
            ax.set_xticks(list(K_VALUES))
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", label="random")

            for m_id in method_ids:
                p = result_dir(self.id, m_id, ds_id) / "metrics.json"
                if not p.exists():
                    continue
                d = json.loads(p.read_text())
                ys = [d.get(f"M-AUCROC_k{k}") for k in K_VALUES]
                if any(v is not None for v in ys):
                    ys_clean = [v if v is not None else float("nan") for v in ys]
                    ax.plot(list(K_VALUES), ys_clean, marker="o", linewidth=1.2,
                            markersize=4, label=m_id)

            if ax_idx == 0:
                ax.legend(fontsize=7, loc="lower right")

        for ax_idx in range(len(self.datasets), len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)

        fig.suptitle(f"{self.id}: kNN AUC-ROC vs k (mean over seeds)")
        fig.tight_layout()
        fig.savefig(plots_dir / "knn_aucroc_vs_k.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
