"""EXP-3a: WelQrate Retrieval (PROJECT_PLAN.md §5.3).

Protocol:
  For each of the 5 scaffold seeds provided by WelQrate:
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

Output files written to `out/`:
  retrieval_rankings.csv   (schema: retrieval_common.RETRIEVAL_COLS)
  metrics.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..cache import hash_file, is_stale, write_manifest
from ..data.base import Stage
from ..data.sources.welqrate import N_SCAFFOLD_SEEDS
from ..methods.distance import pairwise_matrix
from . import retrieval_common as rc
from .base import result_dir

WELQRATE_DATASET_IDS = [
    "AID1798", "AID1843", "AID2258", "AID2689",
    "AID435008", "AID435034", "AID463087", "AID485290", "AID488997",
]


class Exp3aWelQrate:
    id = "EXP-3a"
    version = "v5"
    datasets = WELQRATE_DATASET_IDS

    metric_direction = {
        "M-LOGAUC":   +1,
        "M-BEDROC20": +1,
        "M-EF1":      +1,
        "M-DCG100":   +1,
    }

    def run(self, method, stage_data, embeddings, dataset_id, out):
        df: pd.DataFrame = stage_data[Stage.SMILES].copy()
        df["id"] = df["id"].astype(str)
        out.mkdir(parents=True, exist_ok=True)

        mol_ids: list[str] = df["id"].tolist()
        activity: dict[str, int] = dict(zip(df["id"], df["activity"].astype(int)))

        retrieval_rows: list[dict] = []

        seed_logauc:   list[float] = []
        seed_bedroc:   list[float] = []
        seed_ef1:      list[float] = []
        seed_dcg100:   list[float] = []

        for seed in range(1, N_SCAFFOLD_SEEDS + 1):
            split_col = f"scaffold_seed{seed}"
            split: dict[str, str] = df.set_index("id")[split_col].to_dict()

            test_active_ids = [m for m in mol_ids if split[m] == "test" and activity[m] == 1]
            pool_ids        = [m for m in mol_ids if split[m] in ("valid", "test")]
            n_actives_in_pool_base = sum(activity[c] for c in pool_ids)
            pool_col_of = {c: j for j, c in enumerate(pool_ids)}

            q_logauc = []
            q_bedroc = []
            q_ef1    = []
            q_dcg100 = []

            # One matrix per seed: (test_actives × pool). Phase A picks one row
            # per query and slices out the self-column.
            D = pairwise_matrix(method, embeddings, test_active_ids, pool_ids)

            for i, query_id in enumerate(test_active_ids):
                self_col = pool_col_of[query_id]
                row = D[i]
                # Argsort over the full row, then drop the self index from the
                # ordering — preserves stable sort on ties without an extra mask.
                order = np.argsort(row, kind="stable")
                order = order[order != self_col]

                n_total = len(pool_ids) - 1
                n_actives = n_actives_in_pool_base - 1

                active_ranks: list[int] = []
                for rank_0, idx in enumerate(order):
                    cid = pool_ids[idx]
                    if activity[cid] == 1:
                        rank = rank_0 + 1
                        active_ranks.append(rank)
                        retrieval_rows.append({
                            "seed": seed,
                            "query_id": query_id,
                            "active_id": cid,
                            "rank": rank,
                            "distance": float(row[idx]),
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

        pd.DataFrame(retrieval_rows, columns=list(rc.RETRIEVAL_COLS)).to_csv(
            out / "retrieval_rankings.csv", index=False)
        rc.write_enrichment_summary(out, out / "retrieval_rankings.csv")

        metrics: dict = {}
        metrics.update(rc.metric_entry("M-LOGAUC",   seed_logauc))
        metrics.update(rc.metric_entry("M-BEDROC20", seed_bedroc))
        metrics.update(rc.metric_entry("M-EF1",      seed_ef1))
        metrics.update(rc.metric_entry("M-DCG100",   seed_dcg100))

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
                for fname in ("retrieval_rankings.csv", "metrics.json"):
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
        for stale_name in ("cumulative_recall.png", "rank_distributions.png"):
            (plots_dir / stale_name).unlink(missing_ok=True)
        rc.plot_enrichment_curves(self.id, plots_dir, method_ids, self.datasets)
        rc.plot_metric_heatmap(self.id, plots_dir, method_ids, self.datasets,
                               metrics=["M-LOGAUC", "M-BEDROC20", "M-EF1", "M-DCG100"])
        write_manifest(sentinel, version=self.version, inputs=input_hashes, compute_time=0.0, dataset_size=0)
