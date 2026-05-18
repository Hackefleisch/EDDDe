"""EXP-3b: MUV Retrieval (PROJECT_PLAN.md §5.4).

MUV was designed to defeat analog-based similarity: actives and decoys are
property-matched and topologically dissimilar by construction. This is the
benchmark where a topology-only fingerprint should *not* succeed, so any
MUT advantage should show most clearly here.

Protocol:
  Per target (one Dataset = one MUV assay):
    For each of N_QUERY_DRAWS seeds:
      - Pick one random active as query (deterministically seeded so
        reruns are stable across machines).
      - Rank every other molecule in the target by ascending distance.
      - Record active ranks and per-query metrics (AUC-ROC, BEDROC20, EF1).
    Aggregate: mean ± SE over N_QUERY_DRAWS seeds.

  Test-mode caveat: if a target's downsampled SMILES drops every active
  for a given seed (it shouldn't — the MUV test_mode_subsample preserves
  all 30 actives), that seed contributes NaN. The aggregator silently
  ignores NaN, matching EXP-3a's policy.

Output files written to `out/`:
  retrieval_rankings.csv   (schema: retrieval_common.RETRIEVAL_COLS)
  metrics.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .. import SEED
from ..cache import hash_file, is_stale, write_manifest
from ..data.base import Stage
from ..data.sources.muv import MUV_AIDS
from ..methods.distance import pairwise_matrix
from . import retrieval_common as rc
from .base import result_dir


MUV_DATASET_IDS: list[str] = [f"MUV_{aid}" for aid in MUV_AIDS]

# 5 random query draws per target — matches the protocol in
# PROJECT_PLAN.md §5.4 ("5 random query selections per target").
N_QUERY_DRAWS = 5


def _seeded_rng(dataset_id: str, draw_idx: int) -> np.random.Generator:
    """Deterministic RNG keyed on (global SEED, dataset, draw index).

    Stable across machines/processes so reruns reproduce the same queries.
    """
    h = hashlib.sha256(f"{SEED}|{dataset_id}|{draw_idx}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


class Exp3bMUV:
    id = "EXP-3b"
    version = "v2"
    datasets = MUV_DATASET_IDS

    metric_direction = {
        "M-AUCROC":   +1,
        "M-BEDROC20": +1,
        "M-EF1":      +1,
    }

    def run(self, method, stage_data, embeddings, dataset_id, out):
        df: pd.DataFrame = stage_data[Stage.SMILES].copy()
        df["id"] = df["id"].astype(str)
        out.mkdir(parents=True, exist_ok=True)

        mol_ids: list[str] = df["id"].tolist()
        activity: dict[str, int] = dict(zip(df["id"], df["activity"].astype(int)))
        active_ids: list[str] = [m for m in mol_ids if activity[m] == 1]
        n_actives_total = len(active_ids)

        retrieval_rows: list[dict] = []
        seed_aucroc: list[float] = []
        seed_bedroc: list[float] = []
        seed_ef1:    list[float] = []

        if active_ids:
            draw_queries: list[str] = [
                str(_seeded_rng(dataset_id, d).choice(active_ids))
                for d in range(N_QUERY_DRAWS)
            ]
            # One batched matrix call: (5 queries × n mols). Pool overhead is
            # paid once per target instead of five times, and full-mode MUV
            # (~15k mols × 5 queries = 75k pairs) crosses the parallelism
            # threshold cleanly.
            D = pairwise_matrix(method, embeddings, draw_queries, mol_ids)
            mol_col_of = {m: j for j, m in enumerate(mol_ids)}
        else:
            draw_queries = []
            D = None

        for draw_idx in range(N_QUERY_DRAWS):
            if not active_ids:
                seed_aucroc.append(float("nan"))
                seed_bedroc.append(float("nan"))
                seed_ef1.append(float("nan"))
                continue

            query_id = draw_queries[draw_idx]
            self_col = mol_col_of[query_id]
            row = D[draw_idx]
            cand_mask = np.arange(len(mol_ids)) != self_col
            candidates = [mol_ids[j] for j in range(len(mol_ids)) if j != self_col]
            distances = row[cand_mask]
            order = np.argsort(distances, kind="stable")

            n_total = len(candidates)
            n_actives = n_actives_total - 1  # query removed

            active_ranks: list[int] = []
            for rank_0, idx in enumerate(order):
                cid = candidates[idx]
                if activity[cid] == 1:
                    rank = rank_0 + 1
                    active_ranks.append(rank)
                    retrieval_rows.append({
                        "seed": draw_idx,
                        "query_id": query_id,
                        "active_id": cid,
                        "rank": rank,
                        "distance": float(distances[idx]),
                        "n_total": n_total,
                        "n_actives_in_pool": n_actives,
                    })

            # AUC-ROC over the full pool: scores are negative distances so
            # higher score = closer = more active-like.
            cand_labels = [activity[c] for c in candidates]
            if len(set(cand_labels)) >= 2:
                seed_aucroc.append(float(roc_auc_score(cand_labels, -distances)))
            else:
                seed_aucroc.append(float("nan"))
            seed_bedroc.append(rc.bedroc(active_ranks, n_total, n_actives))
            seed_ef1.append(rc.ef_at_percent(active_ranks, n_total, n_actives, percent=1.0))

        pd.DataFrame(retrieval_rows, columns=list(rc.RETRIEVAL_COLS)).to_csv(
            out / "retrieval_rankings.csv", index=False)
        rc.write_enrichment_summary(out, out / "retrieval_rankings.csv")

        metrics: dict = {}
        metrics.update(rc.metric_entry("M-AUCROC",   seed_aucroc))
        metrics.update(rc.metric_entry("M-BEDROC20", seed_bedroc))
        metrics.update(rc.metric_entry("M-EF1",      seed_ef1))

        (out / "metrics.json").write_text(rc.metrics_to_json(metrics))

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
                               metrics=list(self.metric_direction.keys()))
        write_manifest(sentinel, version=self.version, inputs=input_hashes, compute_time=0.0, dataset_size=0)
