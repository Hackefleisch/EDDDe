"""EXP-6: Scaffold Hopping (PROJECT_PLAN.md §5.8).

Question: Does a method retrieve actives with diverse scaffolds, not just
analogs of the query?

Two retrieval protocols rolled into one experiment, both per target:

Sub-experiment A — standard retrieval (all 37 ChEMBL_II targets)
  Draw 5 random query actives. Rank the entire pool (actives + decoys minus
  query) by ascending distance. Compute M-EF5, M-SCAFEF5, M-SCAFRATIO.

Sub-experiment B — leave-one-scaffold-out (targets with >=5 distinct active
                   scaffolds only)
  For each active scaffold S in target T:
    queries = all actives with S in their scaffold set
    pool    = decoys + actives with S NOT in their scaffold set
  Compute M-RECALL-HELDOUT-SCAFFOLD at top 1 %, 5 %, 10 %.

Scaffold partitions come pre-computed in the upstream Riniker-Landrum data
(see eddde/data/sources/rinlan.py); we inherit them rather than re-deriving
Bemis-Murcko, so the "scaffold" definition is the one the benchmark
intended. Membership is many-to-many (Schuffenhauer hierarchy: one active
can belong to multiple scaffold buckets), stored in the dataset's sidecar
`cache/datasets/<id>/scaffolds.json` as {chembl_id: [scaf_id, ...]}.

Status: Step 4 of scratch/exp6_implementation_plan.md. Sub-experiments A
covers M-EF5, M-SCAFEF5, and M-SCAFRATIO. Sub-experiment B (leave-one-
scaffold-out / M-RECALL-HELDOUT-SCAFFOLD) arrives in Step 5.

Output files written to `out/`:
  retrieval_standard.csv   (sub-experiment A, RETRIEVAL_COLS_EXP6 schema)
  metrics.json             (combined metric values)
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import SEED
from ..data.base import Stage, dataset_dir
from ..data.sources.rinlan import RINLAN_DATASET_IDS
from ..methods.distance import pairwise_matrix
from . import retrieval_common as rc
from .base import result_dir


# 5 random query actives per target — matches PROJECT_PLAN.md §5.8.
N_QUERY_DRAWS = 5

# Top-k percentage for the sub-experiment A enrichment factor. Wider than
# EXP-3a/3b's 1 % because Riniker-Landrum targets have far fewer molecules
# (~100-1500 vs WelQrate's tens of thousands); the top 1 % cutoff would land
# inside the first 5-10 ranks for small targets and bake in undue variance.
EF_PERCENT_STANDARD = 5.0


# Extended schema: retrieval_common.RETRIEVAL_COLS plus the scaffold sets of
# the retrieved active and the query, JSON-encoded so SCAFEF/SCAFRATIO are
# recomputable straight from the CSV without re-loading scaffolds.json.
RETRIEVAL_COLS_EXP6 = (
    *rc.RETRIEVAL_COLS,
    "scaffold_ids",
    "query_scaffold_ids",
)


def _seeded_rng(dataset_id: str, draw_idx: int) -> np.random.Generator:
    """Deterministic RNG keyed on (global SEED, dataset, draw index).

    Stable across machines/processes so reruns reproduce the same queries.
    Same pattern as exp3b_muv._seeded_rng.
    """
    h = hashlib.sha256(f"{SEED}|{dataset_id}|{draw_idx}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _load_scaffold_map(dataset_id: str) -> dict[str, list[int]]:
    """Read the dataset's scaffolds.json sidecar (per-active scaffold lists).

    Returns an empty dict if the sidecar is missing — keeps the experiment
    safe to instantiate on datasets without scaffold annotation.
    """
    p = dataset_dir(dataset_id) / "scaffolds.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


class Exp6ScaffoldHopping:
    id = "EXP-6"
    version = "v0.3-scafef"
    datasets = RINLAN_DATASET_IDS

    metric_direction = {
        "M-EF5":                     +1,
        "M-SCAFEF5":                 +1,
        "M-SCAFRATIO":               +1,
        "M-RECALL-HELDOUT-SCAFFOLD": +1,
    }

    # metric_datasets stays unset for now: M-EF5/SCAFEF5/SCAFRATIO apply to
    # every target, M-RECALL-HELDOUT-SCAFFOLD will gain a per-target
    # eligibility map (≥5 distinct active scaffolds) in Step 5. SUMMARY.md
    # treats absent metrics as "—".

    def run(self, method, stage_data, embeddings, dataset_id: str, out: Path) -> dict[str, Any]:
        df: pd.DataFrame = stage_data[Stage.SMILES].copy()
        df["id"] = df["id"].astype(str)
        out.mkdir(parents=True, exist_ok=True)

        mol_ids: list[str] = df["id"].tolist()
        activity: dict[str, int] = dict(zip(df["id"], df["activity"].astype(int)))
        active_ids: list[str] = [m for m in mol_ids if activity[m] == 1]
        n_actives_total = len(active_ids)

        scaffold_map = _load_scaffold_map(dataset_id)

        retrieval_rows: list[dict] = []
        seed_ef5: list[float] = []
        seed_scafef5: list[float] = []
        seed_scafratio: list[float] = []

        if active_ids:
            # Independent seeded draws (collisions are vanishingly rare at
            # typical ChEMBL_II active counts of 100-1000 per target; matches
            # the MUV pattern for deterministic reproducibility).
            draw_queries: list[str] = [
                str(_seeded_rng(dataset_id, d).choice(active_ids))
                for d in range(N_QUERY_DRAWS)
            ]
            # One batched pairwise_matrix call: (N_QUERY_DRAWS × n_pool).
            # The full pool includes the query itself; we slice out its
            # self-column from each row below.
            D = pairwise_matrix(method, embeddings, draw_queries, mol_ids)
            mol_col_of = {m: j for j, m in enumerate(mol_ids)}
        else:
            draw_queries = []
            D = None
            mol_col_of = {}

        for draw_idx in range(N_QUERY_DRAWS):
            if not active_ids:
                seed_ef5.append(float("nan"))
                seed_scafef5.append(float("nan"))
                seed_scafratio.append(float("nan"))
                continue

            query_id = draw_queries[draw_idx]
            query_scaffolds = scaffold_map.get(query_id, [])
            self_col = mol_col_of[query_id]
            row = D[draw_idx]
            cand_mask = np.arange(len(mol_ids)) != self_col
            candidates = [mol_ids[j] for j in range(len(mol_ids)) if j != self_col]
            distances = row[cand_mask]
            order = np.argsort(distances, kind="stable")

            n_total = len(candidates)
            n_actives = n_actives_total - 1  # query removed

            # Per-scaffold active counts over THIS query's pool (query removed
            # so its scaffolds don't inflate the random baseline).
            scaffold_active_counts: dict[int, int] = defaultdict(int)
            for cid in active_ids:
                if cid == query_id:
                    continue
                for s in scaffold_map.get(cid, []):
                    scaffold_active_counts[s] += 1

            cutoff = max(1, int(np.ceil(n_total * EF_PERCENT_STANDARD / 100.0)))
            retrieved_scaffold_sets: list[list[int]] = []

            active_ranks: list[int] = []
            for rank_0, idx in enumerate(order):
                cid = candidates[idx]
                if activity[cid] == 1:
                    rank = rank_0 + 1
                    active_ranks.append(rank)
                    cid_scaf = scaffold_map.get(cid, [])
                    if rank <= cutoff:
                        retrieved_scaffold_sets.append(cid_scaf)
                    retrieval_rows.append({
                        "seed": draw_idx,
                        "query_id": query_id,
                        "active_id": cid,
                        "rank": rank,
                        "distance": float(distances[idx]),
                        "n_total": n_total,
                        "n_actives_in_pool": n_actives,
                        "scaffold_ids": json.dumps(cid_scaf),
                        "query_scaffold_ids": json.dumps(query_scaffolds),
                    })

            ef5 = rc.ef_at_percent(active_ranks, n_total, n_actives,
                                   percent=EF_PERCENT_STANDARD)
            scafef5 = rc.scafef_at_percent(retrieved_scaffold_sets,
                                           scaffold_active_counts,
                                           n_total,
                                           percent=EF_PERCENT_STANDARD)
            # M-SCAFRATIO: per spec §5.2 it's literally SCAFEF / EF — measures
            # whether the scaffolds-recovered enrichment outpaces the actives-
            # recovered one (i.e. recovered actives are more scaffold-diverse
            # than chance). Per-draw division, then mean across draws — keeps
            # the SE behaviour consistent with the other metrics.
            if ef5 != 0.0 and not (np.isnan(ef5) or np.isnan(scafef5)):
                scafratio = scafef5 / ef5
            else:
                scafratio = float("nan")

            seed_ef5.append(ef5)
            seed_scafef5.append(scafef5)
            seed_scafratio.append(scafratio)

        pd.DataFrame(retrieval_rows, columns=list(RETRIEVAL_COLS_EXP6)).to_csv(
            out / "retrieval_standard.csv", index=False)

        metrics: dict = {}
        metrics.update(rc.metric_entry("M-EF5",       seed_ef5))
        metrics.update(rc.metric_entry("M-SCAFEF5",   seed_scafef5))
        metrics.update(rc.metric_entry("M-SCAFRATIO", seed_scafratio))

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
                        rows.append({"method": m_id, "dataset": ds_id,
                                     "metric": metric, "value": data[metric]})
        return pd.DataFrame(rows)
