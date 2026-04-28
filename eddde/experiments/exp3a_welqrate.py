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
          regardless, and the kNN task (Task 2) uses train-only neighbors
          to stay conservative.

      Record the rank and distance of every active molecule in the pool
      (compact: skip inactive ranks). Sufficient to later compute
      LogAUC, BEDROC20, EF1%, DCG100.

    Task 2 — k-NN classification:
      For each test-split molecule (active or inactive), find its
      K_MAX nearest neighbors within the same valid+test pool (minus
      the molecule itself). Consistent with Task 1. Sufficient for
      AUC-ROC at k=5,10,20.

Output files written to `out/`:
  retrieval_rankings.csv
    seed, query_id, active_id, rank, distance, n_total, n_actives_in_pool
    One row per (seed, query, active-in-pool). Pool = valid + test.
    n_total is the pool size; n_actives_in_pool is the number of actives
    in the pool, i.e. the maximum achievable recall.

  knn_neighbors.csv
    seed, test_id, test_activity, neighbor_rank, neighbor_id,
    neighbor_activity, neighbor_distance
    One row per (seed, test molecule, neighbor rank 1..K_MAX).
    Neighbors are drawn from the same valid+test pool as Task 1
    (minus the molecule being classified).

  metrics.json  — empty dict for now; metrics added in a later pass.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.base import Stage
from ..data.sources.welqrate import N_SCAFFOLD_SEEDS

WELQRATE_DATASET_IDS = [
    "AID1798", "AID1843", "AID2258", "AID2689",
    "AID435008", "AID435034", "AID463087", "AID485290", "AID488997",
]

# Neighbors stored per test molecule for kNN classification (k = 5, 10, 20).
K_MAX = 20


class Exp3aWelQrate:
    id = "EXP-3a"
    version = "v2"
    datasets = WELQRATE_DATASET_IDS

    def run(self, method, stage_data, embeddings, dataset_id, out):
        df: pd.DataFrame = stage_data[Stage.SMILES].copy()
        df["id"] = df["id"].astype(str)
        out.mkdir(parents=True, exist_ok=True)

        mol_ids: list[str] = df["id"].tolist()
        activity: dict[str, int] = dict(zip(df["id"], df["activity"].astype(int)))

        retrieval_rows: list[dict] = []
        knn_rows: list[dict] = []

        for seed in range(1, N_SCAFFOLD_SEEDS + 1):
            split_col = f"scaffold_seed{seed}"
            split: dict[str, str] = df.set_index("id")[split_col].to_dict()

            test_ids        = [m for m in mol_ids if split[m] == "test"]
            test_active_ids = [m for m in test_ids if activity[m] == 1]

            # ------------------------------------------------------------------
            # Task 1: retrieval
            # Pool = valid + test (all non-train molecules).
            # ------------------------------------------------------------------
            pool_ids = [m for m in mol_ids if split[m] in ("valid", "test")]
            n_actives_in_pool_base = sum(activity[c] for c in pool_ids)

            for query_id in test_active_ids:
                # Remove the query itself from candidates.
                candidates = [m for m in pool_ids if m != query_id]
                distances = np.array([
                    method.distance(embeddings[query_id], embeddings[cid])
                    for cid in candidates
                ])
                order = np.argsort(distances, kind="stable")

                n_total = len(candidates)
                # One active (the query) is removed from the pool count.
                n_actives_in_pool = n_actives_in_pool_base - 1

                for rank_0, idx in enumerate(order):
                    cid = candidates[idx]
                    if activity[cid] == 1:
                        retrieval_rows.append({
                            "seed": seed,
                            "query_id": query_id,
                            "active_id": cid,
                            "rank": rank_0 + 1,
                            "distance": float(distances[idx]),
                            "n_total": n_total,
                            "n_actives_in_pool": n_actives_in_pool,
                        })

            # ------------------------------------------------------------------
            # Task 2: k-NN classification
            # Same pool as Task 1 (valid+test, minus the molecule itself).
            # ------------------------------------------------------------------
            for test_id in test_ids:
                candidates = [m for m in pool_ids if m != test_id]
                distances = np.array([
                    method.distance(embeddings[test_id], embeddings[cid])
                    for cid in candidates
                ])
                order = np.argsort(distances, kind="stable")
                for rank_0 in range(min(K_MAX, len(candidates))):
                    neighbor_id = candidates[order[rank_0]]
                    knn_rows.append({
                        "seed": seed,
                        "test_id": test_id,
                        "test_activity": activity[test_id],
                        "neighbor_rank": rank_0 + 1,
                        "neighbor_id": neighbor_id,
                        "neighbor_activity": activity[neighbor_id],
                        "neighbor_distance": float(distances[order[rank_0]]),
                    })

        pd.DataFrame(retrieval_rows).to_csv(out / "retrieval_rankings.csv", index=False)
        pd.DataFrame(knn_rows).to_csv(out / "knn_neighbors.csv", index=False)
        (out / "metrics.json").write_text(json.dumps({}))
        return {}
