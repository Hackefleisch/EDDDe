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
  retrieval_rankings.csv
    seed, query_id, active_id, rank, distance, n_total, n_actives_in_pool
    One row per (seed, query, active-in-pool). Pool = valid + test.
    n_total is the pool size; n_actives_in_pool is the number of actives
    in the pool, i.e. the maximum achievable recall.

  knn_neighbors.csv
    seed, test_id, test_activity, neighbor_rank, neighbor_id,
    neighbor_activity, neighbor_distance
    One row per (seed, test molecule, neighbor rank 1..K_MAX).
    Neighbors are drawn from the same valid+test pool as Task 1.

  metrics.json
    Per-metric mean and SE over the 5 scaffold seeds. Each seed value is
    itself the mean over all queries in that seed (Task 1) or all test
    molecules (Task 2).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..data.base import Stage
from ..data.sources.welqrate import N_SCAFFOLD_SEEDS

WELQRATE_DATASET_IDS = [
    "AID1798", "AID1843", "AID2258", "AID2689",
    "AID435008", "AID435034", "AID463087", "AID485290", "AID488997",
]

K_VALUES = (5, 10, 20)
K_MAX = max(K_VALUES)


# ---------------------------------------------------------------------------
# Metric helpers — all operate on the compact active-rank representation.
# active_ranks: list/array of 1-indexed ranks of the active molecules in the
#               sorted pool (lower = closer to query).
# n_total:      pool size (excluding the query itself).
# n_actives:    number of actives in the pool (excluding the query).
# ---------------------------------------------------------------------------

def _logauc(active_ranks, n_total: int, n_actives: int, min_fpr: float = 1e-3) -> float:
    """Log-AUC over [min_fpr, 1.0] (Truchon & Bayly 2007)."""
    ni = n_total - n_actives
    if n_actives == 0 or ni <= 0:
        return float("nan")

    ranks = np.sort(np.asarray(active_ranks, dtype=float))
    k = len(ranks)

    # After the i-th active (0-indexed): TP = i+1, FP = ranks[i] - (i+1).
    tp = np.arange(1, k + 1, dtype=float)
    fpr_after = (ranks - tp) / ni   # FPR is unchanged at active positions
    tpr_after = tp / n_actives

    # ROC consists of horizontal segments (FPR increases, TPR constant).
    # Segment i goes from fpr_points[i] to fpr_points[i+1] at tpr_points[i]:
    #   i=0   : [0, fpr_after[0]]         TPR = 0          → contributes 0
    #   i=1..k: [fpr_after[i-1], fpr_after[i]]  TPR = tpr_after[i-1]
    #   i=k+1 : [fpr_after[k-1], 1.0]     TPR = 1.0
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


def _bedroc(active_ranks, n_total: int, n_actives: int, alpha: float = 20.0) -> float:
    """BEDROC_α (Truchon & Bayly 2007)."""
    if n_actives == 0 or n_total == 0:
        return float("nan")
    N = n_total
    Ra = n_actives / N
    ranks = np.asarray(active_ranks, dtype=float)

    S = float(np.sum(np.exp(-alpha * ranks / N)))
    # Normalisation: RIE = S * (e^{α/N} - 1) / (Ra * (1 - e^{-α}))
    C = (math.exp(alpha / N) - 1.0) / (Ra * (1.0 - math.exp(-alpha)))
    rie = S * C
    rie_max = (1.0 - math.exp(-alpha * Ra)) / (Ra * (1.0 - math.exp(-alpha)))
    rie_min = math.exp(-alpha * (1.0 - Ra)) * rie_max
    if rie_max == rie_min:
        return float("nan")
    return (rie - rie_min) / (rie_max - rie_min)


def _ef_at_percent(active_ranks, n_total: int, n_actives: int, percent: float = 1.0) -> float:
    """Enrichment factor at `percent`% of the ranked list."""
    if n_actives == 0:
        return float("nan")
    cutoff = math.ceil(n_total * percent / 100.0)
    tp = sum(1 for r in active_ranks if r <= cutoff)
    return tp / (n_actives * percent / 100.0)


def _dcg_at_k(active_ranks, k: int = 100) -> float:
    """Discounted Cumulative Gain at rank k (binary relevance)."""
    return sum(1.0 / math.log2(r + 1) for r in active_ranks if r <= k)


def _mean_se(values: list[float]) -> tuple[float, float]:
    """Mean and standard error, ignoring NaN."""
    v = [x for x in values if not math.isnan(x)]
    if not v:
        return float("nan"), float("nan")
    m = float(np.mean(v))
    se = float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
    return m, se


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class Exp3aWelQrate:
    id = "EXP-3a"
    version = "v3"
    datasets = WELQRATE_DATASET_IDS

    metric_direction = {
        "M-LOGAUC":    +1,
        "M-BEDROC20":  +1,
        "M-EF1":       +1,
        "M-DCG100":    +1,
        "M-AUCROC_k5": +1,
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

        # Per-seed metric accumulators for Task 1 (each entry = mean over queries)
        seed_logauc:   list[float] = []
        seed_bedroc:   list[float] = []
        seed_ef1:      list[float] = []
        seed_dcg100:   list[float] = []
        # Per-seed AUC-ROC for Task 2, keyed by k
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

                # Collect active ranks (compact output)
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

                q_logauc.append(_logauc(active_ranks, n_total, n_actives))
                q_bedroc.append(_bedroc(active_ranks, n_total, n_actives))
                q_ef1.append(_ef_at_percent(active_ranks, n_total, n_actives, percent=1.0))
                q_dcg100.append(_dcg_at_k(active_ranks, k=100))

            seed_logauc.append(float(np.nanmean(q_logauc)) if q_logauc else float("nan"))
            seed_bedroc.append(float(np.nanmean(q_bedroc)) if q_bedroc else float("nan"))
            seed_ef1.append(float(np.nanmean(q_ef1)) if q_ef1 else float("nan"))
            seed_dcg100.append(float(np.nanmean(q_dcg100)) if q_dcg100 else float("nan"))

            # ------------------------------------------------------------------
            # Task 2: k-NN classification
            # ------------------------------------------------------------------
            knn_by_test: dict[str, list[int]] = {}  # test_id -> neighbor activities (top K_MAX)

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

            # AUC-ROC at each k
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

        # ------------------------------------------------------------------
        # Save raw outputs
        # ------------------------------------------------------------------
        pd.DataFrame(retrieval_rows).to_csv(out / "retrieval_rankings.csv", index=False)
        pd.DataFrame(knn_rows).to_csv(out / "knn_neighbors.csv", index=False)

        # ------------------------------------------------------------------
        # Aggregate metrics across seeds (mean ± SE)
        # ------------------------------------------------------------------
        def _entry(key, values):
            m, se = _mean_se(values)
            return {key: m, f"{key}_se": se}

        metrics: dict = {}
        metrics.update(_entry("M-LOGAUC",   seed_logauc))
        metrics.update(_entry("M-BEDROC20", seed_bedroc))
        metrics.update(_entry("M-EF1",      seed_ef1))
        metrics.update(_entry("M-DCG100",   seed_dcg100))
        for k in K_VALUES:
            metrics.update(_entry(f"M-AUCROC_k{k}", seed_aucroc[k]))

        (out / "metrics.json").write_text(
            json.dumps(
                {key: (None if isinstance(v, float) and math.isnan(v) else v)
                 for key, v in metrics.items()},
                indent=2,
                sort_keys=True,
            )
        )
        return metrics
