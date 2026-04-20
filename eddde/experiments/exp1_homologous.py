"""EXP-1: Homologous Series Smoothness (PROJECT_PLAN.md §5.1).

Metrics (§6.1):
  M-MONO   — Spearman ρ between |i−j| and d(i,j) over all pairs. Range [-1,1]; 1 = perfect.
  M-SMOOTH — Std dev of consecutive-distance ratios d(k,k+1)/d(k-1,k). Lower is smoother.
  M-LIN    — R² of linear regression of d(mol_1, mol_k) vs k.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

from ..data.base import Stage


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


class Exp1Homologous:
    id = "EXP-1"
    version = "v4-mlin-pvalue"
    datasets = ["S1", "S2"]

    def run(self, method, stage_data, embeddings, dataset_id, out):
        df: pd.DataFrame = (
            stage_data[Stage.SMILES].sort_values("position").reset_index(drop=True)
        )
        mol_ids = [str(row.id) for row in df.itertuples(index=False)]
        positions = [int(row.position) for row in df.itertuples(index=False)]

        # All pairs — used for M-MONO
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

        # Consecutive distances d(k, k+1) — used for M-SMOOTH
        consec = [
            method.distance(embeddings[mol_ids[i]], embeddings[mol_ids[i + 1]])
            for i in range(len(mol_ids) - 1)
        ]

        # Distances from first molecule d(mol_1, mol_k) — used for M-LIN
        from_first = [
            method.distance(embeddings[mol_ids[0]], embeddings[mol_ids[k]])
            for k in range(1, len(mol_ids))
        ]

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
