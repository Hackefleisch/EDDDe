"""EXP-1: Homologous Series Smoothness (PROJECT_PLAN.md §5.1).

Current metric: M-MONO. M-SMOOTH and M-LIN to follow once we have more
than one series wired up.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from ..data.base import Stage


class Exp1Homologous:
    id = "EXP-1"
    version = "v1-mmono"
    datasets = ["S1"]

    def run(self, method, stage_data, embeddings, dataset_id, out):
        df: pd.DataFrame = (
            stage_data[Stage.SMILES].sort_values("position").reset_index(drop=True)
        )
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
        metrics = {"M-MONO": float(rho), "M-MONO_pvalue": float(p)}
        (out / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True)
        )
        return metrics
