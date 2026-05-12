"""MUT-mean: mean-pool ElektroNN coefficients across atoms, Euclidean distance.

Condenses the per-atom (n_atoms, 127) coefficient matrix to a single 127-dim
vector by averaging over atoms. Distance is plain L2 between these vectors.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from ...data.base import Stage
from ..base import Method


class MutMean(Method):
    id = "MUT-mean"
    version = "mean-euclidean-v1"
    needs = Stage.ELEKTRONN_COEFFS

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        coefficients: dict[str, np.ndarray] = stage_data[Stage.ELEKTRONN_COEFFS]["coefficients"]
        return {mol_id: coeffs.mean(axis=0) for mol_id, coeffs in coefficients.items()}

    def distances(self, embs_q: list[Any], embs_c: list[Any]) -> np.ndarray:
        Q = np.stack(embs_q)
        C = np.stack(embs_c)
        return cdist(Q, C, "euclidean")
