"""MUT-mean family: scalarised mean pool of ElektroNN coefficients.

Strain A's simplest variants (PROJECT_PLAN.md §3.2.2). Each atom's 127-d
ElektroNN feature vector is scalarised to a 39-d rotation-invariant vector
via `strain_a.scalarise` (l>0 multiplicities collapse to their 2-norms),
then averaged across atoms.

- `MUT-mean`        — Euclidean distance over the 39-d per-molecule embedding.
- `MUT-mean-cosine` — cosine distance over the same embedding. A/B against
  MUT-mean to test whether magnitude information in the per-atom invariants
  carries similarity signal or whether direction alone is sufficient.

Both share `embed_dataset`; only the distance differs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from ...data.base import Stage
from ..base import Method
from .strain_a import scalarise


class MutMean(Method):
    id = "MUT-mean"
    version = "mean-scalarised-euclidean-v2"
    needs = Stage.ELEKTRONN_COEFFS
    is_mut = True

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        coefficients: dict[str, np.ndarray] = stage_data[Stage.ELEKTRONN_COEFFS]["coefficients"]
        return {mol_id: scalarise(coeffs).mean(axis=0) for mol_id, coeffs in coefficients.items()}

    def distances(self, embs_q: list[Any], embs_c: list[Any]) -> np.ndarray:
        Q = np.stack(embs_q)
        C = np.stack(embs_c)
        return cdist(Q, C, "euclidean")


class MutMeanCosine(MutMean):
    id = "MUT-mean-cosine"
    version = "mean-scalarised-cosine-v1"

    def distances(self, embs_q: list[Any], embs_c: list[Any]) -> np.ndarray:
        Q = np.stack(embs_q)
        C = np.stack(embs_c)
        return cdist(Q, C, "cosine")
