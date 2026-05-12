"""Shared batched Tanimoto-distance helper for bitvector-fingerprint
baselines (B1 ECFP4, B2 ECFP6, B3 FCFP4, B4 MACCS, B5 AtomPair,
B6 TopologicalTorsion).

Per-pair `1 - TanimotoSimilarity(a, b)` reduces to a numpy subtraction
of `BulkTanimotoSimilarity(query, fp_list)`, which runs the inner loop
in RDKit C++ — typically 50-100x faster than a Python-level pair loop.
The function works on any RDKit fingerprint type accepted by
`BulkTanimotoSimilarity` (ExplicitBitVect, IntSparseIntVect, ...).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import DataStructs


def tanimoto_distance_matrix(embs_q: list[Any], embs_c: list[Any]) -> np.ndarray:
    """Tanimoto distances between query and candidate fingerprints.

    Returns shape (len(embs_q), len(embs_c)).
    """
    M = np.empty((len(embs_q), len(embs_c)), dtype=float)
    for i, q in enumerate(embs_q):
        sims = DataStructs.BulkTanimotoSimilarity(q, embs_c)
        M[i] = 1.0 - np.asarray(sims, dtype=float)
    return M
