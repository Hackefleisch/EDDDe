"""B7: RDKit 2D physicochemical descriptors + cosine distance.

Computes all ~200 RDKit descriptors (Descriptors.descList), replaces any
NaN/inf with 0, and stores as a float32 numpy array. Cosine distance is
used because descriptors span very different scales; it is scale-invariant
without requiring explicit standardisation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from scipy.spatial.distance import cdist

from ...data.base import Stage
from ..base import Method


class RDKitDescriptors(Method):
    id = "B7"
    version = "rdkit2d-cosine-v1"
    needs = Stage.SMILES

    def __init__(self) -> None:
        desc_names = [name for name, _ in Descriptors.descList]
        self._calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        df: pd.DataFrame = stage_data[Stage.SMILES]
        out: dict[str, Any] = {}
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                raise ValueError(f"unparseable SMILES for id={row['id']}: {row['smiles']!r}")
            vec = np.array(self._calc.CalcDescriptors(mol), dtype=np.float32)
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            out[str(row["id"])] = vec
        return out

    def distances(self, embs_q: list[Any], embs_c: list[Any]) -> np.ndarray:
        # Promote to float64 before cosine: a few RDKit descriptors (e.g. Ipc)
        # can take values ~1e30, and squaring them in float32 overflows
        # (max ~3.4e38). Embeddings stay float32 on disk; the cast is local.
        Q = np.stack(embs_q).astype(np.float64, copy=False)
        C = np.stack(embs_c).astype(np.float64, copy=False)
        # cdist returns nan for zero-norm rows; replicate the old per-pair
        # semantics by post-processing: zero-norm vs zero-norm → 0 if rows
        # are equal, 1 otherwise.
        D = cdist(Q, C, "cosine")
        bad = np.isnan(D)
        if bad.any():
            qi, cj = np.nonzero(bad)
            for i, j in zip(qi, cj):
                D[i, j] = 0.0 if np.array_equal(Q[i], C[j]) else 1.0
        return D
