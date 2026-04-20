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
from scipy.spatial.distance import cosine

from ...data.base import Stage


class RDKitDescriptors:
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

    def distance(self, e1: Any, e2: Any) -> float:
        norm1, norm2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0 if np.array_equal(e1, e2) else 1.0
        return float(cosine(e1, e2))
