"""B6: Topological Torsion fingerprint + Tanimoto distance."""
from __future__ import annotations

from typing import Any

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from ...data.base import Stage


class TopologicalTorsion:
    id = "B6"
    version = "toptorsion-2048-v1"
    needs = Stage.SMILES

    def __init__(self) -> None:
        self._gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        df: pd.DataFrame = stage_data[Stage.SMILES]
        out: dict[str, Any] = {}
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                raise ValueError(f"unparseable SMILES for id={row['id']}: {row['smiles']!r}")
            out[str(row["id"])] = self._gen.GetFingerprint(mol)
        return out

    def distance(self, e1: Any, e2: Any) -> float:
        return 1.0 - DataStructs.TanimotoSimilarity(e1, e2)
