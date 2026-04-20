"""B4: MACCS keys (166 bits) + Tanimoto distance."""
from __future__ import annotations

from typing import Any

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

from ...data.base import Stage


class MACCSKeys:
    id = "B4"
    version = "maccs-166-v1"
    needs = Stage.SMILES

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        df: pd.DataFrame = stage_data[Stage.SMILES]
        out: dict[str, Any] = {}
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                raise ValueError(f"unparseable SMILES for id={row['id']}: {row['smiles']!r}")
            out[str(row["id"])] = MACCSkeys.GenMACCSKeys(mol)
        return out

    def distance(self, e1: Any, e2: Any) -> float:
        return 1.0 - DataStructs.TanimotoSimilarity(e1, e2)
