"""B4: MACCS keys (166 bits) + Tanimoto distance."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys

from ...data.base import Stage
from ..base import Method
from ._bitvec_tanimoto import tanimoto_distance_matrix


class MACCSKeys(Method):
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

    def distances(self, embs_q: list[Any], embs_c: list[Any]) -> np.ndarray:
        return tanimoto_distance_matrix(embs_q, embs_c)
