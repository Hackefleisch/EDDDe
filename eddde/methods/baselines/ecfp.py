"""B1: ECFP4 (Morgan radius 2, 2048 bits)
   B2: ECFP6 (Morgan radius 3, 2048 bits)
   B3: FCFP4 (feature-based Morgan radius 2, 2048 bits)
All use Tanimoto distance."""
from __future__ import annotations

from typing import Any

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from ...data.base import Stage


class _MorganFP:
    needs = Stage.SMILES

    def __init__(self, radius: int) -> None:
        self._gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=2048)

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


class ECFP4(_MorganFP):
    id = "B1"
    version = "morgan-r2-2048-v1"

    def __init__(self) -> None:
        super().__init__(radius=2)


class ECFP6(_MorganFP):
    id = "B2"
    version = "morgan-r3-2048-v1"

    def __init__(self) -> None:
        super().__init__(radius=3)


class FCFP4(_MorganFP):
    id = "B3"
    version = "morgan-feat-r2-2048-v1"

    def __init__(self) -> None:
        atom_inv = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048, atomInvariantsGenerator=atom_inv
        )
        # skip super().__init__ — generator is set directly
