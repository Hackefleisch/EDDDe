"""B9: Ultrafast Shape Recognition (USR) -- 12-d shape descriptor.

Per-molecule embedding: distributions of atom-to-reference-point distances
condensed into 3 moments per reference point, 4 reference points -> 12 floats.
Computed on heavy atoms of the shared single conformer.

Distance: inverse-Manhattan similarity (Ballester & Richards, J. Comput. Chem.
2007, 28, 1711), expressed as 1 -> similarity so smaller = more similar.
Uses RDKit's GetUSR / GetUSRScore.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ...data.base import Stage


class USR:
    id = "B9"
    version = "usr-rdkit-heavy-v1"
    needs = Stage.CONFORMERS

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        conformers: dict[str, Chem.Mol] = stage_data[Stage.CONFORMERS]
        out: dict[str, Any] = {}
        for mol_id, mol in conformers.items():
            mol_no_h = Chem.RemoveHs(mol)
            usr_vec = rdMolDescriptors.GetUSR(mol_no_h)
            out[mol_id] = np.asarray(usr_vec, dtype=float)
        return out

    def distance(self, e1: Any, e2: Any) -> float:
        return 1.0 - rdMolDescriptors.GetUSRScore(e1.tolist(), e2.tolist())
