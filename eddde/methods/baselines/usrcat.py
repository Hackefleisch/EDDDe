"""B10: USRCAT -- 60-d pharmacophoric extension of USR.

Same four reference points as B9 USR, but atom-to-reference distance
distributions are computed separately for five pharmacophoric subsets
(all heavy atoms, hydrophobic, aromatic, H-bond acceptor, H-bond donor),
each condensed to 3 moments -> 5 * 4 * 3 = 60 floats.

Distance: inverse-Manhattan similarity (Schreyer & Blundell 2012, J.
Cheminform. 4, 27), expressed as 1 - similarity so smaller = more similar
and consistent with B9. Uses RDKit's `GetUSRCAT`.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from scipy.spatial.distance import cdist

from ...data.base import Stage
from ..base import Method


class USRCAT(Method):
    id = "B10"
    version = "usrcat-rdkit-heavy-v1"
    needs = Stage.CONFORMERS

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        conformers: dict[str, Chem.Mol] = stage_data[Stage.CONFORMERS]
        out: dict[str, Any] = {}
        for mol_id, mol in conformers.items():
            mol_no_h = Chem.RemoveHs(mol)
            usrcat_vec = rdMolDescriptors.GetUSRCAT(mol_no_h)
            out[mol_id] = np.asarray(usrcat_vec, dtype=float)
        return out

    def distances(self, embs_q: list[Any], embs_c: list[Any]) -> np.ndarray:
        # GetUSRScore(a, b) = 1 / (1 + (1/n) * sum|a - b|), n = 60 for USRCAT
        # with uniform weights across the 5 pharmacophore channels.
        # Distance = 1 - sim = (d/n) / (1 + d/n) where d is Manhattan distance.
        Q = np.stack(embs_q)
        C = np.stack(embs_c)
        n = Q.shape[1]
        mean_abs = cdist(Q, C, "cityblock") / n
        return mean_abs / (1.0 + mean_abs)
