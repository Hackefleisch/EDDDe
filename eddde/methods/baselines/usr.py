"""B9: Ultrafast Shape Recognition (USR) -- 12-d shape descriptor.

Per-molecule embedding: distributions of atom-to-reference-point distances
condensed into 3 moments per reference point, 4 reference points -> 12 floats.
Computed on heavy atoms of the shared single conformer.

Distance: inverse-Manhattan similarity (Ballester & Richards, J. Comput. Chem.
2007, 28, 1711), expressed as 1 -> similarity so smaller = more similar.
Uses RDKit's GetUSR / GetUSRScore.

Smoke test: `python -m eddde.methods.baselines.usr`
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


if __name__ == "__main__":
    from rdkit.Chem import AllChem

    probes = {"benzene": "c1ccccc1", "toluene": "Cc1ccccc1", "cyclohexane": "C1CCCCC1"}
    method = USR()

    mols: dict[str, Chem.Mol] = {}
    for name, smi in probes.items():
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, randomSeed=0xEDDDE)
        AllChem.MMFFOptimizeMolecule(mol)
        mols[name] = mol

    embeddings = method.embed_dataset({Stage.CONFORMERS: mols})

    print("USR embeddings (12-d):")
    for name, vec in embeddings.items():
        print(f"  {name:>11s}: {np.array2string(vec, precision=3, suppress_small=True)}")

    print("\nPairwise distances (1 - GetUSRScore):")
    names = list(embeddings)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1 :]:
            d = method.distance(embeddings[n1], embeddings[n2])
            print(f"  d({n1:>11s}, {n2:>11s}) = {d:.4f}")
