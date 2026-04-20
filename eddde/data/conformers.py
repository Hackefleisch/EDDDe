"""Default 3D conformer generation used for datasets without native conformers.

Produces a pickled dict[mol_id -> rdkit.Chem.Mol with embedded conformers].
When this module's VERSION changes, the conformer stage becomes stale for
every dataset with has_native_conformers=False, cascading rebuilds downstream.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


VERSION = "etkdgv3-mmff94-n50-prune0.5-v1"
N_CONFS = 50
PRUNE_RMS_THRESH = 0.5
SEED = 0xEDDDE


def generate(smiles_csv: Path, out: Path) -> None:
    df = pd.read_csv(smiles_csv)
    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = PRUNE_RMS_THRESH
    params.randomSeed = SEED

    mols: dict[str, Chem.Mol] = {}
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            raise ValueError(f"unparseable SMILES for id={row['id']}: {row['smiles']!r}")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFS, params=params)
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        mols[str(row["id"])] = mol

    with open(out, "wb") as f:
        pickle.dump(mols, f)
