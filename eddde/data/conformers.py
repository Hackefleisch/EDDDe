"""Default 3D conformer generation used for datasets without native conformers.

Produces a pickled dict[mol_id -> rdkit.Chem.Mol] where each Mol has exactly
one conformer: the lowest MMFF94 energy geometry found across N_CONFS sampled
starting geometries. All 3D methods use this single conformer.

When this module's VERSION changes, the conformer stage becomes stale for
every dataset with has_native_conformers=False, cascading rebuilds downstream.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


VERSION = "etkdgv3-mmff94-n20-lowest-energy-v1"
N_CONFS = 20
PRUNE_RMS_THRESH = 0.5
SEED = 0xEDDDE


def _keep_lowest_energy_conformer(mol: Chem.Mol) -> Chem.Mol:
    results = AllChem.MMFFOptimizeMoleculeConfs(mol)
    # results is a list of (not_converged, energy) per conformer
    best_idx = min(range(len(results)), key=lambda i: results[i][1])
    best_conf = mol.GetConformer(best_idx)
    new_mol = Chem.RWMol(mol)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(best_conf, assignId=True)
    return new_mol.GetMol()


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
        mols[str(row["id"])] = _keep_lowest_energy_conformer(mol)

    with open(out, "wb") as f:
        pickle.dump(mols, f)
