"""Default 3D conformer generation used for datasets without native conformers.

Produces a pickled dict[mol_id -> rdkit.Chem.Mol] where each Mol has exactly
one conformer: the lowest MMFF94 energy geometry found across N_CONFS sampled
starting geometries. All 3D methods use this single conformer.

When this module's VERSION changes, the conformer stage becomes stale for
every dataset with has_native_conformers=False, cascading rebuilds downstream.

Per-molecule work is parallelised across N_WORKERS processes — each molecule
is independent (deterministic given SMILES + SEED), so embedding + MMFF
optimisation scale linearly with cores.
"""
from __future__ import annotations

import multiprocessing
import os
import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm


VERSION = "etkdgv3-mmff94-n5-lowest-energy-v1"
N_CONFS = 5
PRUNE_RMS_THRESH = 0.5
SEED = 0xEDDDE
N_WORKERS = os.cpu_count() or 1


def _keep_lowest_energy_conformer(mol: Chem.Mol) -> Chem.Mol:
    results = AllChem.MMFFOptimizeMoleculeConfs(mol)
    # results is list of (not_converged, energy); not_converged == -1 means MMFF
    # could not be parameterised for that conformer — exclude from energy ranking.
    valid = [(i, e) for i, (nc, e) in enumerate(results) if nc != -1]
    candidates = valid if valid else list(enumerate(r[1] for r in results))
    best_idx = min(candidates, key=lambda ie: ie[1])[0]
    best_conf = mol.GetConformer(best_idx)
    new_mol = Chem.RWMol(mol)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(best_conf, assignId=True)
    return new_mol.GetMol()


def _embed_one(args: tuple[str, str]) -> tuple[str, Chem.Mol | None]:
    """Worker: embed one molecule and return its lowest-energy conformer (or None on failure)."""
    mol_id, smiles = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES for id={mol_id}: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = PRUNE_RMS_THRESH
    params.randomSeed = SEED
    AllChem.EmbedMultipleConfs(mol, numConfs=N_CONFS, params=params)
    if mol.GetNumConformers() == 0:
        return mol_id, None
    return mol_id, _keep_lowest_energy_conformer(mol)


def generate(smiles_csv: Path, out: Path) -> None:
    df = pd.read_csv(smiles_csv)
    items = [(str(row["id"]), row["smiles"]) for _, row in df.iterrows()]

    mols: dict[str, Chem.Mol] = {}
    skipped: list[str] = []

    if items:
        n_workers = max(1, min(N_WORKERS, len(items)))
        with multiprocessing.Pool(n_workers) as pool:
            with tqdm(total=len(items), desc="conformers", unit="mol") as pbar:
                for mol_id, mol in pool.imap_unordered(_embed_one, items, chunksize=1):
                    if mol is None:
                        skipped.append(mol_id)
                    else:
                        mols[mol_id] = mol
                    pbar.update(1)

    if skipped:
        preview = ", ".join(skipped[:10]) + ("..." if len(skipped) > 10 else "")
        print(f"  [conformers] skipped {len(skipped)} molecule(s) where embedding produced 0 conformers: {preview}")
        from ..cache import append_to_blacklist
        append_to_blacklist(out.parent, skipped)

    with open(out, "wb") as f:
        pickle.dump(mols, f)
