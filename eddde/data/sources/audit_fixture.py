"""S0: equivariance audit fixture for MUTs (PROJECT_PLAN.md §3.2.1).

Five small molecules, each materialised as three variants:

  audit_{base}_orig   — canonical conformer (single MMFF94-lowest from 5 ETKDGv3)
  audit_{base}_rot    — same molecule with positions rotated by a fixed R ∈ SO(3)
  audit_{base}_perm   — same molecule with atoms renumbered by a fixed permutation

15 rows total. ElektroNN runs over all of them via the standard pipeline, so
the rotated/permuted coefficients are real ElektroNN output (not Wigner-derived
or otherwise simulated). The MUT base class reads these from cache at audit
time and asserts that the chosen MUT's embedding+distance produces:

  d(emb(orig), emb(rot))  < ε_rot
  d(emb(orig), emb(perm)) < ε_perm
  d(emb(orig), emb(orig)) < ε_id

No experiment references S0, so the standard runner cross-product never
embeds a non-MUT method on it — the only cost is one-time stage build for 15
molecules (cheap).

Base SMILES are picked to cover the relevant element subset (H, C, N, O, F, S)
with diverse bonding patterns (sp3/sp2/sp/aromatic, polar/nonpolar). Each base
passes the project-wide >=3-heavy-atom filter from [pipeline.py](../pipeline.py).
Cl is intentionally omitted — adding a sixth base costs another ElektroNN run
without changing what the audit covers.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial.transform import Rotation

from ... import SEED
from ..base import Dataset
from ..conformers import _embed_one


AUDIT_BASES: list[tuple[str, str]] = [
    ("ethanol",        "CCO"),
    ("acetonitrile",   "CC#N"),
    ("fluoroethane",   "CCF"),
    ("dimethylsulfide", "CSC"),
    ("benzene",        "c1ccccc1"),
]

VARIANTS = ("orig", "rot", "perm")


def base_ids() -> list[str]:
    return [base for base, _ in AUDIT_BASES]


def variant_id(base: str, variant: str) -> str:
    return f"audit_{base}_{variant}"


def _rotate_mol(mol: Chem.Mol, R: np.ndarray) -> Chem.Mol:
    new = Chem.Mol(mol)
    conf = new.GetConformer()
    n = new.GetNumAtoms()
    pos = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in range(n)])
    rotated = pos @ R.T
    for i in range(n):
        conf.SetAtomPosition(i, (float(rotated[i, 0]),
                                 float(rotated[i, 1]),
                                 float(rotated[i, 2])))
    return new


def _permute_mol(mol: Chem.Mol, rng: np.random.Generator) -> Chem.Mol:
    n = mol.GetNumAtoms()
    # Reject the identity permutation; otherwise the permutation test is a
    # tautology of the identity test.
    while True:
        order = rng.permutation(n).tolist()
        if order != list(range(n)):
            break
    return Chem.RenumberAtoms(mol, order)


class AuditFixture(Dataset):
    id = "S0"
    version = "v1"
    has_native_conformers = True

    def build_smiles(self, out: Path) -> None:
        rows = []
        for base, smi in AUDIT_BASES:
            for variant in VARIANTS:
                rows.append({"id": variant_id(base, variant), "smiles": smi})
        pd.DataFrame(rows).to_csv(out, index=False)

    def build_native_conformers(self, smiles_csv: Path, out: Path) -> None:
        # Generate the canonical conformer once per base SMILES. Reusing
        # `_embed_one` keeps the conformer-generation policy identical to the
        # default pipeline (5×ETKDGv3, lowest MMFF94), so the fixture inherits
        # the same determinism guarantees as every other dataset.
        base_to_mol: dict[str, Chem.Mol] = {}
        for base, smi in AUDIT_BASES:
            mol_id, mol = _embed_one((variant_id(base, "orig"), smi))
            if mol is None:
                raise RuntimeError(
                    f"S0 audit fixture: embedding failed for {mol_id} "
                    f"(SMILES={smi!r}); audit cannot run without a conformer."
                )
            base_to_mol[base] = mol

        rng = np.random.default_rng(SEED)
        R = Rotation.random(random_state=rng).as_matrix()

        df = pd.read_csv(smiles_csv)
        mols: dict[str, Chem.Mol] = {}
        for _, row in df.iterrows():
            mol_id = str(row["id"])
            base, _, variant = mol_id.removeprefix("audit_").rpartition("_")
            canonical = base_to_mol[base]
            if variant == "orig":
                mols[mol_id] = Chem.Mol(canonical)
            elif variant == "rot":
                mols[mol_id] = _rotate_mol(canonical, R)
            elif variant == "perm":
                mols[mol_id] = _permute_mol(canonical, rng)
            else:
                raise ValueError(f"unknown variant in mol_id={mol_id!r}")

        with open(out, "wb") as f:
            pickle.dump(mols, f)

    def test_mode_subsample(self, df, n, rng):
        return df
