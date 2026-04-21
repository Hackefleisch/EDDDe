"""S8: para-substituted benzoic acids — Hammett series (PROJECT_PLAN.md §5.2 / EXP-2).

σ_para values from Hansch, Leo & Taft (1991) Chem. Rev. 91, 165–195.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..base import Dataset

# (substituent_label, SMILES, sigma_para)
_COMPOUNDS = [
    ("NH2",  "Nc1ccc(C(=O)O)cc1",          -0.66),
    ("OCH3", "COc1ccc(C(=O)O)cc1",          -0.27),
    ("CH3",  "Cc1ccc(C(=O)O)cc1",           -0.17),
    ("H",    "OC(=O)c1ccccc1",               0.00),
    ("F",    "OC(=O)c1ccc(F)cc1",            0.06),
    ("Cl",   "OC(=O)c1ccc(Cl)cc1",           0.23),
    ("Br",   "OC(=O)c1ccc(Br)cc1",           0.23),
    ("CF3",  "OC(=O)c1ccc(C(F)(F)F)cc1",     0.54),
    ("CN",   "OC(=O)c1ccc(C#N)cc1",          0.66),
    ("NO2",  "OC(=O)c1ccc([N+](=O)[O-])cc1", 0.78),
]


class HammettSeries(Dataset):
    id = "S8"
    version = "v1"
    has_native_conformers = False

    def build_smiles(self, out: Path) -> None:
        rows = [
            {"id": f"s8_{sub}", "smiles": smi, "substituent": sub, "sigma_para": sigma}
            for sub, smi, sigma in _COMPOUNDS
        ]
        pd.DataFrame(rows).to_csv(out, index=False)
