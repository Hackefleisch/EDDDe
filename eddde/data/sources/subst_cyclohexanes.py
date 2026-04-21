"""S7: monosubstituted cyclohexanes (PROJECT_PLAN.md §5.2 / EXP-2).

Same substituent set as S6 minus -Br and -NO₂.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..base import Dataset

_SUBSTITUENTS = [
    ("H",     "C1CCCCC1",        "neutral"),
    ("CH3",   "CC1CCCCC1",       "donor"),
    ("OH",    "OC1CCCCC1",       "donor"),
    ("NH2",   "NC1CCCCC1",       "donor"),
    ("F",     "FC1CCCCC1",       "neutral"),
    ("Cl",    "ClC1CCCCC1",      "neutral"),
    ("COOH",  "OC(=O)C1CCCCC1", "acceptor"),
    ("CHO",   "O=CC1CCCCC1",     "acceptor"),
    ("COCH3", "CC(=O)C1CCCCC1",  "acceptor"),
    ("OCH3",  "COC1CCCCC1",      "donor"),
]


class SubstCyclohexanes(Dataset):
    id = "S7"
    version = "v1"
    has_native_conformers = False

    def build_smiles(self, out: Path) -> None:
        rows = [
            {"id": f"s7_{sub}", "smiles": smi, "substituent": sub, "label": label}
            for sub, smi, label in _SUBSTITUENTS
        ]
        pd.DataFrame(rows).to_csv(out, index=False)
