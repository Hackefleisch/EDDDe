"""S6: monosubstituted benzenes (PROJECT_PLAN.md §5.2 / EXP-2)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..base import Dataset

# Donor / acceptor / neutral labels follow standard electronic-effect classification:
#   donor   — strong resonance donors (NH₂, OH, OCH₃) + weak inductive donor (CH₃)
#   acceptor — π-withdrawing groups (NO₂, COOH, CHO, COCH₃)
#   neutral  — H plus halogens (inductively withdrawing, resonance donating — placed here
#               to separate them from the clear EWG cluster and because M-HALO-ORDER
#               measures ordering *within* this group)
_SUBSTITUENTS = [
    ("H",     "c1ccccc1",              "neutral"),
    ("CH3",   "Cc1ccccc1",             "donor"),
    ("OH",    "Oc1ccccc1",             "donor"),
    ("NH2",   "Nc1ccccc1",             "donor"),
    ("F",     "Fc1ccccc1",             "neutral"),
    ("Cl",    "Clc1ccccc1",            "neutral"),
    ("Br",    "Brc1ccccc1",            "neutral"),
    ("NO2",   "O=[N+]([O-])c1ccccc1", "acceptor"),
    ("COOH",  "OC(=O)c1ccccc1",       "acceptor"),
    ("CHO",   "O=Cc1ccccc1",           "acceptor"),
    ("COCH3", "CC(=O)c1ccccc1",        "acceptor"),
    ("OCH3",  "COc1ccccc1",            "donor"),
]


class SubstBenzenes(Dataset):
    id = "S6"
    version = "v1"
    has_native_conformers = False

    def build_smiles(self, out: Path) -> None:
        rows = [
            {"id": f"s6_{sub}", "smiles": smi, "substituent": sub, "label": label}
            for sub, smi, label in _SUBSTITUENTS
        ]
        pd.DataFrame(rows).to_csv(out, index=False)
