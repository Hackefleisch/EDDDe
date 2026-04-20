"""S5 homologous series: polyethylene glycols HO-(CH2CH2O)n-H, n=1-10 (PROJECT_PLAN.md §5.1)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..base import Dataset


class PEG(Dataset):
    id = "S5"
    version = "v1"
    has_native_conformers = False

    def build_smiles(self, out: Path) -> None:
        df = pd.DataFrame(
            {
                "id": [f"peg_{n}" for n in range(1, 11)],
                "smiles": ["O" + "CCO" * n for n in range(1, 11)],
                "position": list(range(1, 11)),
            }
        )
        df.to_csv(out, index=False)
