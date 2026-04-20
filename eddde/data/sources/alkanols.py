"""S2 homologous series: n-alkanols, methanol to 1-dodecanol (PROJECT_PLAN.md §5.1)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..base import Dataset


class Alkanols(Dataset):
    id = "S2"
    version = "v1"
    has_native_conformers = False

    def build_smiles(self, out: Path) -> None:
        df = pd.DataFrame(
            {
                "id": [f"alkanol_{n}" for n in range(1, 13)],
                "smiles": ["C" * n + "O" for n in range(1, 13)],
                "position": list(range(1, 13)),
            }
        )
        df.to_csv(out, index=False)
