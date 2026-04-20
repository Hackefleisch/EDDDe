"""S4 homologous series: n-alkylamines, methylamine to dodecylamine (PROJECT_PLAN.md §5.1)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..base import Dataset


class Alkylamines(Dataset):
    id = "S4"
    version = "v1"
    has_native_conformers = False

    def build_smiles(self, out: Path) -> None:
        df = pd.DataFrame(
            {
                "id": [f"alkylamine_{n}" for n in range(1, 13)],
                "smiles": ["C" * n + "N" for n in range(1, 13)],
                "position": list(range(1, 13)),
            }
        )
        df.to_csv(out, index=False)
