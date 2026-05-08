"""Dataset stages and on-disk layout."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class Stage(str, Enum):
    SMILES = "smiles"
    CONFORMERS = "conformers"
    ELEKTRONN_COEFFS = "elektronn_coeffs"


STAGE_ORDER: dict[Stage, int] = {
    Stage.SMILES: 0,
    Stage.CONFORMERS: 1,
    Stage.ELEKTRONN_COEFFS: 2,
}

STAGE_EXT: dict[Stage, str] = {
    Stage.SMILES: ".csv",
    Stage.CONFORMERS: ".pkl",
    Stage.ELEKTRONN_COEFFS: ".pkl",
}

CACHE_ROOT = Path("cache")


def dataset_dir(ds_id: str) -> Path:
    return CACHE_ROOT / "datasets" / ds_id


def stage_path(ds_id: str, stage: Stage) -> Path:
    return dataset_dir(ds_id) / f"{stage.value}{STAGE_EXT[stage]}"


def dataset_size(ds_id: str) -> int:
    """Canonical molecule count for a dataset: number of rows in the SMILES
    stage CSV. Used by manifests to record per-molecule compute cost."""
    import pandas as pd

    smiles = stage_path(ds_id, Stage.SMILES)
    if not smiles.exists():
        return 0
    return len(pd.read_csv(smiles))


class Dataset:
    """Base class for datasets. Subclasses override build_smiles and,
    if has_native_conformers is True, build_native_conformers."""

    id: str = ""
    version: str = "v1"
    has_native_conformers: bool = False

    # Bump when changing test_mode_subsample logic to invalidate cached
    # test-mode SMILES files for this dataset only. Full-mode caches are
    # unaffected because this string is only included in the SMILES stage
    # version when --test-mode is active.
    test_mode_version: str = "v1-uniform"

    def build_smiles(self, out: Path) -> None:
        raise NotImplementedError(f"{self.id} must implement build_smiles")

    def build_native_conformers(self, smiles_csv: Path, out: Path) -> None:
        raise NotImplementedError(
            f"{self.id} declared has_native_conformers=True but did not "
            "implement build_native_conformers"
        )

    def test_mode_subsample(
        self, df: "pd.DataFrame", n: int, rng: "np.random.Generator"
    ) -> "pd.DataFrame":
        """Pick at most `n` rows from the post-filter SMILES dataframe.

        Default: uniform random without replacement. Override to preserve rows
        the downstream experiments depend on — e.g. WelQrate keeps every active
        so EXP-3a's retrieval task has at least one query per scaffold seed.

        Bumping `test_mode_version` after changing this method invalidates
        cached test-mode artifacts for the dataset.
        """
        if len(df) <= n:
            return df
        idx = rng.choice(len(df), size=n, replace=False)
        idx.sort()
        return df.iloc[idx]
