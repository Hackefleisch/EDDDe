"""Dataset stages and on-disk layout."""
from __future__ import annotations

from enum import Enum
from pathlib import Path


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


class Dataset:
    """Base class for datasets. Subclasses override build_smiles and,
    if has_native_conformers is True, build_native_conformers."""

    id: str = ""
    version: str = "v1"
    has_native_conformers: bool = False

    def build_smiles(self, out: Path) -> None:
        raise NotImplementedError(f"{self.id} must implement build_smiles")

    def build_native_conformers(self, smiles_csv: Path, out: Path) -> None:
        raise NotImplementedError(
            f"{self.id} declared has_native_conformers=True but did not "
            "implement build_native_conformers"
        )
