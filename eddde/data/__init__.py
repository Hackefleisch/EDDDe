"""Dataset registry. Import a dataset module and call `_register` to add it."""
from __future__ import annotations

from .base import Dataset
from .sources.alkanes import Alkanes
from .sources.alkanols import Alkanols


DATASETS: dict[str, Dataset] = {}


def _register(ds: Dataset) -> None:
    if ds.id in DATASETS:
        raise ValueError(f"duplicate dataset id: {ds.id}")
    DATASETS[ds.id] = ds


_register(Alkanes())
_register(Alkanols())
