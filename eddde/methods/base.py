"""Method protocol and embedding cache helpers.

All methods (MUTs and baselines) share one interface:
  - id: stable identifier used in filenames and result columns
  - version: bump when embed or distance implementation changes
  - needs: highest dataset stage this method reads
  - embed_dataset(stage_data) -> dict[mol_id -> embedding]
  - distance(e1, e2) -> float (smaller = more similar)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Protocol

from ..data.base import Stage


EMBEDDING_CACHE_ROOT = Path("cache/embeddings")


def embedding_path(method_id: str, dataset_id: str) -> Path:
    return EMBEDDING_CACHE_ROOT / method_id / f"{dataset_id}.pkl"


class Method(Protocol):
    id: str
    version: str
    needs: Stage

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]: ...

    def distance(self, e1: Any, e2: Any) -> float: ...


def save_embeddings(path: Path, embeddings: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)
