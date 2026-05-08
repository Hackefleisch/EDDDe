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
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .. import SEED
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


# Time-budgeted distance benchmark: target ~target_seconds of measured time,
# with a small warmup to skip JIT/lazy-import effects, bounded by min/max
# pair counts so even sub-millisecond methods get a stable mean and
# multi-second methods (future OT-style MUTs) don't blow up the run.
_BENCHMARK_TARGET_SECONDS = 1.0
_BENCHMARK_WARMUP_PAIRS = 5
_BENCHMARK_MIN_PAIRS = 20
_BENCHMARK_MAX_PAIRS = 2000


def benchmark_distance(method, embeddings: dict[str, Any]) -> tuple[float, int]:
    """Measure mean wall-clock time of one `method.distance(e1, e2)` call.

    Samples random distinct pairs from `embeddings` with a fixed seed, runs
    a few warmup calls to flush JIT / lazy imports, then keeps timing pairs
    until either the time budget or the pair-count cap is reached. Returns
    (mean_seconds_per_pair, n_pairs_measured); (0.0, 0) when there are
    fewer than 2 embeddings.
    """
    ids = list(embeddings.keys())
    if len(ids) < 2:
        return 0.0, 0

    rng = np.random.default_rng(SEED)

    def _sample_pair():
        i, j = rng.choice(len(ids), size=2, replace=False)
        return embeddings[ids[i]], embeddings[ids[j]]

    # Warmup — discard timing.
    for _ in range(min(_BENCHMARK_WARMUP_PAIRS, len(ids))):
        e1, e2 = _sample_pair()
        method.distance(e1, e2)

    n = 0
    start = time.perf_counter()
    deadline = start + _BENCHMARK_TARGET_SECONDS
    while n < _BENCHMARK_MAX_PAIRS:
        e1, e2 = _sample_pair()
        method.distance(e1, e2)
        n += 1
        if n >= _BENCHMARK_MIN_PAIRS and time.perf_counter() >= deadline:
            break
    elapsed = time.perf_counter() - start

    if n == 0:
        return 0.0, 0
    return elapsed / n, n
