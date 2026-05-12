"""Method base class and embedding cache helpers.

All methods (MUTs and baselines) share one interface:
  - id: stable identifier used in filenames and result columns
  - version: bump when embed or distance implementation changes
  - needs: highest dataset stage this method reads
  - embed_dataset(stage_data) -> dict[mol_id -> embedding]

Plus one of:
  - distance(e1, e2) -> float — per-pair, for methods that have no
    natural batched form (e.g. B8 Gaussian shape alignment). The
    framework parallelises these across `multiprocessing.Pool` when
    `eddde.methods.distance.pairwise_matrix` is called with a large
    matrix.
  - distances(embs_q, embs_c) -> ndarray — batched, for methods with a
    vectorised BLAS / RDKit-bulk / GPU implementation (B1-B7, B9,
    MUT-mean). The framework calls this directly and skips the worker
    pool — a single C-level call usually beats any IPC-bound fanout.

A subclass must implement at least one (enforced by __init_subclass__).
The framework derives the other mechanically: a 1×1 batched call when
only distances() is provided; a serial nested loop when only distance()
is provided. This keeps a single source of truth per method (no drift
between two implementations of the same semantics).
"""
from __future__ import annotations

import pickle
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .. import SEED
from ..data.base import Stage


EMBEDDING_CACHE_ROOT = Path("cache/embeddings")


def embedding_path(method_id: str, dataset_id: str) -> Path:
    return EMBEDDING_CACHE_ROOT / method_id / f"{dataset_id}.pkl"


class Method(ABC):
    id: str
    version: str
    needs: Stage

    # Set by the runner from the embedding-stage benchmark before any
    # experiment runs. `pairwise_matrix` reads it to predict serial cost
    # and parallelise only when the predicted time exceeds pool overhead.
    # None means "no measurement yet" — the static pair-count threshold
    # is used as a fallback.
    _distance_time_per_pair: float | None = None

    @abstractmethod
    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        ...

    def distance(self, e1: Any, e2: Any) -> float:
        """Per-pair distance.

        Default: a 1×1 batched call via `distances()`. Subclasses that have
        a per-pair-only implementation override this; subclasses with a
        batched implementation typically leave it alone.
        """
        return float(self.distances([e1], [e2])[0, 0])

    def distances(self, embs_q: list, embs_c: list) -> np.ndarray:
        """Batched (queries × candidates) distance matrix.

        Default: serial nested loop over `distance()`. Subclasses with a
        vectorised / BLAS / GPU implementation override this; subclasses
        that only have a per-pair `distance()` leave it alone — the
        framework's `pairwise_matrix` detects the default and routes
        through `multiprocessing.Pool` for large matrices.
        """
        return np.array(
            [[self.distance(q, c) for c in embs_c] for q in embs_q],
            dtype=float,
        )

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        # At least one of distance / distances must be overridden in the
        # concrete subclass; otherwise both defaults call each other and
        # any invocation infinite-recurses. Caught at class-definition
        # time rather than at first call.
        if (cls.distance is Method.distance
                and cls.distances is Method.distances):
            raise TypeError(
                f"{cls.__name__} must implement either `distance()` or "
                "`distances()` (or both)."
            )


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
