"""Distance-matrix computation shared by every experiment that needs
multi-pair `method.distance` calls.

`pairwise_matrix` is the single entry point. Dispatch follows from
whether the method has overridden the batched `Method.distances` default:

  1. Method overrides `distances()` (B1-B7, B9, MUT-mean today; future
     GPU / OT methods tomorrow) — call it once and return. A single
     vectorised C / BLAS / GPU call is the right thing regardless of
     matrix size, and there is no per-pair IPC to amortise.

  2. Method only ships per-pair `distance()` (B8 Gaussian shape align;
     future alignment-style methods) — fan out across
     `multiprocessing.Pool` when the matrix is big enough to amortise
     pool overhead (~50_000 pairs), else fall through to the
     inherited serial-loop `distances()`. Workers receive embeddings
     once via initializer (broadcast into worker globals), so per-task
     IPC payload is just an (i, j) index tuple. Assumes `fork` start
     method (Linux); copy-on-write keeps memory bounded even for the
     large pools we expect on a cluster.

Memory: only the (queries × candidates) result matrix is held in the
driver. Pair indices are streamed via a generator into pool.imap so
the driver never materializes the full pair list (matters once
full-mode pairs cross 10^8).

Why this module is under `eddde.methods` and not `eddde.experiments`:
matrix-shaped distance is needed by EXP-2 (functional-group), EXP-3a,
EXP-3b, and the planned EXP-5/EXP-6 — not just retrieval experiments.
A future `pair_distances(method, embeddings, pairs)` variant for
explicit pair-list inputs (EXP-4 cliffs, EXP-5 bioisostere pair sets)
will live here too.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np

import eddde
from .base import Method


# Threshold below which we skip the worker pool. Tuned for a typical
# 8 µs/pair fingerprint method: 50 000 pairs at 8 µs ≈ 400 ms serial,
# competitive with pool overhead. Slow methods (~1 ms/pair) win much
# earlier, so the threshold biases toward serial for cheap methods
# without hurting slow ones meaningfully.
_PARALLEL_THRESHOLD = 50_000


# Globals set by `_worker_init` after fork. Workers read these instead
# of pickling embeddings on every task.
_W_METHOD: Any = None
_W_EMBS_Q: list[Any] | None = None
_W_EMBS_C: list[Any] | None = None


def _worker_init(method: Any, embs_q: list[Any], embs_c: list[Any]) -> None:
    global _W_METHOD, _W_EMBS_Q, _W_EMBS_C
    _W_METHOD = method
    _W_EMBS_Q = embs_q
    _W_EMBS_C = embs_c


def _worker_pair(ij: tuple[int, int]) -> float:
    i, j = ij
    return _W_METHOD.distance(_W_EMBS_Q[i], _W_EMBS_C[j])


def _pair_indices(nq: int, nc: int):
    for i in range(nq):
        for j in range(nc):
            yield (i, j)


def _pairwise_parallel(
    method: Any,
    embs_q: list[Any],
    embs_c: list[Any],
    n_workers: int,
) -> np.ndarray:
    nq, nc = len(embs_q), len(embs_c)
    n_pairs = nq * nc
    # Target ~16 chunks per worker for decent load balance without per-chunk
    # overhead dominating. Floor at 64 so tiny matrices don't hand workers
    # one-pair tasks.
    chunksize = max(64, n_pairs // (n_workers * 16))
    with mp.Pool(
        n_workers,
        initializer=_worker_init,
        initargs=(method, embs_q, embs_c),
    ) as pool:
        results = pool.imap(_worker_pair, _pair_indices(nq, nc), chunksize=chunksize)
        arr = np.fromiter(results, dtype=float, count=n_pairs)
    return arr.reshape(nq, nc)


def pairwise_matrix(
    method: Method,
    embeddings: dict[str, Any],
    query_ids: list[str],
    candidate_ids: list[str],
    *,
    n_workers: int | None = None,
) -> np.ndarray:
    """Distance matrix where `M[i, j] = method.distance(emb[query_ids[i]], emb[candidate_ids[j]])`.

    Dispatch is decided by whether the method overrides
    `Method.distances`:

      - Override present (B1-B7, B9, MUT-mean) → call the batched
        implementation directly.
      - No override (B8 + future alignment-style methods) → fan out
        across `multiprocessing.Pool` over `method.distance` if
        `n_workers > 1` and the matrix has at least
        `_PARALLEL_THRESHOLD` pairs; else fall through to the
        inherited serial `distances()` (nested loop over `distance`).

    `n_workers` defaults to `eddde.N_WORKERS` (the CLI's `--num-workers`
    overrides it). Pass `n_workers=1` to force serial regardless of size.
    """
    nq, nc = len(query_ids), len(candidate_ids)
    if nq == 0 or nc == 0:
        return np.zeros((nq, nc), dtype=float)

    embs_q = [embeddings[q] for q in query_ids]
    embs_c = [embeddings[c] for c in candidate_ids]

    # Batched override present → trust it. Vectorised / BLAS / GPU calls
    # don't benefit from external IPC fanout.
    if type(method).distances is not Method.distances:
        M = method.distances(embs_q, embs_c)
        return np.asarray(M, dtype=float).reshape(nq, nc)

    # Otherwise this method only ships per-pair `distance()`. For big
    # matrices fan out across processes; otherwise let the inherited
    # default `distances()` (serial nested loop) handle it. The default
    # is one source of truth for what "serial" means.
    if n_workers is None:
        n_workers = eddde.N_WORKERS
    if n_workers > 1 and nq * nc >= _PARALLEL_THRESHOLD:
        return _pairwise_parallel(method, embs_q, embs_c, n_workers)

    return method.distances(embs_q, embs_c)
