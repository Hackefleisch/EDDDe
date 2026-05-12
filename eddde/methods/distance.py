"""Distance-matrix computation shared by every experiment that needs
multi-pair `method.distance` calls.

`pairwise_matrix` is the single entry point. It dispatches across three
implementations depending on the method and dataset size:

  1. Method-provided batched `distances(embs_q, embs_c) -> ndarray` —
     for GPU or vectorised CPU kernels. The method handles all internal
     batching; we call it once and trust the result shape. Today no
     method overrides this; the hook is here so future learned /
     OT-style methods can plug in without changing experiments.

  2. multiprocessing.Pool over `method.distance(e1, e2)` — for slow
     per-pair methods (B8 today, B11 / OT-MUTs tomorrow). Embeddings
     are bound to worker globals by an initializer so per-task IPC
     payload is just an (i, j) index tuple. Assumes `fork` start
     method (Linux); copy-on-write keeps memory bounded even for the
     large pools we expect on a cluster.

  3. Serial nested loop — for small matrices or single-worker config.
     Pool spin-up is ~50–200 ms; below the threshold serial wins.

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
    method: Any,
    embeddings: dict[str, Any],
    query_ids: list[str],
    candidate_ids: list[str],
    *,
    n_workers: int | None = None,
) -> np.ndarray:
    """Distance matrix where `M[i, j] = method.distance(emb[query_ids[i]], emb[candidate_ids[j]])`.

    Dispatch (in order):
      1. `method.distances(embs_q, embs_c)` if the method's class defines it
         (single batched call — for GPU / vectorised methods).
      2. `multiprocessing.Pool` over `method.distance` if `n_workers > 1`
         and the matrix has at least `_PARALLEL_THRESHOLD` pairs.
      3. Plain serial loop.

    `n_workers` defaults to `eddde.N_WORKERS` (which the CLI's `--num-workers`
    can override). Pass `n_workers=1` to force serial regardless of size.
    """
    nq, nc = len(query_ids), len(candidate_ids)
    if nq == 0 or nc == 0:
        return np.zeros((nq, nc), dtype=float)

    embs_q = [embeddings[q] for q in query_ids]
    embs_c = [embeddings[c] for c in candidate_ids]

    # 1. Method-provided batched implementation (GPU / vectorised).
    if "distances" in vars(type(method)):
        M = method.distances(embs_q, embs_c)
        return np.asarray(M, dtype=float).reshape(nq, nc)

    if n_workers is None:
        n_workers = eddde.N_WORKERS

    # 2. Multiprocessing pool.
    if n_workers > 1 and nq * nc >= _PARALLEL_THRESHOLD:
        return _pairwise_parallel(method, embs_q, embs_c, n_workers)

    # 3. Serial fallback.
    M = np.empty((nq, nc), dtype=float)
    for i in range(nq):
        for j in range(nc):
            M[i, j] = method.distance(embs_q[i], embs_c[j])
    return M
