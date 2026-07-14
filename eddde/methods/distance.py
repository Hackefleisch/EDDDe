"""Distance-matrix / pair-list computation shared by every experiment
that needs multi-pair `method.distance` calls.

Two entry points:

  * `pairwise_matrix` — dense (queries × candidates) matrix. Used by
    EXP-2, EXP-3a, EXP-3b (and planned EXP-6) where the experiment
    naturally needs a rectangular block.
  * `pair_distances` — distances for an explicit list of (id_a, id_b)
    pairs. Used by EXP-4 (and planned EXP-5) where the pair set is
    sparse relative to N² (similar-pair cliffs, bioisostere sets).

Dispatch for both follows from whether the method has overridden the
batched `Method.distances` default:

  1. Method overrides `distances()` (B1-B7, B9, MUT-mean today; future
     GPU / OT methods tomorrow) — call it once and return. A single
     vectorised C / BLAS / GPU call is the right thing regardless of
     matrix size, and there is no per-pair IPC to amortise.
     For `pair_distances`, unique IDs appearing in the pair list are
     densified into a submatrix and the listed pairs are indexed out
     — typically far smaller than the full N × N when pairs are sparse.

  2. Method only ships per-pair `distance()` (B8 Gaussian shape align;
     future alignment-style methods) — decide parallelisation by
     predicted serial time = n_calls × measured-per-pair-cost, and fan
     out across `multiprocessing.Pool` when that exceeds the pool-
     overhead break-even. The per-pair cost comes from one of two
     places, in priority order:
       (a) `method._distance_time_per_pair`, set by the runner from
           the embedding-stage benchmark — either freshly measured on
           rebuild or read from the embedding manifest on cache hit.
       (b) An on-the-spot `benchmark_distance` against the embeddings
           passed in, run iff (a) is absent (e.g. ad-hoc usage outside
           the runner). Result is stashed back on the method so
           subsequent calls reuse it.
     There is no static pair-count fallback — we always have a real
     per-pair measurement before deciding.
     Workers receive embeddings once via initializer (broadcast into
     worker globals), so per-task IPC payload is just an (i, j) index
     or (id_a, id_b) tuple. Assumes `fork` start method (Linux);
     copy-on-write keeps memory bounded even for the large pools we
     expect on a cluster.

Memory (`pairwise_matrix`): only the (queries × candidates) result
matrix is held in the driver. Pair indices are streamed via a
generator into pool.imap so the driver never materializes the full
pair list (matters once full-mode pairs cross 10^8).

Why this module is under `eddde.methods` and not `eddde.experiments`:
matrix-shaped and pair-list distance are needed by EXP-2, EXP-3a/3b,
EXP-4, and the planned EXP-5/EXP-6 — not just retrieval experiments.
"""
from __future__ import annotations

import multiprocessing as mp
from collections.abc import Sequence
from typing import Any

import numpy as np

import eddde
from .base import Method, benchmark_distance


# Parallelise iff the measured per-pair cost predicts serial time above
# this budget. ~1 s comfortably covers fork + initializer broadcast for
# the pool sizes we run at (cpu_count up to ~64) without flipping
# borderline cases that would lose to serial after chunk-imbalance
# overhead.
_PARALLEL_MIN_SERIAL_SECONDS = 1.0


# Globals set by worker initializers after fork. Workers read these
# instead of pickling embeddings on every task.
_W_METHOD: Any = None
_W_EMBS_Q: list[Any] | None = None
_W_EMBS_C: list[Any] | None = None
_W_EMBS: dict[str, Any] | None = None
_W_SYMMETRISE: bool = False


def _worker_init(method: Any, embs_q: list[Any], embs_c: list[Any]) -> None:
    global _W_METHOD, _W_EMBS_Q, _W_EMBS_C
    _W_METHOD = method
    _W_EMBS_Q = embs_q
    _W_EMBS_C = embs_c


def _worker_pair(ij: tuple[int, int]) -> float:
    i, j = ij
    return _W_METHOD.distance(_W_EMBS_Q[i], _W_EMBS_C[j])


def _worker_init_pairs(
    method: Any,
    embeddings: dict[str, Any],
    symmetrise: bool,
) -> None:
    global _W_METHOD, _W_EMBS, _W_SYMMETRISE
    _W_METHOD = method
    _W_EMBS = embeddings
    _W_SYMMETRISE = symmetrise


def _worker_id_pair(ab: tuple[str, str]) -> float:
    a, b = ab
    d_ab = _W_METHOD.distance(_W_EMBS[a], _W_EMBS[b])
    if not _W_SYMMETRISE:
        return d_ab
    return 0.5 * (d_ab + _W_METHOD.distance(_W_EMBS[b], _W_EMBS[a]))


def _pair_indices(nq: int, nc: int):
    for i in range(nq):
        for j in range(nc):
            yield (i, j)


def _ensure_pair_cost(method: Method, embeddings: dict[str, Any]) -> float | None:
    """Return measured s/pair, measuring and stashing if absent."""
    t_pair = getattr(method, "_distance_time_per_pair", None)
    if t_pair:
        return t_pair
    t_pair, n_measured = benchmark_distance(method, embeddings)
    if n_measured:
        method._distance_time_per_pair = t_pair
    return t_pair or None


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


def _pair_list_parallel(
    method: Any,
    embeddings: dict[str, Any],
    pairs: Sequence[tuple[str, str]],
    n_workers: int,
    symmetrise: bool,
) -> np.ndarray:
    n_pairs = len(pairs)
    chunksize = max(64, n_pairs // (n_workers * 16))
    with mp.Pool(
        n_workers,
        initializer=_worker_init_pairs,
        initargs=(method, embeddings, symmetrise),
    ) as pool:
        results = pool.imap(_worker_id_pair, pairs, chunksize=chunksize)
        return np.fromiter(results, dtype=float, count=n_pairs)


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
        implementation directly. The per-pair benchmark is irrelevant
        and never consulted.
      - No override (B8 + future alignment-style methods) → use the
        measured per-pair cost on `method._distance_time_per_pair` to
        predict serial time and fan out across `multiprocessing.Pool`
        iff that exceeds `_PARALLEL_MIN_SERIAL_SECONDS`. If no
        measurement is stashed, run `benchmark_distance` on the
        embeddings being passed in and stash the result.

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

    # Per-pair-only method. Need a measured per-pair cost; either it was
    # stashed (by the runner from the embedding manifest, or by an
    # earlier call here), or we measure now on the passed-in embeddings.
    # Stash either way so subsequent calls reuse it.
    if n_workers is None:
        n_workers = eddde.N_WORKERS

    t_pair = _ensure_pair_cost(method, embeddings)

    if n_workers > 1 and t_pair and (nq * nc) * t_pair >= _PARALLEL_MIN_SERIAL_SECONDS:
        return _pairwise_parallel(method, embs_q, embs_c, n_workers)

    return method.distances(embs_q, embs_c)


def pair_distances(
    method: Method,
    embeddings: dict[str, Any],
    pairs: Sequence[tuple[str, str]],
    *,
    n_workers: int | None = None,
    symmetrise: bool = False,
) -> np.ndarray:
    """Distances for an explicit list of (id_a, id_b) pairs.

    Returns a length-`len(pairs)` float array. When `symmetrise=True`,
    each entry is `0.5 * (d(a,b) + d(b,a))` — the same step EXP-2/EXP-4
    apply to square self-matrices for methods with intrinsically
    asymmetric distance (B8, B11). No-op for naturally-symmetric methods.

    Dispatch mirrors `pairwise_matrix`:

      - Batched override present → densify the unique IDs that appear in
        `pairs`, call `method.distances` once on that submatrix, index
        the listed pairs out (and average with the transpose when
        symmetrising). Cheap for B1–B7 / B9 / MUT-mean even at MoleculeACE
        target sizes; typically much smaller than a full N × N when the
        pair set is sparse.
      - Per-pair-only → evaluate only the listed pairs (×2 if
        symmetrising). Fan out across `multiprocessing.Pool` when the
        predicted serial cost exceeds `_PARALLEL_MIN_SERIAL_SECONDS`.

    `n_workers` defaults to `eddde.N_WORKERS`. Pass `n_workers=1` to
    force serial regardless of size.
    """
    n_pairs = len(pairs)
    if n_pairs == 0:
        return np.zeros(0, dtype=float)

    if type(method).distances is not Method.distances:
        # Unique IDs preserve first-seen order so index maps are stable.
        unique: list[str] = list(dict.fromkeys(i for ab in pairs for i in ab))
        idx = {mol_id: k for k, mol_id in enumerate(unique)}
        embs = [embeddings[mol_id] for mol_id in unique]
        M = np.asarray(method.distances(embs, embs), dtype=float)
        if symmetrise:
            M = 0.5 * (M + M.T)
        return np.array(
            [M[idx[a], idx[b]] for a, b in pairs],
            dtype=float,
        )

    if n_workers is None:
        n_workers = eddde.N_WORKERS

    # Each listed pair costs one distance() call; symmetrise doubles it.
    n_calls = n_pairs * (2 if symmetrise else 1)
    t_pair = _ensure_pair_cost(method, embeddings)

    if n_workers > 1 and t_pair and n_calls * t_pair >= _PARALLEL_MIN_SERIAL_SECONDS:
        return _pair_list_parallel(method, embeddings, pairs, n_workers, symmetrise)

    out = np.empty(n_pairs, dtype=float)
    for k, (a, b) in enumerate(pairs):
        d_ab = method.distance(embeddings[a], embeddings[b])
        if symmetrise:
            out[k] = 0.5 * (d_ab + method.distance(embeddings[b], embeddings[a]))
        else:
            out[k] = d_ab
    return out
