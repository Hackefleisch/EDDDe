"""MUT-mean family: scalarised mean pool of ElektroNN coefficients.

Strain A's simplest variants (PROJECT_PLAN.md §3.2.2). Each atom's 127-d
ElektroNN feature vector is scalarised to a 39-d rotation-invariant vector
via `strain_a.scalarise` (l>0 multiplicities collapse to their 2-norms),
then averaged across atoms.

- `MUT-mean`        — mean-pool atoms → 39-d, Euclidean distance.
- `MUT-mean-cosine` — mean-pool atoms → 39-d, cosine distance. A/B against
  MUT-mean to test whether magnitude information in the per-atom invariants
  carries similarity signal or whether direction alone is sufficient.
- `MUT-mean-max`    — component-wise max over atoms → 39-d, Euclidean
  distance. Control variant: tests the additivity assumption of mean-pooling
  by asking "is there *any* atom with extreme feature X?" instead of "what
  is the average feature X?".
- `MUT-mean-perelement` — mean within each element type (H, C, N, O, F, S,
  Cl); per-element 39-d blocks concatenated in fixed atomic-number order →
  7 × 39 = 273-d, Euclidean distance. Missing element → zero block.
  *Negative control:* per-element mean strips out the stoichiometric
  weighting that whole-molecule mean carries implicitly, so this is worse
  than plain `MUT-mean` rather than better.
- `MUT-mean-perelement-sum` — per-element *sum* with the same layout.
  `sum = count × mean` per block, so the composition (atom counts per
  element) is preserved alongside the per-element feature info. Direct
  test of whether composition adds signal on top of plain mean.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from ...data.base import Stage
from ..base import Method
from .strain_a import SCALARISED_DIM, scalarise


# Atomic numbers in ascending order. Must stay in sync with
# `eddde.data.elektronn_runner.supported_elements()` — currently
# {H, C, N, O, F, S, Cl} per the project-wide SMILES filter in
# `eddde/data/pipeline.py`. Hardcoded to avoid importing elektronn at
# methods-registry init time. If ElektroNN gains new elements, extend here,
# bump `MUT-mean-perelement` version, and re-embed.
ELEMENTS: tuple[int, ...] = (1, 6, 7, 8, 9, 16, 17)


class MutMean(Method):
    id = "MUT-mean"
    version = "mean-scalarised-euclidean-v2"
    needs = Stage.ELEKTRONN_COEFFS
    is_mut = True

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        coefficients: dict[str, np.ndarray] = stage_data[Stage.ELEKTRONN_COEFFS]["coefficients"]
        return {mol_id: scalarise(coeffs).mean(axis=0) for mol_id, coeffs in coefficients.items()}

    def distances(self, embs_q: list[Any], embs_c: list[Any]) -> np.ndarray:
        Q = np.stack(embs_q)
        C = np.stack(embs_c)
        return cdist(Q, C, "euclidean")


class MutMeanCosine(MutMean):
    id = "MUT-mean-cosine"
    version = "mean-scalarised-cosine-v1"

    def distances(self, embs_q: list[Any], embs_c: list[Any]) -> np.ndarray:
        Q = np.stack(embs_q)
        C = np.stack(embs_c)
        return cdist(Q, C, "cosine")


class MutMeanMax(MutMean):
    id = "MUT-mean-max"
    version = "max-scalarised-euclidean-v1"

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        coefficients: dict[str, np.ndarray] = stage_data[Stage.ELEKTRONN_COEFFS]["coefficients"]
        return {mol_id: scalarise(coeffs).max(axis=0) for mol_id, coeffs in coefficients.items()}


def _embed_per_element(stage_data: dict, aggregator) -> dict[str, np.ndarray]:
    """Per-element pool of scalarised features → 7×39 = 273-d, fixed element order.

    `aggregator` reduces a `(k, 39)` block of one element's atoms to a `(39,)`
    vector — e.g. `np.mean` or `np.sum` along axis 0. Missing elements get a
    zero 39-d block so embeddings stay aligned across molecules.
    """
    coefficients: dict[str, np.ndarray] = stage_data[Stage.ELEKTRONN_COEFFS]["coefficients"]
    conformers = stage_data[Stage.CONFORMERS]

    out: dict[str, np.ndarray] = {}
    for mol_id, coeffs in coefficients.items():
        scalarised = scalarise(coeffs)
        atomic_nums = np.array(
            [a.GetAtomicNum() for a in conformers[mol_id].GetAtoms()],
            dtype=np.int32,
        )
        blocks = [
            aggregator(scalarised[atomic_nums == z], axis=0)
            if (atomic_nums == z).any()
            else np.zeros(SCALARISED_DIM)
            for z in ELEMENTS
        ]
        out[mol_id] = np.concatenate(blocks)
    return out


class MutMeanPerElement(MutMean):
    id = "MUT-mean-perelement"
    version = "perelement-scalarised-euclidean-v1"

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        return _embed_per_element(stage_data, np.mean)


class MutMeanPerElementSum(MutMean):
    id = "MUT-mean-perelement-sum"
    version = "perelement-sum-scalarised-euclidean-v1"

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        return _embed_per_element(stage_data, np.sum)
