"""Strain A shared helpers — scalar-invariant per-atom feature reduction.

ElektroNN's 127-d per-atom output decomposes by irrep as
`14x0e + 14x1o + 5x2e + 4x3o + 2x4e` (PROJECT_PLAN.md §3.2.2). Raw mean-pooling
across atoms is not rotation-invariant because the l>0 blocks transform by
the Wigner D-matrix under rotation. `scalarise` collapses each l>0
multiplicity to its 2-norm in its `(2l+1)`-dim irrep subspace, leaving 39
rotation-invariant numbers per atom: 14 scalars + 14 l=1 norms + 5 l=2
norms + 4 l=3 norms + 2 l=4 norms.

Every Strain A variant (mean, cosine, irrep-weighted, mahalanobis, etc.)
operates on this 39-d representation, so the scalarisation lives here as a
shared helper rather than in any one variant.
"""
from __future__ import annotations

import numpy as np


# (l, multiplicity) per irrep block, in the order ElektroNN concatenates them.
# Block sizes:  14*1 + 14*3 + 5*5 + 4*7 + 2*9 = 14 + 42 + 25 + 28 + 18 = 127.
# Output sizes: 14   + 14   + 5   + 4   + 2   = 39 per atom.
IRREP_LAYOUT: list[tuple[int, int]] = [(0, 14), (1, 14), (2, 5), (3, 4), (4, 2)]

SCALARISED_DIM = sum(mult for _, mult in IRREP_LAYOUT)  # 39
RAW_DIM = sum(mult * (2 * l + 1) for l, mult in IRREP_LAYOUT)  # 127


def scalarise(coeffs: np.ndarray) -> np.ndarray:
    """Per-atom ElektroNN coefficients → rotation-invariant 39-d vectors.

    coeffs: `(n_atoms, 127)`. Returns `(n_atoms, 39)`.
    """
    assert coeffs.shape[-1] == RAW_DIM, (
        f"expected last dim {RAW_DIM}, got {coeffs.shape[-1]}"
    )
    n_atoms = coeffs.shape[0]
    blocks: list[np.ndarray] = []
    offset = 0
    for l, mult in IRREP_LAYOUT:
        size_per_mult = 2 * l + 1
        block_size = mult * size_per_mult
        block = coeffs[:, offset:offset + block_size].reshape(n_atoms, mult, size_per_mult)
        if l == 0:
            blocks.append(block.squeeze(-1))
        else:
            blocks.append(np.linalg.norm(block, axis=2))
        offset += block_size
    return np.concatenate(blocks, axis=1)
