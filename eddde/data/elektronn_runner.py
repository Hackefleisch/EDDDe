"""ElektroNN integration: generate DFT-theory-level electron density coefficient matrices.

Expected output: pickled dict[mol_id -> np.ndarray of shape (n_atoms, 127)].
Not yet wired up; the first method that needs ELEKTRONN_COEFFS will trigger
implementation. When wired, bump VERSION to invalidate downstream caches.
"""
from __future__ import annotations

from pathlib import Path


VERSION = "stub-v0"


def generate(conformers_pkl: Path, out: Path) -> None:
    raise NotImplementedError(
        "ElektroNN integration is not implemented yet. "
        "Expected output at `out`: pickled dict[mol_id -> np.ndarray (n_atoms, 127)]. "
        "Register a method that needs Stage.ELEKTRONN_COEFFS to force this to be written."
    )
