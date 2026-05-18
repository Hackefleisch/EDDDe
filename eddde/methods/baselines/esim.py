"""B11: shape + electrostatic similarity (open-source eSim analogue).

PROJECT_PLAN.md cites Jain et al. 2020 for B11, which is the proprietary
Optibrium / Jain-lab "eSim". This implementation substitutes the
open-source `espsim` package (Bolcato et al. JCIM 2022, MIT,
`hesther/espsim`). Same compromise as B8 (cites OpenEye ROCS, implements
RDKit's `rdShapeAlign`): we get the same conceptual measurement -- a
combined shape + electrostatic-potential overlap on an aligned pair --
without pulling in commercial software.

Two variants ship side-by-side to isolate the contribution of the
alignment step:

  B11-shape : rdShapeAlign.AlignMol  -- Grant-Gaussian shape solver,
                                        same machinery B8 uses for ROCS.
  B11-o3a   : rdMolAlign.GetO3A      -- atom-correspondence alignment
                                        (espsim's own EmbedAlignScore
                                        uses this under the hood).

Both score the aligned pose with espsim's GetShapeSim + GetEspSim using
MMFF partial charges (espsim falls back to Gasteiger when MMFF
parameterisation fails for a molecule). The combo follows the B8 /
ROCS convention: combo = 0.5 * (shape_sim + esp_sim), distance = 1 -
combo.

Like B8, the alignment is inherently per-pair so both classes implement
`distance()` only -- `pairwise_matrix` fans out across
`multiprocessing.Pool` for large matrices. Distance is asymmetric (the
probe is rotated onto the reference, so d(a,b) != d(b,a)); experiments
that build self-distance matrices must symmetrise as EXP-2 already
does (see CLAUDE.md "Distance computation in experiments").
"""
from __future__ import annotations

import warnings
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdShapeAlign

# espsim imports pkg_resources at module load and emits a deprecation
# warning on every interpreter start; suppress so the runner log stays
# clean. The warning is upstream-only and doesn't affect behaviour.
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
from espsim import GetEspSim, GetShapeSim  # noqa: E402

from ...data.base import Stage
from ..base import Method


def _espsim_combo(prb_aligned: Chem.Mol, ref: Chem.Mol) -> float:
    """Shape + ESP combo on a pre-aligned pair; returns similarity in [0, 1].

    When one or both molecules have an all-zero ESP (typical for pure
    hydrocarbons where every MMFF partial charge is ~0), the Carbo
    denominator ``√(intAA · intBB)`` is zero and espsim raises
    ``ValueError("Denominator in similarity calculation equals zero.")``.
    There is genuinely no ESP signal to compare in that case, so we
    degrade gracefully to shape-only similarity (i.e. ESP contributes
    nothing). This affects whole datasets of non-polar molecules
    (S1 alkanes) uniformly across all pairs, so the comparison stays
    fair.
    """
    shape_sim = GetShapeSim(prb_aligned, ref)
    try:
        esp_sim = GetEspSim(
            prb_aligned, ref,
            partialCharges="mmff",
            metric="carbo",
            renormalize=True,
        )
    except ValueError:
        return shape_sim
    return 0.5 * (shape_sim + esp_sim)


def _embed_conformer_copies(stage_data: dict) -> dict[str, Any]:
    conformers: dict[str, Chem.Mol] = stage_data[Stage.CONFORMERS]
    return {mol_id: Chem.Mol(mol) for mol_id, mol in conformers.items()}


class ESimShape(Method):
    """B11-shape: shape-driven pose search (rdShapeAlign) + espsim combo."""

    id = "B11-shape"
    version = "espsim-shape-mmff-v2"
    needs = Stage.CONFORMERS

    embed_dataset = staticmethod(_embed_conformer_copies)

    def distance(self, e1: Any, e2: Any) -> float:
        ref = e1
        prb = Chem.Mol(e2)  # AlignMol mutates the probe's conformer in place
        rdShapeAlign.AlignMol(ref, prb)
        return 1.0 - _espsim_combo(prb, ref)


class ESimO3A(Method):
    """B11-o3a: Open3DAlign atom-correspondence alignment + espsim combo.

    Known characteristic: for symmetric molecules (e.g. aniline, benzene)
    O3A can pick the symmetry-mapped atom correspondence rather than the
    identity, producing a 180° flip and a non-zero self-distance (~0.05-
    0.10). rdShapeAlign (B11-shape) has no atom correspondence and
    doesn't suffer this -- the contrast is exactly the signal isolating
    the two variants was meant to capture.
    """

    id = "B11-o3a"
    version = "espsim-o3a-mmff-v2"
    needs = Stage.CONFORMERS

    embed_dataset = staticmethod(_embed_conformer_copies)

    def distance(self, e1: Any, e2: Any) -> float:
        ref = e1
        prb = Chem.Mol(e2)
        # GetO3A needs MMFF atom types. They occasionally fail for
        # unusual atom environments -- fall back to Crippen O3A, which
        # uses LogP/MR contributions and parameterises everything in our
        # supported {H,C,N,O,F,S,Cl} set.
        try:
            o3a = rdMolAlign.GetO3A(prb, ref)
        except (ValueError, RuntimeError):
            o3a = rdMolAlign.GetCrippenO3A(prb, ref)
        o3a.Align()
        return 1.0 - _espsim_combo(prb, ref)
