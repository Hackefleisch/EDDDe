"""B8: Gaussian shape + color Tanimoto via RDKit's `rdShapeAlign`.

Open-source ROCS-equivalent: PAPER-algorithm Gaussian-volume overlay with
optional pharmacophore-feature ("color") scoring. The original B8 in the
plan points at OpenEye ROCS (commercial); this implementation uses the
RDKit port, which exposes the same shape-Tanimoto + color-Tanimoto outputs.

Distance is `1 - 0.5 * (shape_tanimoto + color_tanimoto)` (i.e. 1 - combo
Tanimoto). The optimizer is run with `opt_param=0.5` so the pose search
balances shape and color, matching ROCS combo convention.

Embedding: per-mol `Chem.Mol` (with the shared single conformer). The
`ShapeInput` object that `rdShapeAlign` needs is not picklable, so we build
it fresh in each `distance` call. Construction is ~0.04 ms vs. ~0.5-15 ms
for the alignment itself, so caching it would not move the needle.
"""
from __future__ import annotations

from typing import Any

from rdkit import Chem
from rdkit.Chem import rdShapeAlign

from ...data.base import Stage


class GaussianShapeAlign:
    id = "B8"
    version = "rdshapealign-combo-v1"
    needs = Stage.CONFORMERS

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        conformers: dict[str, Chem.Mol] = stage_data[Stage.CONFORMERS]
        return {mol_id: Chem.Mol(mol) for mol_id, mol in conformers.items()}

    def distance(self, e1: Any, e2: Any) -> float:
        ref_shape = rdShapeAlign.PrepareConformer(e1)
        probe_shape = rdShapeAlign.PrepareConformer(e2)
        shape_score, color_score, _ = rdShapeAlign.AlignShapes(
            ref_shape, probe_shape, 0.5
        )
        return 1.0 - 0.5 * (shape_score + color_score)
