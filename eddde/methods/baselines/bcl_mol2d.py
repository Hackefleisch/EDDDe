"""B18: BCL::Mol2D atom-environment descriptor ([Vu et al. JCAMD 2019](https://doi.org/10.1007/s10822-019-00199-8)).

A 574-feature count vector indexing the atom-environment library that
BCL ships with the binary (compiled from ~900,000 drug-like molecules).
Each entry is the number of times a particular atom-type-encoded
1-bond-sphere environment appears in the input molecule. Topology only --
no 3D coordinates required -- so we depend on `Stage.SMILES` and the
upstream SMILES filter (heavy-atom minimum, supported-element check)
catches the only failure modes BCL has on small molecules.

Implementation: the BCL binary is a closed-source C++ application (its
license forbids redistribution, so we can't vendor or build via uv).
The path is read from the central `eddde.BCL_BIN` setting, which in
turn reads `BCL_BIN` from `eddde/local_settings.py` (gitignored). When
unset, the method does not register at all -- the soft-skip happens in
`eddde/methods/__init__.py` so the runner never sees B18 and there is
no empty-results-column to mistake for a failure. If `BCL_BIN` IS set
but the path doesn't exist on disk, we raise loudly: that means the
user opted in but the install is broken.

Workflow per `embed_dataset` call:
  1. Materialise the dataset's SMILES into a temporary SDF (no conformers
     needed; UMol2D is purely topological).
  2. Invoke `bcl.exe molecule:Properties -tabulate UMol2D` once for the
     whole SDF -- BCL itself batches, so we never pay per-molecule
     subprocess overhead.
  3. Parse the resulting "Index,UMol2D" TSV-like table back into a dict
     keyed by mol id (SDF write order is preserved by BCL).

Distance: cosine, computed via batched BLAS (`distances` override). Zero
vectors (would only occur for degenerate inputs the SMILES filter would
have already dropped) are mapped to distance 1 to avoid NaN.

Published defaults are used verbatim: `atom hashing type=Atom`,
`feature size=574`, `Atom environment height=1`. These are also UMol2D's
in-binary defaults, so calling it as bare `UMol2D` is paper-identical
without any explicit parameter pinning.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from ... import BCL_BIN
from ...data.base import Stage
from ..base import Method


UMOL2D_FEATURES = 574


def _bcl_binary_path() -> Path:
    # The conditional registration in `eddde/methods/__init__.py` should make
    # this path unreachable when BCL_BIN is None; the explicit check survives
    # ad-hoc usage (e.g. instantiating BCLMol2D() directly in a notebook).
    if not BCL_BIN:
        raise RuntimeError(
            "B18 (BCL::Mol2D) requires the BCL binary. Set BCL_BIN in "
            "eddde/local_settings.py (copy from local_settings.example.py); "
            "download the prebuilt release from "
            "https://github.com/BCLCommons/bcl/releases."
        )
    p = Path(BCL_BIN).expanduser()
    if not p.exists():
        raise RuntimeError(
            f"BCL_BIN points at {p}, which does not exist. Re-check the path."
        )
    return p


class BCLMol2D(Method):
    id = "B18"
    version = "umol2d-atom-h1-574-cosine-v1"
    needs = Stage.SMILES

    def embed_dataset(self, stage_data: dict) -> dict[str, Any]:
        bcl = _bcl_binary_path()
        df = stage_data[Stage.SMILES]
        mol_ids = df["id"].astype(str).tolist()
        smiles = df["smiles"].tolist()

        with tempfile.TemporaryDirectory(prefix="bcl_mol2d_") as td:
            tdp = Path(td)
            sdf_path = tdp / "in.sdf"
            tsv_path = tdp / "out.tsv"

            # SDWriter as a context manager flushes + closes cleanly even on
            # an exception during construction.
            with Chem.SDWriter(str(sdf_path)) as w:
                for mid, smi in zip(mol_ids, smiles):
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        raise ValueError(
                            f"B18: SMILES failed to parse for id={mid!r}: {smi!r}"
                        )
                    mol.SetProp("_Name", mid)
                    w.write(mol)

            subprocess.run(
                [str(bcl), "molecule:Properties",
                 "-input_filenames", str(sdf_path),
                 "-tabulate", "UMol2D",
                 "-output_table", str(tsv_path)],
                check=True, capture_output=True,
            )

            out: dict[str, np.ndarray] = {}
            with open(tsv_path) as f:
                next(f)  # "Index,UMol2D" header
                for line in f:
                    idx_str, vec_str = line.strip().split(",", 1)
                    vec = np.fromstring(vec_str, sep=" ", dtype=np.float32)
                    if vec.size != UMOL2D_FEATURES:
                        raise RuntimeError(
                            f"B18: expected {UMOL2D_FEATURES} features per row, "
                            f"got {vec.size} for SDF index {idx_str}"
                        )
                    out[mol_ids[int(idx_str)]] = vec
        return out

    def distances(self, embs_q: list[Any], embs_c: list[Any]) -> np.ndarray:
        Q = np.stack(embs_q).astype(np.float64)
        C = np.stack(embs_c).astype(np.float64)
        qn = np.linalg.norm(Q, axis=1, keepdims=True)
        cn = np.linalg.norm(C, axis=1, keepdims=True)
        # Map zero vectors (would only occur on degenerate input already
        # screened by the SMILES filter) to norm 1; the resulting cosine of
        # 0 -> distance 1 marks them as maximally dissimilar.
        qn[qn == 0] = 1.0
        cn[cn == 0] = 1.0
        sim = (Q / qn) @ (C / cn).T
        return 1.0 - sim
