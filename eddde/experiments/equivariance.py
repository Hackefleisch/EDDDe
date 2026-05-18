"""EXP-EQUIVAR: MUT equivariance audit (PROJECT_PLAN.md §3.2.1).

For every MUT, four distances per S0 base molecule:

  d_rot       = sym_distance(emb(orig),    emb(rot))         actual rotation distance
  d_rot_floor = sym_distance(emb(rot_ideal), emb(rot))       ElektroNN noise floor, MUT units
  d_perm      = sym_distance(emb(orig),    emb(perm))        permutation invariance
  d_id        = sym_distance(emb(orig),    emb(orig))        identity

`rot_ideal` is computed inside the experiment: take `coeff_orig` (cached
ElektroNN output for the unrotated geometry) and multiply by the Wigner
D-matrix for the irreps `14x0e + 14x1o + 5x2e + 4x3o + 2x4e` (PROJECT_PLAN.md
§3.2.2). The result is what `coeff_rot` would be if ElektroNN were perfectly
SO(3)-equivariant. Feeding it through `method.embed_dataset()` gives the
embedding the MUT *would* produce on perfect rotated features.

`d_rot_floor` then compares the MUT's embedding under perfect-rotated input
vs. real ElektroNN-rerun input. The two inputs are the *same* rotated
molecule; any difference at the embedding is ElektroNN's noise translated
through the MUT's pooling. That's the baseline `d_rot` should be read
against — not the feature-space Frobenius residual, which lives in
different units:

  d_rot ≈ d_rot_floor       → MUT is rotation-invariant; what you see is ElektroNN noise
  d_rot ≫ d_rot_floor       → MUT itself is rotation-leaky (e.g. raw 127-d mean on l>0 blocks)

Distances are symmetrised — `0.5 · (d(a, b) + d(b, a))` — so naturally
symmetric MUTs incur no extra cost (no-op) and asymmetric ones (B8-style
alignment optimisers, if any future MUT adopts that pattern) get the same
treatment EXP-2 already applies to its self-distance matrices.

Metrics are the **max** across the 5 fixture molecules — a single failing
base dominates the score, which is the right framing for a contract: any
failure is a failure. Per-base values are persisted to `raw.csv` for
diagnosis. `method_filter` restricts the experiment to MUTs, so baselines
never embed on S0.

A **feature-space noise floor** — `‖D(R)·coeff_orig − coeff_rot‖_F` per
base — is also computed and written to `results/EXP-EQUIVAR/noise_floor.json`.
It tests ElektroNN's equivariance directly (and, by proxy, that its SH
convention matches e3nn's per
[docs/strain_b_e3nn_central_atom.md §3](../../docs/strain_b_e3nn_central_atom.md)).
An order-1 relative residual there would point at a convention mismatch
needing a per-`l` basis transform. The SUMMARY preamble shows the worst base.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .. import SEED
from ..data.base import Stage
from ..data.sources.audit_fixture import base_ids, variant_id
from .base import RESULTS_ROOT, result_dir


# Irreps spec for ElektroNN's per-atom 127-d output. Locked in PROJECT_PLAN.md
# §3.2.2 and docs/strain_b_e3nn_central_atom.md §1.
_ELEKTRONN_IRREPS_STR = "14x0e + 14x1o + 5x2e + 4x3o + 2x4e"


def _sym_distance(method, e1, e2) -> float:
    return 0.5 * (float(method.distance(e1, e2)) + float(method.distance(e2, e1)))


def _audit_rotation_matrix() -> np.ndarray:
    """Reconstruct the rotation matrix S0 uses for its `_rot` variants.

    Must consume the seeded RNG in the same order as
    `AuditFixture.build_native_conformers` — `Rotation.random` is the first
    draw from the SEED-initialised generator. If that ordering changes,
    update this helper alongside it.
    """
    rng = np.random.default_rng(SEED)
    return Rotation.random(random_state=rng).as_matrix()


def _wigner_d_matrix() -> np.ndarray:
    import e3nn.o3 as o3
    import torch

    R = _audit_rotation_matrix()
    irreps = o3.Irreps(_ELEKTRONN_IRREPS_STR)
    return irreps.D_from_matrix(torch.from_numpy(R)).numpy()


def _compute_noise_floor(stage_data: dict, D: np.ndarray) -> dict[str, dict[str, float]]:
    """ElektroNN equivariance residual per base: ‖D(R)·coeff_orig − coeff_rot‖_F."""
    coeffs: dict[str, np.ndarray] = stage_data[Stage.ELEKTRONN_COEFFS]["coefficients"]
    out: dict[str, dict[str, float]] = {}
    for base in base_ids():
        c_orig = coeffs[variant_id(base, "orig")]
        c_rot  = coeffs[variant_id(base, "rot")]
        predicted = c_orig @ D.T
        residual = float(np.linalg.norm(predicted - c_rot, ord="fro"))
        magnitude = float(np.linalg.norm(c_orig, ord="fro"))
        out[base] = {
            "residual_frobenius": residual,
            "feature_magnitude": magnitude,
            "relative_residual": residual / magnitude if magnitude > 0 else 0.0,
        }
    return out


def _build_ideal_stage_data(stage_data: dict, D: np.ndarray) -> dict:
    """Synthetic stage_data with Wigner-rotated coefficients per base.

    Mol ids are `audit_{base}_rot_ideal`. Positions, adjacency, and pairwise
    distances are sourced from the cached `_rot` variants (already correct
    for the rotated geometry); coefficients are replaced with
    `coeff_orig @ D^T` so they reflect ElektroNN's *expected* equivariant
    output rather than the actual rerun.
    """
    coeffs_in = stage_data[Stage.ELEKTRONN_COEFFS]["coefficients"]
    adj_in    = stage_data[Stage.ELEKTRONN_COEFFS]["adjacencies"]
    dist_in   = stage_data[Stage.ELEKTRONN_COEFFS]["distances"]
    confs_in: dict[str, Any] = stage_data.get(Stage.CONFORMERS, {})

    out_coeffs: dict[str, np.ndarray] = {}
    out_adj:    dict[str, np.ndarray] = {}
    out_dist:   dict[str, np.ndarray] = {}
    out_confs:  dict[str, Any] = {}

    for base in base_ids():
        ideal_id = ideal_variant_id(base)
        rot_id   = variant_id(base, "rot")
        orig_id  = variant_id(base, "orig")

        out_coeffs[ideal_id] = coeffs_in[orig_id] @ D.T
        # Adjacency + pairwise distance are rotation-invariant; copy from
        # either orig or rot (both must agree). Pull from rot so the synthetic
        # mol is internally consistent with its rotated positions.
        out_adj[ideal_id]  = adj_in[rot_id]
        out_dist[ideal_id] = dist_in[rot_id]
        if confs_in:
            out_confs[ideal_id] = confs_in[rot_id]

    synthetic: dict = {
        Stage.ELEKTRONN_COEFFS: {
            "coefficients": out_coeffs,
            "adjacencies": out_adj,
            "distances": out_dist,
        }
    }
    if confs_in:
        synthetic[Stage.CONFORMERS] = out_confs
    if Stage.SMILES in stage_data:
        synthetic[Stage.SMILES] = stage_data[Stage.SMILES]
    return synthetic


def ideal_variant_id(base: str) -> str:
    return f"audit_{base}_rot_ideal"


def _noise_floor_path() -> Path:
    return RESULTS_ROOT / "EXP-EQUIVAR" / "noise_floor.json"


class ExpEquivariance:
    id = "EXP-EQUIVAR"
    version = "v3-rot-floor"
    datasets = ["S0"]

    metric_direction = {
        "M-EQUIVAR-ROT":       -1,
        "M-EQUIVAR-ROT-FLOOR": -1,
        "M-EQUIVAR-PERM":      -1,
        "M-EQUIVAR-ID":        -1,
    }

    # Per-process flag: prevents recomputing the noise floor for every MUT.
    # The file on disk is the authoritative cache across sessions.
    _noise_floor_done: bool = False

    def method_filter(self, method) -> bool:
        return method.is_mut

    def _ensure_noise_floor(self, stage_data: dict, D: np.ndarray) -> None:
        if self._noise_floor_done:
            return
        floor = _compute_noise_floor(stage_data, D)
        path = _noise_floor_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(floor, indent=2, sort_keys=True))
        self._noise_floor_done = True

    def run(self, method, stage_data, embeddings, dataset_id, out):
        D = _wigner_d_matrix()
        self._ensure_noise_floor(stage_data, D)

        ideal_embeddings = method.embed_dataset(_build_ideal_stage_data(stage_data, D))

        rows = []
        for base in base_ids():
            e_orig  = embeddings[variant_id(base, "orig")]
            e_rot   = embeddings[variant_id(base, "rot")]
            e_perm  = embeddings[variant_id(base, "perm")]
            e_ideal = ideal_embeddings[ideal_variant_id(base)]

            rows.append({
                "base": base,
                # Total rotation distance the MUT actually produces.
                "d_rot":       _sym_distance(method, e_orig, e_rot),
                # ElektroNN's noise floor expressed in MUT-distance units:
                # two embeddings of the *same* rotated molecule, one fed real
                # ElektroNN-rerun coefficients, one fed Wigner-derived perfect
                # equivariant coefficients. Any gap is ElektroNN's noise.
                "d_rot_floor": _sym_distance(method, e_ideal, e_rot),
                "d_perm":      _sym_distance(method, e_orig, e_perm),
                "d_id":        _sym_distance(method, e_orig, e_orig),
            })

        out.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out / "raw.csv", index=False)

        metrics = {
            "M-EQUIVAR-ROT":       float(df["d_rot"].max()),
            "M-EQUIVAR-ROT-FLOOR": float(df["d_rot_floor"].max()),
            "M-EQUIVAR-PERM":      float(df["d_perm"].max()),
            "M-EQUIVAR-ID":        float(df["d_id"].max()),
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
        return metrics

    def collect_results(self, method_ids: list[str]) -> pd.DataFrame:
        rows = []
        for m_id in method_ids:
            for ds_id in self.datasets:
                p = result_dir(self.id, m_id, ds_id) / "metrics.json"
                if not p.exists():
                    continue
                data = json.loads(p.read_text())
                for metric in self.metric_direction:
                    val = data.get(metric)
                    rows.append({
                        "method": m_id,
                        "dataset": ds_id,
                        "metric": metric,
                        "value": float(val) if val is not None and not np.isnan(val) else None,
                    })
        return pd.DataFrame(rows)

    def preamble(self, method_ids: list[str]) -> str | None:
        path = _noise_floor_path()
        if not path.exists():
            return None
        data: dict[str, dict[str, float]] = json.loads(path.read_text())
        if not data:
            return None
        worst = max(data.items(), key=lambda kv: kv[1]["relative_residual"])
        worst_base, worst_vals = worst
        return (
            f"*ElektroNN feature-space equivariance residual* "
            f"(`‖D(R)·coeff_orig − coeff_rot‖_F` across S0 bases): "
            f"worst is `{worst_base}` at "
            f"{worst_vals['relative_residual'] * 100:.2f}% of feature norm "
            f"(per-base breakdown in `results/EXP-EQUIVAR/noise_floor.json`). "
            f"For the MUT-comparable baseline see the `M-EQUIVAR-ROT-FLOOR` column.\n"
        )
