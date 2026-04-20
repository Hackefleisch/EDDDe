"""Materialize dataset stages with cascading staleness checks.

For each dataset, stages are built in order (SMILES -> CONFORMERS ->
ELEKTRONN_COEFFS). A stage is rebuilt when its producer version or any
input hash has changed. Because each stage's manifest records the
upstream artifact hash as an input, a single version bump anywhere in
the chain cascades to rebuild every downstream stage.
"""
from __future__ import annotations

from ..cache import (
    Manifest,
    hash_file,
    is_stale,
    manifest_path,
    timed,
    write_manifest,
)
from . import conformers, elektronn_runner
from .base import STAGE_ORDER, Dataset, Stage, stage_path


def _stage_version(ds: Dataset, stage: Stage) -> str:
    if stage == Stage.SMILES:
        return ds.version
    if stage == Stage.CONFORMERS:
        return ds.version if ds.has_native_conformers else conformers.VERSION
    if stage == Stage.ELEKTRONN_COEFFS:
        return elektronn_runner.VERSION
    raise ValueError(stage)


def _stage_inputs(ds: Dataset, stage: Stage) -> dict:
    if stage == Stage.SMILES:
        return {}
    if stage == Stage.CONFORMERS:
        smiles = stage_path(ds.id, Stage.SMILES)
        return {"smiles_output_hash": hash_file(smiles)}
    if stage == Stage.ELEKTRONN_COEFFS:
        conf = stage_path(ds.id, Stage.CONFORMERS)
        return {"conformers_output_hash": hash_file(conf)}
    raise ValueError(stage)


def _upstream_chain_time(ds: Dataset, stage: Stage) -> float:
    if stage == Stage.SMILES:
        return 0.0
    prev = Stage.SMILES if stage == Stage.CONFORMERS else Stage.CONFORMERS
    m = Manifest.load(manifest_path(stage_path(ds.id, prev)))
    return m.chain_time() if m else 0.0


def _build_stage(ds: Dataset, stage: Stage) -> None:
    out = stage_path(ds.id, stage)
    out.parent.mkdir(parents=True, exist_ok=True)
    with timed() as t:
        if stage == Stage.SMILES:
            ds.build_smiles(out)
        elif stage == Stage.CONFORMERS:
            smiles = stage_path(ds.id, Stage.SMILES)
            if ds.has_native_conformers:
                ds.build_native_conformers(smiles, out)
            else:
                conformers.generate(smiles, out)
        elif stage == Stage.ELEKTRONN_COEFFS:
            elektronn_runner.generate(stage_path(ds.id, Stage.CONFORMERS), out)
        else:
            raise ValueError(stage)

    write_manifest(
        out,
        version=_stage_version(ds, stage),
        inputs=_stage_inputs(ds, stage),
        compute_time=t["seconds"],
        upstream_compute_time=_upstream_chain_time(ds, stage),
    )


def build_up_to(ds: Dataset, target: Stage, verbose: bool = True) -> None:
    """Build or rebuild all stages up to and including `target`."""
    for stage in sorted(Stage, key=lambda s: STAGE_ORDER[s]):
        if STAGE_ORDER[stage] > STAGE_ORDER[target]:
            return
        out = stage_path(ds.id, stage)
        expected_version = _stage_version(ds, stage)
        expected_inputs = _stage_inputs(ds, stage)
        if not is_stale(out, expected_version, expected_inputs):
            if verbose:
                print(f"  [{ds.id}:{stage.value}] fresh")
            continue
        if verbose:
            print(f"  [{ds.id}:{stage.value}] rebuilding...")
        _build_stage(ds, stage)
