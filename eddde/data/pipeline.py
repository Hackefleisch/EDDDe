"""Materialize dataset stages with cascading staleness checks.

For each dataset, stages are built in order (SMILES -> CONFORMERS ->
ELEKTRONN_COEFFS). A stage is rebuilt when its producer version or any
input hash has changed. Because each stage's manifest records the
upstream artifact hash as an input, a single version bump anywhere in
the chain cascades to rebuild every downstream stage.
"""
from __future__ import annotations

from pathlib import Path

from ..cache import (
    Manifest,
    hash_file,
    is_stale,
    manifest_path,
    timed,
    write_manifest,
)
from . import conformers, elektronn_runner
from .base import STAGE_ORDER, Dataset, Stage, dataset_size, stage_path

# Bump to invalidate every cached SMILES CSV project-wide (cascades to downstream stages).
SMILES_FILTER_VERSION = "v1-elektronn-elements"


def _filter_unsupported_atoms(csv: Path, ds_id: str) -> None:
    """Drop rows whose SMILES contains atoms outside ElektroNN's supported basis set.

    Project-wide hard filter: ensures every method (SMILES-only baselines and
    ElektroNN-based MUTs alike) sees the same molecule set. See CLAUDE.md.
    """
    import pandas as pd
    from rdkit import Chem

    supported = elektronn_runner.supported_elements()
    df = pd.read_csv(csv)
    keep = []
    dropped: list[str] = []
    for row in df.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None:
            keep.append(False)
            dropped.append(str(row.id))
            continue
        mol_h = Chem.AddHs(mol)
        ok = all(a.GetAtomicNum() in supported for a in mol_h.GetAtoms())
        keep.append(ok)
        if not ok:
            dropped.append(str(row.id))
    if dropped:
        preview = ", ".join(dropped[:10]) + ("..." if len(dropped) > 10 else "")
        print(f"  [{ds_id}:smiles] dropped {len(dropped)} molecule(s) with unsupported atoms: {preview}")
    df[keep].to_csv(csv, index=False)


def _stage_version(ds: Dataset, stage: Stage) -> str:
    if stage == Stage.SMILES:
        return f"{ds.version}+{SMILES_FILTER_VERSION}"
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
            _filter_unsupported_atoms(out, ds.id)
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
        dataset_size=dataset_size(ds.id),
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
