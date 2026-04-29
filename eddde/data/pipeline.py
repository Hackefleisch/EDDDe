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
    read_blacklist,
    timed,
    write_manifest,
)
from . import conformers, elektronn_runner
from .base import STAGE_ORDER, Dataset, Stage, dataset_size, stage_path

# Bump to invalidate every cached SMILES CSV project-wide (cascades to downstream stages).
SMILES_FILTER_VERSION = "v2-elements-saltstripped"


def _strip_salts(csv: Path, ds_id: str) -> None:
    """Replace each SMILES with its largest fragment to drop salt counterions.

    Counterions (Na+, Cl-, ...) alter the electron density without biological
    relevance. Stripping them before _filter_unsupported_atoms prevents
    molecules with e.g. Na+ counterions from being dropped for the wrong reason.
    """
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    chooser = rdMolStandardize.LargestFragmentChooser()
    df = pd.read_csv(csv)
    modified = 0
    new_smiles = list(df["smiles"])
    for i, smi in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        largest = chooser.choose(mol)
        if largest.GetNumAtoms() < mol.GetNumAtoms():
            new_smiles[i] = Chem.MolToSmiles(largest)
            modified += 1
    df["smiles"] = new_smiles
    if modified:
        print(f"  [{ds_id}:smiles] stripped salts from {modified} molecule(s)")
    df.to_csv(csv, index=False)


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
            _strip_salts(out, ds.id)
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


def _patch_manifest(artifact: Path, new_output_hash: str, new_dataset_size: int, new_inputs: dict | None = None) -> None:
    """Update an existing manifest's output_hash, dataset_size, and optionally inputs in-place.

    Used by the sync pass to keep the manifest chain consistent after blacklist filtering
    without changing version or triggering downstream staleness on the next run.
    """
    m_path = manifest_path(artifact)
    m = Manifest.load(m_path)
    if m is None:
        return
    m.output_hash = new_output_hash
    m.dataset_size = new_dataset_size
    m.compute_time_per_mol = m.compute_time / new_dataset_size if new_dataset_size else 0.0
    if new_inputs:
        m.inputs.update(new_inputs)
    m_path.write_text(m.to_json())


def _sync_blacklisted(ds: Dataset, up_to: Stage, verbose: bool) -> None:
    """After all stages are built, remove blacklisted mol_ids from every stage artifact.

    Each stage (conformers, elektronn) appends to a per-dataset blacklist.txt when it
    cannot process a molecule. This function reads that blacklist, filters the SMILES CSV
    and all downstream PKL files in-place, then patches the manifest chain so that the
    next staleness check sees consistent hashes and does not spuriously rebuild anything.
    """
    import pickle

    import pandas as pd

    cache_dir = stage_path(ds.id, Stage.SMILES).parent
    blacklisted = read_blacklist(cache_dir)
    if not blacklisted:
        return

    ordered = sorted(Stage, key=lambda s: STAGE_ORDER[s])
    built = [s for s in ordered if STAGE_ORDER[s] <= STAGE_ORDER[up_to] and stage_path(ds.id, s).exists()]

    # Track updated output hashes so downstream manifest inputs stay in sync.
    current_hash: dict[Stage, str] = {}

    # --- SMILES ---
    smiles_path = stage_path(ds.id, Stage.SMILES)
    df = pd.read_csv(smiles_path)
    df_filtered = df[~df["id"].astype(str).isin(blacklisted)]
    removed = len(df) - len(df_filtered)
    if removed:
        df_filtered.to_csv(smiles_path, index=False)
        if verbose:
            print(f"  [{ds.id}:smiles] sync: removed {removed} blacklisted molecule(s)")
    current_hash[Stage.SMILES] = hash_file(smiles_path)
    _patch_manifest(smiles_path, current_hash[Stage.SMILES], len(df_filtered))

    # --- CONFORMERS ---
    if Stage.CONFORMERS in built:
        conf_path = stage_path(ds.id, Stage.CONFORMERS)
        with open(conf_path, "rb") as f:
            confs: dict = pickle.load(f)
        confs_filtered = {k: v for k, v in confs.items() if k not in blacklisted}
        removed = len(confs) - len(confs_filtered)
        if removed:
            with open(conf_path, "wb") as f:
                pickle.dump(confs_filtered, f)
            if verbose:
                print(f"  [{ds.id}:conformers] sync: removed {removed} blacklisted molecule(s)")
        current_hash[Stage.CONFORMERS] = hash_file(conf_path)
        _patch_manifest(
            conf_path,
            current_hash[Stage.CONFORMERS],
            len(confs_filtered),
            new_inputs={"smiles_output_hash": current_hash[Stage.SMILES]},
        )

    # --- ELEKTRONN_COEFFS ---
    if Stage.ELEKTRONN_COEFFS in built:
        elec_path = stage_path(ds.id, Stage.ELEKTRONN_COEFFS)
        with open(elec_path, "rb") as f:
            elec: dict = pickle.load(f)
        keep = set(elec["coefficients"].keys()) - blacklisted
        removed = len(elec["coefficients"]) - len(keep)
        if removed:
            elec_filtered = {key: {mid: arr for mid, arr in sub.items() if mid in keep} for key, sub in elec.items()}
            with open(elec_path, "wb") as f:
                pickle.dump(elec_filtered, f)
            if verbose:
                print(f"  [{ds.id}:elektronn_coeffs] sync: removed {removed} blacklisted molecule(s)")
        current_hash[Stage.ELEKTRONN_COEFFS] = hash_file(elec_path)
        _patch_manifest(
            elec_path,
            current_hash[Stage.ELEKTRONN_COEFFS],
            len(keep),
            new_inputs={"conformers_output_hash": current_hash[Stage.CONFORMERS]},
        )


def build_up_to(ds: Dataset, target: Stage, verbose: bool = True) -> None:
    """Build or rebuild all stages up to and including `target`, then sync the blacklist."""
    for stage in sorted(Stage, key=lambda s: STAGE_ORDER[s]):
        if STAGE_ORDER[stage] > STAGE_ORDER[target]:
            break
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
    _sync_blacklisted(ds, target, verbose)
