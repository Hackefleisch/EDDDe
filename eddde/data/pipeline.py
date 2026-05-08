"""Materialize dataset stages with cascading staleness checks.

For each dataset, stages are built in order (SMILES -> CONFORMERS ->
ELEKTRONN_COEFFS). A stage is rebuilt when its producer version or any
input hash has changed. Because each stage's manifest records the
upstream artifact hash as an input, a single version bump anywhere in
the chain cascades to rebuild every downstream stage.
"""
from __future__ import annotations

from pathlib import Path

import eddde
from .. import SEED
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
SMILES_FILTER_VERSION = "v3-elements-saltstripped-minheavy3"

# Heavy-atom floor for the project-wide filter. Molecules below this are dropped
# before any downstream stage runs. Several baselines (B9/B10 USR family) require
# >= 3 heavy atoms; B6 topological torsion needs >= 4-atom paths and others
# (B11 eSim, B14 Chemprop) degenerate on tiny graphs. See CLAUDE.md.
MIN_HEAVY_ATOMS = 3

# Test-mode downsampling. When set (via the --test-mode CLI flag in
# eddde/__main__.py), every dataset's SMILES stage is randomly downsampled to at
# most TEST_MODE_SIZE rows after the project-wide filters run. Seeded with the
# project SEED so the sample is stable across runs and downstream caches do not
# rebuild on every test invocation. The size+seed are appended to the SMILES
# stage version so test-mode and full-mode artifacts invalidate each other when
# toggled.
TEST_MODE_SIZE: int | None = None


# Worker-process state, populated by _filter_init.
_WORKER_CHEM = None
_WORKER_CHOOSER = None
_WORKER_SUPPORTED: frozenset[int] = frozenset()


def _filter_init(supported: frozenset[int]) -> None:
    """Pool initializer: import RDKit + create the salt chooser once per worker."""
    global _WORKER_CHEM, _WORKER_CHOOSER, _WORKER_SUPPORTED
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    _WORKER_CHEM = Chem
    _WORKER_CHOOSER = rdMolStandardize.LargestFragmentChooser()
    _WORKER_SUPPORTED = supported


def _filter_one(item: tuple[int, str]) -> tuple[int, str | None, str | None, bool]:
    """Worker: parse SMILES once, run salt-strip + element + heavy-atom checks.

    Returns (idx, new_smiles_or_None, drop_reason_or_None, salts_stripped).
    drop_reason ∈ {"unparseable", "unsupported", "too_small", None}.
    """
    Chem = _WORKER_CHEM
    chooser = _WORKER_CHOOSER
    supported = _WORKER_SUPPORTED
    idx, smi = item

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return idx, None, "unparseable", False

    largest = chooser.choose(mol)
    stripped = largest.GetNumAtoms() < mol.GetNumAtoms()
    new_smi = Chem.MolToSmiles(largest) if stripped else smi

    # H is always in the supported set, so heavy atoms alone are sufficient —
    # skipping AddHs avoids a per-mol allocation.
    if not all(a.GetAtomicNum() in supported for a in largest.GetAtoms()):
        return idx, None, "unsupported", stripped

    if largest.GetNumHeavyAtoms() < MIN_HEAVY_ATOMS:
        return idx, None, "too_small", stripped

    return idx, new_smi, None, stripped


def _filter_and_normalize(csv: Path, ds_id: str) -> None:
    """Salt-strip + drop unsupported elements + drop too-small mols, in one parallel pass.

    Replaces three sequential single-core passes that each parsed every SMILES
    independently. Behaviour is preserved: same drop set, same logging shape.
    See CLAUDE.md for the rationale behind each filter.
    """
    import multiprocessing

    import pandas as pd

    df = pd.read_csv(csv)
    n = len(df)
    if n == 0:
        return

    items = list(enumerate(df["smiles"].tolist()))
    supported = frozenset(elektronn_runner.supported_elements())
    n_workers = max(1, min(eddde.N_WORKERS, n))

    new_smiles = list(df["smiles"])
    keep = [True] * n
    n_stripped = 0
    drops: dict[str, list[int]] = {"unparseable": [], "unsupported": [], "too_small": []}

    with multiprocessing.Pool(n_workers, initializer=_filter_init, initargs=(supported,)) as pool:
        for idx, new_smi, reason, stripped in pool.imap_unordered(_filter_one, items, chunksize=500):
            if stripped:
                n_stripped += 1
            if reason is None:
                new_smiles[idx] = new_smi
            else:
                keep[idx] = False
                drops[reason].append(idx)

    df["smiles"] = new_smiles
    if n_stripped:
        print(f"  [{ds_id}:smiles] stripped salts from {n_stripped} molecule(s)")

    def _log_drop(reason: str, message: str) -> None:
        idxs = drops[reason]
        if not idxs:
            return
        ids = [str(df.iloc[i]["id"]) for i in idxs]
        preview = ", ".join(ids[:10]) + ("..." if len(ids) > 10 else "")
        print(f"  [{ds_id}:smiles] {message.format(n=len(ids))}: {preview}")

    _log_drop("unparseable", "dropped {n} molecule(s) with unparseable SMILES")
    _log_drop("unsupported", "dropped {n} molecule(s) with unsupported atoms")
    _log_drop("too_small", f"dropped {{n}} molecule(s) with < {MIN_HEAVY_ATOMS} heavy atoms")

    df[keep].to_csv(csv, index=False)


def _downsample_for_test_mode(csv: Path, ds_id: str) -> None:
    """Randomly downsample the SMILES CSV to at most TEST_MODE_SIZE rows."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(csv)
    if TEST_MODE_SIZE is None or len(df) <= TEST_MODE_SIZE:
        return
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(df), size=TEST_MODE_SIZE, replace=False)
    idx.sort()  # preserve original ordering for readability
    df.iloc[idx].to_csv(csv, index=False)
    print(f"  [{ds_id}:smiles] test-mode: downsampled {len(df)} -> {TEST_MODE_SIZE} (seed={SEED})")


def _stage_version(ds: Dataset, stage: Stage) -> str:
    if stage == Stage.SMILES:
        v = f"{ds.version}+{SMILES_FILTER_VERSION}"
        if TEST_MODE_SIZE is not None:
            v += f"+test:{TEST_MODE_SIZE}:{SEED}"
        return v
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
    # Load ElektroNN weights outside the timed block so the ~12 s one-off
    # weight load doesn't get billed to the first dataset's compute_time.
    # Idempotent — subsequent datasets pay nothing.
    if stage == Stage.ELEKTRONN_COEFFS:
        elektronn_runner.prewarm()
    with timed() as t:
        if stage == Stage.SMILES:
            ds.build_smiles(out)
            _filter_and_normalize(out, ds.id)
            _downsample_for_test_mode(out, ds.id)
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
