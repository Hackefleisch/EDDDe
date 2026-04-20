"""Main runner.

Three passes per invocation:
  1. Materialize every dataset up to the highest stage any registered
     method needs. Stale stages rebuild; fresh ones are skipped.
  2. For each (method, dataset) pair where the method is applicable,
     (re)compute embeddings into cache/embeddings/{method}/{dataset}.pkl.
  3. For each (experiment, method, dataset), run the experiment if any
     input has changed since the last result manifest was written.

Staleness is decided by comparing the artifact's stored manifest against
the expected version + expected input hashes computed now. Because each
artifact's inputs include the upstream artifact's output hash, one bump
anywhere cascades.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from .cache import (
    Manifest,
    hash_file,
    is_stale,
    manifest_path,
    timed,
    write_manifest,
)
from .data import DATASETS
from .data.base import STAGE_ORDER, Stage, stage_path
from .data.pipeline import build_up_to
from .experiments import EXPERIMENTS
from .experiments.base import result_dir
from .methods import METHODS
from .methods.base import embedding_path, load_embeddings, save_embeddings


def _max_needed_stage() -> Stage:
    if not METHODS:
        return Stage.SMILES
    return max(METHODS.values(), key=lambda m: STAGE_ORDER[m.needs]).needs


def _load_stage_data(ds_id: str, up_to: Stage) -> dict:
    data: dict = {}
    if STAGE_ORDER[up_to] >= STAGE_ORDER[Stage.SMILES]:
        data[Stage.SMILES] = pd.read_csv(stage_path(ds_id, Stage.SMILES))
    if STAGE_ORDER[up_to] >= STAGE_ORDER[Stage.CONFORMERS]:
        with open(stage_path(ds_id, Stage.CONFORMERS), "rb") as f:
            data[Stage.CONFORMERS] = pickle.load(f)
    if STAGE_ORDER[up_to] >= STAGE_ORDER[Stage.ELEKTRONN_COEFFS]:
        with open(stage_path(ds_id, Stage.ELEKTRONN_COEFFS), "rb") as f:
            data[Stage.ELEKTRONN_COEFFS] = pickle.load(f)
    return data


def _embed_if_stale(method, ds_id: str, stage_data: dict) -> Path:
    path = embedding_path(method.id, ds_id)
    stage_file = stage_path(ds_id, method.needs)
    expected_version = method.version
    expected_inputs = {"stage_hash": hash_file(stage_file)}

    if not is_stale(path, expected_version, expected_inputs):
        print(f"  [embed {method.id} on {ds_id}] fresh")
        return path

    print(f"  [embed {method.id} on {ds_id}] computing...")
    with timed() as t:
        emb = method.embed_dataset(stage_data)
    save_embeddings(path, emb)

    stage_m = Manifest.load(manifest_path(stage_file))
    upstream = stage_m.chain_time() if stage_m else 0.0
    write_manifest(
        path,
        version=expected_version,
        inputs=expected_inputs,
        compute_time=t["seconds"],
        upstream_compute_time=upstream,
    )
    return path


def _run_experiment_if_stale(exp, method, ds_id: str, stage_data: dict) -> None:
    out = result_dir(exp.id, method.id, ds_id)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "metrics.json"

    emb_path = embedding_path(method.id, ds_id)
    expected_version = f"{exp.version}+{method.version}"
    expected_inputs = {"embeddings_hash": hash_file(emb_path)}

    if not is_stale(marker, expected_version, expected_inputs):
        print(f"  [{exp.id} x {method.id} on {ds_id}] fresh")
        return

    print(f"  [{exp.id} x {method.id} on {ds_id}] running...")
    embeddings = load_embeddings(emb_path)
    with timed() as t:
        exp.run(method, stage_data, embeddings, ds_id, out)

    emb_m = Manifest.load(manifest_path(emb_path))
    upstream = emb_m.chain_time() if emb_m else 0.0
    write_manifest(
        marker,
        version=expected_version,
        inputs=expected_inputs,
        compute_time=t["seconds"],
        upstream_compute_time=upstream,
    )


def main() -> None:
    max_stage = _max_needed_stage()

    print("=== Dataset stages ===")
    for ds_id, ds in DATASETS.items():
        print(f"[{ds_id}] target stage: {max_stage.value}")
        build_up_to(ds, max_stage)

    print("\n=== Experiments ===")
    for exp_id, exp in EXPERIMENTS.items():
        for ds_id in exp.datasets:
            if ds_id not in DATASETS:
                print(f"[{exp_id}] dataset {ds_id} not registered, skipping")
                continue
            stage_data = _load_stage_data(ds_id, max_stage)
            for m_id, method in METHODS.items():
                _embed_if_stale(method, ds_id, stage_data)
                _run_experiment_if_stale(exp, method, ds_id, stage_data)


if __name__ == "__main__":
    main()
