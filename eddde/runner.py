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

import numpy as np
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
from .data.base import STAGE_ORDER, Stage, dataset_size, stage_path
from .data.pipeline import build_up_to
from .experiments import EXPERIMENTS
from .experiments.base import RESULTS_ROOT, result_dir
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
        dataset_size=dataset_size(ds_id),
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
        dataset_size=dataset_size(ds_id),
    )


def _write_summary_md(method_ids: list[str]) -> None:
    """Write results/SUMMARY.md — per-experiment metric table + cross-metric average rank."""
    lines: list[str] = ["# EDDDe Benchmark Summary\n"]

    all_avg_ranks: dict[str, list[float]] = {m: [] for m in method_ids}
    all_time_per_mol: dict[str, list[float]] = {m: [] for m in method_ids}

    for exp_id, exp in EXPERIMENTS.items():
        if not hasattr(exp, "collect_results") or not hasattr(exp, "metric_direction"):
            continue

        df = exp.collect_results(method_ids)
        if df.empty:
            continue

        metrics = list(exp.metric_direction.keys())
        n_datasets = len(exp.datasets)

        # --- Per-metric stats: mean ± SE and coverage ---
        stat_rows: dict[str, dict[str, str]] = {m: {} for m in method_ids}
        rank_rows: dict[str, dict[str, float]] = {m: {} for m in method_ids}

        for metric in metrics:
            direction = exp.metric_direction[metric]
            sub = df[df["metric"] == metric]

            method_vals: dict[str, list[float]] = {}
            for m_id in method_ids:
                vals = sub[sub["method"] == m_id]["value"].dropna().tolist()
                method_vals[m_id] = vals

            # Mean ± SE with coverage
            for m_id in method_ids:
                vals = method_vals[m_id]
                n = len(vals)
                if n == 0:
                    stat_rows[m_id][metric] = f"— (0/{n_datasets})"
                elif n == 1:
                    stat_rows[m_id][metric] = f"{vals[0]:.3f} (1/{n_datasets})"
                else:
                    mean = float(np.mean(vals))
                    se = float(np.std(vals, ddof=1) / np.sqrt(n))
                    stat_rows[m_id][metric] = f"{mean:.3f}±{se:.3f} ({n}/{n_datasets})"

            # Rank (lower rank = better). Null → worst rank = n_methods + 1
            mean_vals: list[tuple[str, float | None]] = []
            for m_id in method_ids:
                vals = method_vals[m_id]
                mean_vals.append((m_id, float(np.mean(vals)) if vals else None))

            # Sort: nonnull first, ordered by direction, then nulls
            nonnull = [(m, v) for m, v in mean_vals if v is not None]
            nonnull.sort(key=lambda x: -direction * x[1])  # best first
            null_methods = [m for m, v in mean_vals if v is None]

            for rank_idx, (m_id, _) in enumerate(nonnull, start=1):
                rank_rows[m_id][metric] = float(rank_idx)
            for m_id in null_methods:
                rank_rows[m_id][metric] = float(len(method_ids) + 1)

        # Accumulate average ranks
        for m_id in method_ids:
            ranks = list(rank_rows[m_id].values())
            if ranks:
                all_avg_ranks[m_id].append(float(np.mean(ranks)))

        # Average rank per method within this experiment
        exp_avg_rank: dict[str, float] = {}
        for m_id in method_ids:
            ranks = list(rank_rows[m_id].values())
            exp_avg_rank[m_id] = float(np.mean(ranks)) if ranks else float("nan")

        sorted_methods = sorted(method_ids, key=lambda m: exp_avg_rank.get(m, float("nan")))

        # --- Per-method end-to-end time per molecule (chain time of the
        # experiment manifest, averaged across this experiment's datasets). ---
        time_per_mol: dict[str, list[float]] = {m: [] for m in method_ids}
        for m_id in method_ids:
            for ds_id in exp.datasets:
                marker = result_dir(exp_id, m_id, ds_id) / "metrics.json"
                em = Manifest.load(manifest_path(marker))
                if em and em.dataset_size:
                    time_per_mol[m_id].append(em.chain_time_per_mol())
            all_time_per_mol[m_id].extend(time_per_mol[m_id])

        # Build markdown table
        lines.append(f"\n## {exp_id}\n")
        header_cols = ["Method"] + [f"{m}" for m in metrics] + ["Avg rank", "s/mol"]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

        for m_id in sorted_methods:
            metric_cells = [stat_rows[m_id].get(m, "—") for m in metrics]
            avg_r = exp_avg_rank[m_id]
            avg_str = f"{avg_r:.2f}" if not np.isnan(avg_r) else "—"
            times = time_per_mol[m_id]
            time_str = f"{float(np.mean(times)):.3g}" if times else "—"
            lines.append("| " + " | ".join([m_id] + metric_cells + [avg_str, time_str]) + " |")

        dir_str = ", ".join(f"{m}({'up' if d > 0 else 'down'})" for m, d in exp.metric_direction.items())
        lines.append(f"\n*Directions: {dir_str}. s/mol: end-to-end chain time divided by dataset size, averaged across datasets.*\n")

    # --- Cross-experiment aggregate ---
    if any(v for v in all_avg_ranks.values()):
        lines.append("\n## Cross-experiment average rank\n")
        lines.append("| Method | Avg rank (all experiments) | s/mol |")
        lines.append("| --- | --- | --- |")
        overall: list[tuple[str, float]] = []
        for m_id in method_ids:
            ranks = all_avg_ranks[m_id]
            overall.append((m_id, float(np.mean(ranks)) if ranks else float("nan")))
        overall.sort(key=lambda x: x[1])
        for m_id, avg in overall:
            avg_str = f"{avg:.2f}" if not np.isnan(avg) else "—"
            times = all_time_per_mol[m_id]
            time_str = f"{float(np.mean(times)):.3g}" if times else "—"
            lines.append(f"| {m_id} | {avg_str} | {time_str} |")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\nSummary written to {RESULTS_ROOT / 'SUMMARY.md'}")


def main() -> None:
    max_stage = _max_needed_stage()

    if STAGE_ORDER[max_stage] >= STAGE_ORDER[Stage.ELEKTRONN_COEFFS]:
        from .data import elektronn_runner
        print("Pre-loading ElektroNN weights (one-off, excluded from per-dataset timing)...")
        elektronn_runner.prewarm()

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
        if hasattr(exp, "make_plots"):
            exp.make_plots(RESULTS_ROOT / exp_id, list(METHODS.keys()))
        if hasattr(exp, "summarize"):
            exp.summarize(RESULTS_ROOT / exp_id, list(METHODS.keys()))

    _write_summary_md(list(METHODS.keys()))


if __name__ == "__main__":
    main()
