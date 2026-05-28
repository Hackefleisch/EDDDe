"""EXP-6: Scaffold Hopping (PROJECT_PLAN.md §5.8).

Question: Does a method retrieve actives with diverse scaffolds, not just
analogs of the query?

Two retrieval protocols rolled into one experiment, both per target:

Sub-experiment A — standard retrieval (all 37 ChEMBL_II targets)
  Draw 5 random query actives. Rank the entire pool (actives + decoys minus
  query) by ascending distance. Compute M-EF5, M-SCAFEF5, M-SCAFRATIO.

Sub-experiment B — leave-one-scaffold-out (targets with >=5 distinct active
                   scaffolds only)
  For each active scaffold S in target T:
    queries = all actives with S in their scaffold set
    pool    = decoys + actives with S NOT in their scaffold set
  Compute M-RECALL-HELDOUT-SCAFFOLD at top 1 %, 5 %, 10 %.

Scaffold partitions come pre-computed in the upstream Riniker-Landrum data
(see eddde/data/sources/rinlan.py); we inherit them rather than re-deriving
Bemis-Murcko, so the "scaffold" definition is the one the benchmark
intended. Membership is many-to-many (Schuffenhauer hierarchy: one active
can belong to multiple scaffold buckets), stored in the dataset's sidecar
`cache/datasets/<id>/scaffolds.json` as {chembl_id: [scaf_id, ...]}.

Status: Step 5 of scratch/exp6_implementation_plan.md. Both sub-
experiments implemented:
  A: M-EF5, M-SCAFEF5, M-SCAFRATIO over 5 random query draws per target.
  B: M-RECALL-HELDOUT-SCAFFOLD at 1/5/10 % via leave-one-scaffold-out on
     targets with >= 5 distinct active scaffolds. Multi-bucket actives
     appear as queries once per scaffold in their set — each (S, query)
     pair tests one "perspective" on the chemistry.

Output files written to `out/`:
  retrieval_standard.csv   (sub-experiment A, RETRIEVAL_COLS_EXP6 schema)
  retrieval_heldout.csv    (sub-experiment B, RETRIEVAL_COLS_EXP6 schema;
                            "seed" column carries the held-out scaffold S)
  metrics.json             (combined metric values)
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .. import SEED
from ..cache import hash_file, is_stale, write_manifest
from ..data.base import Stage, dataset_dir
from ..data.sources.rinlan import RINLAN_DATASET_IDS
from ..methods.distance import pairwise_matrix
from . import retrieval_common as rc
from .base import result_dir


# 5 random query actives per target — matches PROJECT_PLAN.md §5.8.
N_QUERY_DRAWS = 5

# Top-k percentage for the sub-experiment A enrichment factor. Wider than
# EXP-3a/3b's 1 % because Riniker-Landrum targets have far fewer molecules
# (~100-1500 vs WelQrate's tens of thousands); the top 1 % cutoff would land
# inside the first 5-10 ranks for small targets and bake in undue variance.
EF_PERCENT_STANDARD = 5.0

# Sub-experiment B eligibility floor and the percent cutoffs to report.
MIN_SCAFFOLDS_FOR_HELDOUT = 5
RECALL_PERCENTS = (1.0, 5.0, 10.0)
# Which cutoff feeds the un-suffixed primary metric. 5 % matches M-EF5 and
# is the spec's recommended default (plan §4.3 stratification note).
RECALL_PRIMARY_PERCENT = 5.0

# Curve-cache filenames, one per retrieval CSV (write_enrichment_summary
# would collide otherwise — the helper writes to a single npz per call).
NPZ_STANDARD = "enrichment_curve_standard.npz"
NPZ_HELDOUT  = "enrichment_curve_heldout.npz"


# Extended schema: retrieval_common.RETRIEVAL_COLS plus the scaffold sets of
# the retrieved active and the query, JSON-encoded so SCAFEF/SCAFRATIO are
# recomputable straight from the CSV without re-loading scaffolds.json.
RETRIEVAL_COLS_EXP6 = (
    *rc.RETRIEVAL_COLS,
    "scaffold_ids",
    "query_scaffold_ids",
)


def _seeded_rng(dataset_id: str, draw_idx: int) -> np.random.Generator:
    """Deterministic RNG keyed on (global SEED, dataset, draw index).

    Stable across machines/processes so reruns reproduce the same queries.
    Same pattern as exp3b_muv._seeded_rng.
    """
    h = hashlib.sha256(f"{SEED}|{dataset_id}|{draw_idx}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _load_scaffold_map(dataset_id: str) -> dict[str, list[int]]:
    """Read the dataset's scaffolds.json sidecar (per-active scaffold lists).

    Returns an empty dict if the sidecar is missing — keeps the experiment
    safe to instantiate on datasets without scaffold annotation.
    """
    p = dataset_dir(dataset_id) / "scaffolds.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


class Exp6ScaffoldHopping:
    id = "EXP-6"
    version = "v0.5-plots"
    datasets = RINLAN_DATASET_IDS

    metric_direction = {
        "M-EF5":                     +1,
        "M-SCAFEF5":                 +1,
        "M-SCAFRATIO":               +1,
        "M-RECALL-HELDOUT-SCAFFOLD": +1,
    }

    # metric_datasets stays unset for now: M-EF5/SCAFEF5/SCAFRATIO apply to
    # every target, M-RECALL-HELDOUT-SCAFFOLD will gain a per-target
    # eligibility map (≥5 distinct active scaffolds) in Step 5. SUMMARY.md
    # treats absent metrics as "—".

    def run(self, method, stage_data, embeddings, dataset_id: str, out: Path) -> dict[str, Any]:
        df: pd.DataFrame = stage_data[Stage.SMILES].copy()
        df["id"] = df["id"].astype(str)
        out.mkdir(parents=True, exist_ok=True)

        mol_ids: list[str] = df["id"].tolist()
        activity: dict[str, int] = dict(zip(df["id"], df["activity"].astype(int)))
        active_ids: list[str] = [m for m in mol_ids if activity[m] == 1]
        n_actives_total = len(active_ids)

        scaffold_map = _load_scaffold_map(dataset_id)

        retrieval_rows: list[dict] = []
        seed_ef5: list[float] = []
        seed_scafef5: list[float] = []
        seed_scafratio: list[float] = []

        if active_ids:
            # Independent seeded draws (collisions are vanishingly rare at
            # typical ChEMBL_II active counts of 100-1000 per target; matches
            # the MUV pattern for deterministic reproducibility).
            draw_queries: list[str] = [
                str(_seeded_rng(dataset_id, d).choice(active_ids))
                for d in range(N_QUERY_DRAWS)
            ]
            # One batched pairwise_matrix call: (N_QUERY_DRAWS × n_pool).
            # The full pool includes the query itself; we slice out its
            # self-column from each row below.
            D = pairwise_matrix(method, embeddings, draw_queries, mol_ids)
            mol_col_of = {m: j for j, m in enumerate(mol_ids)}
        else:
            draw_queries = []
            D = None
            mol_col_of = {}

        for draw_idx in range(N_QUERY_DRAWS):
            if not active_ids:
                seed_ef5.append(float("nan"))
                seed_scafef5.append(float("nan"))
                seed_scafratio.append(float("nan"))
                continue

            query_id = draw_queries[draw_idx]
            query_scaffolds = scaffold_map.get(query_id, [])
            self_col = mol_col_of[query_id]
            row = D[draw_idx]
            cand_mask = np.arange(len(mol_ids)) != self_col
            candidates = [mol_ids[j] for j in range(len(mol_ids)) if j != self_col]
            distances = row[cand_mask]
            order = np.argsort(distances, kind="stable")

            n_total = len(candidates)
            n_actives = n_actives_total - 1  # query removed

            # Per-scaffold active counts over THIS query's pool (query removed
            # so its scaffolds don't inflate the random baseline).
            scaffold_active_counts: dict[int, int] = defaultdict(int)
            for cid in active_ids:
                if cid == query_id:
                    continue
                for s in scaffold_map.get(cid, []):
                    scaffold_active_counts[s] += 1

            cutoff = max(1, int(np.ceil(n_total * EF_PERCENT_STANDARD / 100.0)))
            retrieved_scaffold_sets: list[list[int]] = []

            active_ranks: list[int] = []
            for rank_0, idx in enumerate(order):
                cid = candidates[idx]
                if activity[cid] == 1:
                    rank = rank_0 + 1
                    active_ranks.append(rank)
                    cid_scaf = scaffold_map.get(cid, [])
                    if rank <= cutoff:
                        retrieved_scaffold_sets.append(cid_scaf)
                    retrieval_rows.append({
                        "seed": draw_idx,
                        "query_id": query_id,
                        "active_id": cid,
                        "rank": rank,
                        "distance": float(distances[idx]),
                        "n_total": n_total,
                        "n_actives_in_pool": n_actives,
                        "scaffold_ids": json.dumps(cid_scaf),
                        "query_scaffold_ids": json.dumps(query_scaffolds),
                    })

            ef5 = rc.ef_at_percent(active_ranks, n_total, n_actives,
                                   percent=EF_PERCENT_STANDARD)
            scafef5 = rc.scafef_at_percent(retrieved_scaffold_sets,
                                           scaffold_active_counts,
                                           n_total,
                                           percent=EF_PERCENT_STANDARD)
            # M-SCAFRATIO: per spec §5.2 it's literally SCAFEF / EF — measures
            # whether the scaffolds-recovered enrichment outpaces the actives-
            # recovered one (i.e. recovered actives are more scaffold-diverse
            # than chance). Per-draw division, then mean across draws — keeps
            # the SE behaviour consistent with the other metrics.
            if ef5 != 0.0 and not (np.isnan(ef5) or np.isnan(scafef5)):
                scafratio = scafef5 / ef5
            else:
                scafratio = float("nan")

            seed_ef5.append(ef5)
            seed_scafef5.append(scafef5)
            seed_scafratio.append(scafratio)

        std_csv = out / "retrieval_standard.csv"
        pd.DataFrame(retrieval_rows, columns=list(RETRIEVAL_COLS_EXP6)).to_csv(
            std_csv, index=False)
        rc.write_enrichment_summary(out, std_csv, npz_name=NPZ_STANDARD)

        metrics: dict = {}
        metrics.update(rc.metric_entry("M-EF5",       seed_ef5))
        metrics.update(rc.metric_entry("M-SCAFEF5",   seed_scafef5))
        metrics.update(rc.metric_entry("M-SCAFRATIO", seed_scafratio))

        # --- Sub-experiment B: leave-one-scaffold-out ---------------------
        # Eligibility: targets must have at least MIN_SCAFFOLDS_FOR_HELDOUT
        # distinct active scaffolds. Ineligible targets skip B entirely;
        # M-RECALL-HELDOUT-SCAFFOLD is simply absent from their metrics.json
        # and the SUMMARY table reads "—" for those cells.
        distinct_scaffolds = sorted({s for sids in scaffold_map.values() for s in sids})
        heldout_rows: list[dict] = []
        recall_per_query: dict[float, list[float]] = {p: [] for p in RECALL_PERCENTS}

        if len(distinct_scaffolds) >= MIN_SCAFFOLDS_FOR_HELDOUT and active_ids:
            decoy_ids = [m for m in mol_ids if activity[m] == 0]
            for s in distinct_scaffolds:
                queries_s = [a for a in active_ids if s in scaffold_map.get(a, [])]
                held_out_s = [a for a in active_ids if s not in scaffold_map.get(a, [])]
                if not queries_s or not held_out_s:
                    # Scaffold S where every active is also in some other bucket
                    # that ends up being identical to S's bucket, or the
                    # complement is empty — skip the round (no signal).
                    continue

                pool_s = decoy_ids + held_out_s
                held_out_set = set(held_out_s)
                n_total = len(pool_s)
                n_targets = len(held_out_s)

                # Queries-S and pool-S are disjoint by construction (queries
                # have S in their scaffold set; held-out actives do not), so
                # no self-column to remove.
                D = pairwise_matrix(method, embeddings, queries_s, pool_s)

                for i, q in enumerate(queries_s):
                    q_scafs = scaffold_map.get(q, [])
                    row = D[i]
                    order = np.argsort(row, kind="stable")

                    ranks_of_held_out: list[int] = []
                    for rank_0, idx in enumerate(order):
                        cid = pool_s[idx]
                        if cid in held_out_set:
                            rank = rank_0 + 1
                            ranks_of_held_out.append(rank)
                            heldout_rows.append({
                                "seed": s,
                                "query_id": q,
                                "active_id": cid,
                                "rank": rank,
                                "distance": float(row[idx]),
                                "n_total": n_total,
                                "n_actives_in_pool": n_targets,
                                "scaffold_ids": json.dumps(scaffold_map.get(cid, [])),
                                "query_scaffold_ids": json.dumps(q_scafs),
                            })

                    for percent in RECALL_PERCENTS:
                        recall_per_query[percent].append(
                            rc.recall_at_percent(ranks_of_held_out, n_total,
                                                 n_targets, percent=percent)
                        )

            heldout_csv = out / "retrieval_heldout.csv"
            pd.DataFrame(heldout_rows, columns=list(RETRIEVAL_COLS_EXP6)).to_csv(
                heldout_csv, index=False)
            rc.write_enrichment_summary(out, heldout_csv, npz_name=NPZ_HELDOUT)

            for percent in RECALL_PERCENTS:
                key = f"M-RECALL-HELDOUT-SCAFFOLD_at{int(percent)}pct"
                metrics.update(rc.metric_entry(key, recall_per_query[percent]))
            # Primary alias for the headline metric (un-suffixed) at the
            # spec-recommended 5 % cutoff. Pulled from the suffixed mean/SE
            # so there's exactly one source of truth.
            primary_key = f"M-RECALL-HELDOUT-SCAFFOLD_at{int(RECALL_PRIMARY_PERCENT)}pct"
            metrics["M-RECALL-HELDOUT-SCAFFOLD"]    = metrics[primary_key]
            metrics["M-RECALL-HELDOUT-SCAFFOLD_se"] = metrics[f"{primary_key}_se"]

        (out / "metrics.json").write_text(rc.metrics_to_json(metrics))
        return metrics

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _eligible_heldout_datasets(self, method_ids: list[str]) -> list[str]:
        """Datasets where at least one method emitted retrieval_heldout.csv.

        Eligibility is decided per-target at run-time (>= 5 active scaffolds);
        we detect it post-hoc from the presence of the CSV rather than re-
        reading scaffolds.json so plots stay in sync with what actually ran.
        """
        return [
            ds for ds in self.datasets
            if any(
                (result_dir(self.id, m, ds) / "retrieval_heldout.csv").exists()
                for m in method_ids
            )
        ]

    def _plot_ef5_vs_scafef5(self, plots_dir: Path, method_ids: list[str]) -> None:
        """One point per (method, dataset): x = M-EF5, y = M-SCAFEF5.

        The y = x diagonal separates "scaffold-tracking matches actives-
        tracking" (on the line) from "more scaffold-diverse than analog-
        biased" (above the line). A horizontal line at 1.0 marks the random
        baseline for SCAFEF5 alone.
        """
        import matplotlib.pyplot as plt

        # Stable color per method across plots — uses matplotlib's tab cycle
        # so we don't fix a palette that fights themes downstream.
        cmap = plt.get_cmap("tab20")
        method_color = {m: cmap(i % 20) for i, m in enumerate(method_ids)}

        fig, ax = plt.subplots(figsize=(7, 7))
        any_point = False
        xs_all: list[float] = []
        ys_all: list[float] = []
        for m_id in method_ids:
            xs: list[float] = []
            ys: list[float] = []
            for ds_id in self.datasets:
                p = result_dir(self.id, m_id, ds_id) / "metrics.json"
                if not p.exists():
                    continue
                data = json.loads(p.read_text())
                ef5 = data.get("M-EF5")
                scafef5 = data.get("M-SCAFEF5")
                if ef5 is None or scafef5 is None:
                    continue
                xs.append(ef5)
                ys.append(scafef5)
            if not xs:
                continue
            any_point = True
            xs_all.extend(xs)
            ys_all.extend(ys)
            ax.scatter(xs, ys, color=method_color[m_id], label=m_id, s=40,
                       alpha=0.75, edgecolor="black", linewidth=0.4)

        if not any_point:
            plt.close(fig)
            return

        lo = min(0.0, min(xs_all), min(ys_all))
        hi = max(max(xs_all), max(ys_all)) * 1.05
        ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", linewidth=0.8,
                label="y = x (analog = scaffold)")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.6,
                   label="SCAFEF random baseline")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("M-EF5 (active enrichment, top 5 %)")
        ax.set_ylabel("M-SCAFEF5 (distinct-scaffold enrichment, top 5 %)")
        ax.set_title(f"{self.id}: EF5 vs SCAFEF5 — scaffold tracking vs analog tracking")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(plots_dir / "ef5_vs_scafef5.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def make_plots(self, exp_results_dir: Path, method_ids: list[str]) -> None:
        plots_dir = exp_results_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        input_hashes: dict[str, str] = {}
        for m_id in method_ids:
            for ds_id in self.datasets:
                for fname in ("retrieval_standard.csv", "retrieval_heldout.csv",
                              "metrics.json"):
                    p = result_dir(self.id, m_id, ds_id) / fname
                    if p.exists():
                        input_hashes[f"{m_id}/{ds_id}/{fname}"] = hash_file(p)

        if not input_hashes:
            return

        sentinel = plots_dir / "enrichment_curves_standard.png"
        if not is_stale(sentinel, self.version, input_hashes):
            print(f"  [{self.id}] plots fresh")
            return

        print(f"  [{self.id}] generating plots...")

        # 1) Sub-A enrichment curves (log-FPR vs TPR) — one panel per target.
        rc.plot_enrichment_curves(
            self.id, plots_dir, method_ids, self.datasets,
            csv_name="retrieval_standard.csv",
            npz_name=NPZ_STANDARD,
            output_name="enrichment_curves_standard.png",
            title_suffix=" — sub-A standard retrieval",
        )

        # 2) Methods × datasets heatmaps, one PNG per metric. The headline
        #    is M-SCAFRATIO (scaffold-diversity heatmap from plan §7.2);
        #    the rest are diagnostic.
        rc.plot_metric_heatmap(
            self.id, plots_dir, method_ids, self.datasets,
            metrics=["M-EF5", "M-SCAFEF5", "M-SCAFRATIO",
                     "M-RECALL-HELDOUT-SCAFFOLD"],
        )

        # 3) EF5 vs SCAFEF5 scatter — y = x is the "analog-tracking equals
        #    scaffold-tracking" line; points above it favour hops over analogs.
        self._plot_ef5_vs_scafef5(plots_dir, method_ids)

        # 4) Sub-B held-out enrichment curves — only for eligible targets.
        eligible = self._eligible_heldout_datasets(method_ids)
        if eligible:
            rc.plot_enrichment_curves(
                self.id, plots_dir, method_ids, eligible,
                csv_name="retrieval_heldout.csv",
                npz_name=NPZ_HELDOUT,
                output_name="enrichment_curves_heldout.png",
                title_suffix=" — sub-B leave-one-scaffold-out",
            )

        write_manifest(sentinel, version=self.version, inputs=input_hashes,
                       compute_time=0.0, dataset_size=0)

    def collect_results(self, method_ids: list[str]) -> pd.DataFrame:
        rows = []
        for m_id in method_ids:
            for ds_id in self.datasets:
                p = result_dir(self.id, m_id, ds_id) / "metrics.json"
                if not p.exists():
                    continue
                data = json.loads(p.read_text())
                for metric in self.metric_direction:
                    if metric in data:
                        rows.append({"method": m_id, "dataset": ds_id,
                                     "metric": metric, "value": data[metric]})
        return pd.DataFrame(rows)
