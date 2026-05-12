# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Scope

EDDDe is a **benchmarking framework for electron-density-based molecular similarity**. The upstream repo [ElektroNN](https://github.com/Hackefleisch/ElektroNN) produces DFT-theory-level electron density representations as coefficient matrices of shape `(n_atoms, 127)` — one row per atom, 127 basis-function coefficients per row.

This repo's job is twofold:

1. **Design and implement MUTs ("methods under test")**: different ways to condense each `(n_atoms, 127)` coefficient matrix into a fixed-size embedding paired with a distance function. MUT is a *family* of methods — we will try multiple condensing schemes, not just one.
2. **Benchmark them** against established baselines (topological fingerprints, 3D shape, learned embeddings, QM descriptors) to test whether electron-density-derived distances track functional similarity better than existing approaches.

## Planning Documents — Source of Truth

Two planning files drive all work. Read them before proposing design changes:

- [PROJECT_PLAN.md](PROJECT_PLAN.md) — **AI-optimized spec**. Structured tables: 17 baseline methods (B1–B17), 9 datasets (D1–D9), 6 experiments (EXP-1..EXP-6), ~20 metric definitions (M-*), output artifact layout, success criteria. Canonical reference — when in doubt about what to build, check here first.
- [experimental_plan.md](experimental_plan.md) — **Human-readable rationale**. Narrative version of the same plan explaining the "why" behind method choices, expected behaviors, and known dataset biases.

If a user asks for a change to the plan, update both files — structured for machine lookup, prose for reasoning.

## Environment

Uses `uv` with Python >= 3.10. The `elektronn` dep is fetched via SSH from `git@github.com:Hackefleisch/ElektroNN`, so install requires SSH access to that repo.

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
python -m eddde   # run the pipeline
```

Current dependencies ([pyproject.toml](pyproject.toml)): `numpy`, `scipy`, `pandas`, `rdkit`, `matplotlib`, `tqdm`, `ipykernel`, `elektronn`. Additional deps (e.g. `scikit-learn`, `pot`, `dscribe`, `mmpdb`) will be added as specific experiments are implemented — add them to `pyproject.toml`, don't silently assume they are present.

**Project-wide constants** live at the package root in [eddde/__init__.py](eddde/__init__.py):
- `SEED = 0xEDDDE` — used by every component that needs deterministic randomness (conformer embedding, test-mode downsampling, etc.) so reruns are reproducible and content-hashes stay stable.
- `N_WORKERS = os.cpu_count() or 1` — process-pool size for CPU-bound stages (SMILES filtering and conformer generation). Consumers read it as `eddde.N_WORKERS` at call time so the CLI override (`--num-workers`) propagates to every stage.

**CLI flags** ([eddde/__main__.py](eddde/__main__.py)):
- `--batch-size N` — ElektroNN GPU batch size (default 32).
- `--dataloader-workers N` — torch DataLoader workers for ElektroNN inference (default 0).
- `--num-workers N` — process pool size for SMILES filtering and conformer generation (default = `cpu_count`). Mutates `eddde.N_WORKERS`.
- `--test-mode` (with optional `--test-size N`, default 1000) — dev acceleration. Each dataset's post-filter SMILES is downsampled to ≤N rows via `Dataset.test_mode_subsample(df, n, rng)`. The default policy is uniform random; datasets override when uniform sampling would degrade downstream experiments — e.g. `_WelQrateBase.test_mode_subsample` keeps every active and fills the remaining budget with random inactives so each scaffold seed retains its full test-active set (uniform sampling at hit-rate ~1/300 leaves ≤3 actives in expectation, often zero in test). The downsample size, seed, and the dataset's `test_mode_version` string are appended to the SMILES stage version, so changing a dataset's policy only invalidates that dataset's test-mode cache and full-mode caches stay untouched. Pick one mode and stay in it for cache reuse — toggling test ↔ full forces a SMILES → conformers → elektronn rebuild for every dataset. See [eddde/data/pipeline.py](eddde/data/pipeline.py) (`TEST_MODE_SIZE`, `_downsample_for_test_mode`) and [eddde/data/base.py](eddde/data/base.py) (`Dataset.test_mode_subsample`, `test_mode_version`).

No test suite, linter, or formatter is configured. Don't invent lint/test commands; ask first if one is needed.

## Code Architecture

### Runner ([eddde/runner.py](eddde/runner.py))

`main()` runs three passes:
1. Build dataset stages up to the highest stage any registered method needs.
2. For each `(method, dataset)`, compute and cache embeddings if stale.
3. For each `(experiment, method, dataset)`, run the experiment if stale.

"Stale" means the producer's `version` string or any upstream artifact's content hash has changed since the manifest was written. Cascades automatically: bumping `conformers.VERSION` invalidates every downstream artifact for all datasets without native conformers.

When `pipeline._build_stage` is about to (re)build a dataset's `Stage.ELEKTRONN_COEFFS`, it calls `elektronn_runner.prewarm()` *outside* the `timed()` block so the ~12 s weight load is excluded from `compute_time`. `prewarm()` is idempotent (the model lives in `_MODEL_CACHE` inside `elektronn_runner` for the duration of the process), so only the first stale dataset pays the load cost. If every dataset's elektronn coeffs are already fresh, the model is never loaded — runs that just shuffle methods and experiments around cached coeffs start instantly.

### Caching ([eddde/cache.py](eddde/cache.py))

Every artifact (CSV, pkl, metrics.json) gets a sidecar `*.manifest.json` with:

```
version, inputs, output_hash, compute_time, upstream_compute_time,
timestamp, dataset_size, compute_time_per_mol,
distance_time_per_pair, n_pairs_benchmarked
```

`upstream_compute_time` accumulates the full chain cost so it can be read in O(1) from any manifest. `dataset_size` records the number of molecules at write time; `compute_time_per_mol` is `compute_time / dataset_size`. `Manifest.chain_time_per_mol()` divides the full chain time by `dataset_size` — this is the `s/mol` column in `results/SUMMARY.md`.

`distance_time_per_pair` (only populated on **embedding** manifests) is the mean wall-clock time of one `method.distance(e1, e2)` call, measured by [eddde/methods/base.py](eddde/methods/base.py) `benchmark_distance` immediately after embedding via a time-budgeted sample (target ~1 s, bounded `[20, 2000]` pairs, 5-call warmup, seeded). It surfaces in `results/SUMMARY.md` as the `s/pair` column. The benchmark re-runs whenever embeddings rebuild; existing pre-benchmark manifests get a one-shot in-place upgrade in `runner._embed_if_stale` so adopting the field doesn't require cascade rebuilds.

### Adding a method

Create a class in `eddde/methods/baselines/` or `eddde/methods/muts/` with these attributes and methods:

```python
id: str        # plan identifier verbatim — "B2", "MUT-mean", etc.
version: str   # bump manually when embed or distance logic changes
needs: Stage   # Stage.SMILES | Stage.CONFORMERS | Stage.ELEKTRONN_COEFFS

def embed_dataset(self, stage_data: dict) -> dict[str, Any]: ...
def distance(self, e1, e2) -> float: ...  # smaller = more similar
```

Register the instance in [eddde/methods/__init__.py](eddde/methods/__init__.py). On next run, the runner embeds + runs all experiments automatically.

**MUT training-data fairness:** Any learned component (on the embedding side, the distance side, or both) must be fit only on a held-out training split — never on the evaluation datasets (D3–D9). Early MUT variants (MUT-mean and siblings) keep the embedding non-trainable to characterise the raw coefficient signal first; trainable-embedding variants (attention pooling, graph-pooled GNN, etc.) are expected later and allowed under the same fairness rule. See [PROJECT_PLAN.md §3.2](PROJECT_PLAN.md) and [experimental_plan.md §2.6](experimental_plan.md) for the current variant list and rationale.

### Adding a dataset

Subclass `Dataset` in `eddde/data/sources/`. Implement `build_smiles(out: Path)` to write a CSV with columns `id`, `smiles`, and any experiment-specific columns (e.g. `position` for homologous series). If the dataset ships its own 3D structures, set `has_native_conformers = True` and implement `build_native_conformers(smiles_csv, out)` — the conformer stage version will then track `dataset.version` rather than the global `conformers.VERSION`, so it won't be affected by changes to the default conformer generator.

Register in [eddde/data/__init__.py](eddde/data/__init__.py).

### Adding an experiment

Implement the `Experiment` protocol in `eddde/experiments/`. Declare `datasets: list[str]` for the dataset IDs it runs on. The `run(method, stage_data, embeddings, dataset_id, out)` method writes raw results and a `metrics.json` to `out/`. Register in [eddde/experiments/__init__.py](eddde/experiments/__init__.py).

**Retrieval experiments** (EXP-3a/3b, and EXP-3c if/when implemented) share metric definitions, the raw-CSV schema, and most plots via [eddde/experiments/retrieval_common.py](eddde/experiments/retrieval_common.py). Use it for any new retrieval-style experiment to keep metrics consistent across them:
- Math helpers: `logauc`, `bedroc`, `ef_at_percent`, `dcg_at_k`, `nanmean`, `mean_se`.
- IO helpers: `RETRIEVAL_COLS` (canonical raw-CSV schema), `read_csv_or_empty`, `metric_entry`, `metrics_to_json`.
- Plot helpers: `plot_enrichment_curves`, `plot_metric_heatmap`, `plot_cumulative_recall`, `plot_rank_distributions` — each takes `(exp_id, plots_dir, method_ids, datasets)` and auto-shapes the subplot grid from `len(datasets)`, so the same code handles 9-target (WelQrate) and 17-target (MUV) layouts.

Experiments compose these in their own `run()` and `make_plots()` rather than inheriting from a shared base class — the differences across retrieval experiments are at the *query/pool definition* level (scaffold seeds vs. random query draws vs. one-active-per-target), which hooks cleanly into the call site without an abstraction layer.

### Data stages

Stages are built in order: `SMILES → CONFORMERS → ELEKTRONN_COEFFS`. Methods declare the highest stage they need via `needs`; stages above that are never built. Stage paths:

```
cache/datasets/{dataset_id}/smiles.csv
cache/datasets/{dataset_id}/conformers.pkl   # dict[mol_id -> rdkit.Mol with conformers]
cache/datasets/{dataset_id}/elektronn_coeffs.pkl  # {"coefficients": {mol_id: (n_atoms, 127)}, "adjacencies": {mol_id: (n_atoms, n_atoms)}, "distances": {mol_id: (n_atoms, n_atoms)}}
cache/embeddings/{method_id}/{dataset_id}.pkl     # dict[mol_id -> embedding]
results/EXP-X/{method_id}/{dataset_id}/metrics.json
```

`cache/` and `results/` are gitignored; fully regenerated on demand.

**Atom-index alignment.** Row `i` of `coefficients[mol_id]` corresponds to atom `i` of `conformers[mol_id].GetAtomWithIdx(i)`, and `adjacencies[mol_id][i, :]` / `distances[mol_id][i, :]` use the same indexing. Hydrogens are included (from `Chem.AddHs`), so `n_atoms` counts them — filter by `atom.GetAtomicNum() != 1` if a method wants heavy-atom-only. Positions and bond types are **not** duplicated into the coefficient bundle; a method that needs them should declare `needs = Stage.ELEKTRONN_COEFFS` and read them from `stage_data[Stage.CONFORMERS][mol_id]` via `mol.GetConformer().GetAtomPosition(i)` and `mol.GetAtomWithIdx(i).GetBonds()`.

**SMILES-stage element filter.** The SMILES stage applies a project-wide hard filter: any molecule containing an element outside ElektroNN's supported basis set — currently **{H, C, N, O, F, S, Cl}** (atomic numbers {1, 6, 7, 8, 9, 16, 17}, queried live from `MoleculeDataset({}).basisfunction_params`) — is dropped before any downstream stage runs. Every method, including SMILES-only baselines, sees the same filtered set; this keeps comparisons fair and prevents the `KeyError` that occurs when an ElektroNN-based method silently skips a molecule while a baseline keeps it. Consequences:
- Datasets lose any molecule with Br, I, P, B, Si, etc. (e.g. S6 drops bromobenzene, S8 drops 4-bromobenzoic acid). Log lines like `[S6:smiles] dropped 1 molecule(s) with unsupported atoms: s6_Br` appear at build time.
- **M-HALO-ORDER is dropped from EXP-2** (defined on F/Cl/Br; only F and Cl remain).
- When adding large real-world datasets (D3–D9) the drop rate may be non-trivial; revisit whether the filter should become per-dataset-opt-out before those experiments land.
- The filter lives in [eddde/data/pipeline.py](eddde/data/pipeline.py) (`_filter_and_normalize`, runs in a multiprocessing pool together with salt-stripping and the heavy-atom check); bump `SMILES_FILTER_VERSION` there to invalidate all cached SMILES CSVs when the supported set changes.

**SMILES-stage minimum-heavy-atom filter.** The SMILES stage also drops any molecule with fewer than `MIN_HEAVY_ATOMS` heavy atoms (currently **3**). Several baselines either error or produce degenerate output on tiny molecules: B9/B10 (USR, USRCAT) require ≥3 heavy atoms in RDKit; B6 (topological torsion) needs ≥4-atom paths; B11 (eSim) and B14 (Chemprop D-MPNN) degenerate on near-empty 3D shapes / edgeless graphs. Filtering at the dataset level keeps the "every method sees the same molecules" invariant intact instead of scattering per-method guards. Consequences:
- S1 loses methane and ethane; S2 loses methanol; S4 loses methylamine. Series stay long enough (10–11 points) for M-MONO/M-SMOOTH/M-LIN.
- The filter lives in [eddde/data/pipeline.py](eddde/data/pipeline.py) (`_filter_and_normalize`, `MIN_HEAVY_ATOMS`); bump `SMILES_FILTER_VERSION` to invalidate caches when changing the floor.

## Helper Scripts

Scripts live in [scripts/](scripts/). Run them from the project root with the virtual environment active.

### `scripts/draw_dataset.py`

Renders a PNG grid of 2D molecule images for any registered dataset.

```bash
python scripts/draw_dataset.py <dataset_id> [--out <path>] [--cols <n>] [--mol-size <px>]
```

- Builds the SMILES stage via the standard pipeline (cached; only runs if stale).
- Output defaults to `figures/<dataset_id>_molecules.png`.
- `--cols` controls grid width (default 4); `--mol-size` controls tile pixel size (default 200).

Example: `python scripts/draw_dataset.py S6 --cols 3`

---

## Conventions

- Distance convention: smaller = more similar. For similarity-based methods (Tanimoto, ROCS), convert as `distance = 1 - similarity`.
- Identifiers from the plan (B1, D3, EXP-4, M-BEDROC20, etc.) are load-bearing — use them verbatim in filenames, `id` fields, and result columns so cross-referencing stays trivial.
- Each MUT variant needs its own stable `id` (e.g. `MUT-mean`, `MUT-attention`) so benchmark outputs can distinguish condensing schemes.
- EXP-3c (DUD-E / D5) is **deferred** — see [PROJECT_PLAN.md §5.5](PROJECT_PLAN.md) and [experimental_plan.md §4.3c](experimental_plan.md). The dataset has known analog bias and property-matching shortcuts, EXP-3a + EXP-3b cover the same question more rigorously, and the compute cost (~1.15 M molecules) is 2–4× of D3+D4 combined. The spec is preserved in case a future contributor implements it; the paper must explicitly justify the omission.
- EXP-5 (bioisostere recognition) is the critical hypothesis test. If MUTs fail there, the core hypothesis is weakened regardless of other results.
- Statistical reporting: always pair p-values with effect sizes (Cliff's delta or matched-pairs rank-biserial). Never report a single cherry-picked metric.
- Conformer fairness: all 3D methods use the **same single conformer per molecule** — the lowest MMFF94 energy geometry found across 5 ETKDGv3 starting geometries. Each `Mol` in the conformers pkl has exactly one conformer. The global generator is in [eddde/data/conformers.py](eddde/data/conformers.py) — don't let any method silently generate its own. **Potential follow-up**: add a best-of-ensemble variant (more conformers, pick minimum pairwise distance) uniformly across all 3D methods to test whether conformational flexibility adds signal.
