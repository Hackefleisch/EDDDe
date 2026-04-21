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

No test suite, linter, or formatter is configured. Don't invent lint/test commands; ask first if one is needed.

## Code Architecture

### Runner ([eddde/runner.py](eddde/runner.py))

`main()` runs three passes:
1. Build dataset stages up to the highest stage any registered method needs.
2. For each `(method, dataset)`, compute and cache embeddings if stale.
3. For each `(experiment, method, dataset)`, run the experiment if stale.

"Stale" means the producer's `version` string or any upstream artifact's content hash has changed since the manifest was written. Cascades automatically: bumping `conformers.VERSION` invalidates every downstream artifact for all datasets without native conformers.

### Caching ([eddde/cache.py](eddde/cache.py))

Every artifact (CSV, pkl, metrics.json) gets a sidecar `*.manifest.json` with `{version, inputs, output_hash, compute_time, upstream_compute_time}`. `upstream_compute_time` accumulates the full chain cost so it can be read in O(1) from any manifest.

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

### Adding a dataset

Subclass `Dataset` in `eddde/data/sources/`. Implement `build_smiles(out: Path)` to write a CSV with columns `id`, `smiles`, and any experiment-specific columns (e.g. `position` for homologous series). If the dataset ships its own 3D structures, set `has_native_conformers = True` and implement `build_native_conformers(smiles_csv, out)` — the conformer stage version will then track `dataset.version` rather than the global `conformers.VERSION`, so it won't be affected by changes to the default conformer generator.

Register in [eddde/data/__init__.py](eddde/data/__init__.py).

### Adding an experiment

Implement the `Experiment` protocol in `eddde/experiments/`. Declare `datasets: list[str]` for the dataset IDs it runs on. The `run(method, stage_data, embeddings, dataset_id, out)` method writes raw results and a `metrics.json` to `out/`. Register in [eddde/experiments/__init__.py](eddde/experiments/__init__.py).

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

## Conventions

- Distance convention: smaller = more similar. For similarity-based methods (Tanimoto, ROCS), convert as `distance = 1 - similarity`.
- Identifiers from the plan (B1, D3, EXP-4, M-BEDROC20, etc.) are load-bearing — use them verbatim in filenames, `id` fields, and result columns so cross-referencing stays trivial.
- Each MUT variant needs its own stable `id` (e.g. `MUT-mean`, `MUT-attention`) so benchmark outputs can distinguish condensing schemes.
- DUD-E (D5) results are **weighted below** WelQrate (D3) and MUV (D4) in final conclusions due to known analog bias.
- EXP-5 (bioisostere recognition) is the critical hypothesis test. If MUTs fail there, the core hypothesis is weakened regardless of other results.
- Statistical reporting: always pair p-values with effect sizes (Cliff's delta or matched-pairs rank-biserial). Never report a single cherry-picked metric.
- Conformer fairness: all 3D methods use the **same single conformer per molecule** — the lowest MMFF94 energy geometry found across 20 ETKDGv3 starting geometries. Each `Mol` in the conformers pkl has exactly one conformer. The global generator is in [eddde/data/conformers.py](eddde/data/conformers.py) — don't let any method silently generate its own. **Potential follow-up**: add a best-of-ensemble variant (20 conformers, pick minimum pairwise distance) uniformly across all 3D methods to test whether conformational flexibility adds signal.
