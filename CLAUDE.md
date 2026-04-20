# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Scope

EDDDe is being repurposed as a **benchmarking framework for electron-density-based molecular similarity**. The upstream repo [ElektroNN](https://github.com/Hackefleisch/ElektroNN) produces DFT-theory-level electron density representations as coefficient matrices of shape `(n_atoms, 127)` — one row per atom, 127 basis-function coefficients per row.

This repo's job is twofold:

1. **Design and implement MUTs ("methods under test")**: different ways to condense each `(n_atoms, 127)` coefficient matrix into a fixed-size embedding paired with a distance function. MUT is a *family* of methods — we will try multiple condensing schemes, not just one.
2. **Benchmark them** against established baselines (topological fingerprints, 3D shape, learned embeddings, QM descriptors) to test whether electron-density-derived distances track functional similarity better than existing approaches.

The codebase is currently empty — prior SVM/analysis scripts were removed in preparation for the rewrite. README.md still describes the old scope and is **stale**; trust [PROJECT_PLAN.md](PROJECT_PLAN.md) until the README is rewritten.

## Planning Documents — Source of Truth

Two planning files drive all work. Read them before proposing design changes:

- [PROJECT_PLAN.md](PROJECT_PLAN.md) — **AI-optimized spec**. Structured tables: 17 baseline methods (B1–B17), 9 datasets (D1–D9), 6 experiments (EXP-1..EXP-6), ~20 metric definitions (M-*), output artifact layout, success criteria. Canonical reference — when in doubt about what to build, check here first.
- [experimental_plan.md](experimental_plan.md) — **Human-readable rationale**. Narrative version of the same plan explaining the "why" behind method choices, expected behaviors, and known dataset biases.

The two are intended to stay in sync. If a user asks for a change to the plan, update both — structured for machine lookup, prose for reasoning.

## Environment

Uses `uv` with Python >= 3.10. The `elektronn` dep is fetched via SSH from `git@github.com:Hackefleisch/ElektroNN`, so install requires SSH access to that repo.

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

Current dependencies ([pyproject.toml](pyproject.toml)): `numpy`, `rdkit`, `matplotlib`, `tqdm`, `ipykernel`, `elektronn`. Additional deps (e.g. `scikit-learn`, `pot`, `dscribe`, `mmpdb`) will be added as specific experiments are implemented — add them to `pyproject.toml`, don't silently assume they are present.

No test suite, linter, or formatter is configured. Don't invent lint/test commands; ask first if one is needed.

## Architectural Expectations (from the plan)

When implementing the framework, follow the structure implied by [PROJECT_PLAN.md §7](PROJECT_PLAN.md):

- **MUTs and baselines share one interface**: every method, MUT or baseline, exposes `embed(SMILES) -> vector` and `distance(v1, v2) -> float`. Any condensing scheme we invent for the `(n_atoms, 127)` ElektroNN output must plug in behind this same interface.
- **Conformer fairness**: all 3D methods (B8–B11, B13, B15–B17, MUT) share one conformer ensemble — RDKit ETKDGv3, `numConfs=50`, `pruneRmsThresh=0.5`, MMFF94-minimized. Don't let any method silently use its own.
- **Results layout**: each experiment writes to `results/EXP-X/{raw,aggregated.csv,summary.csv,statistics.csv,plots,metadata.json}`. Intermediate embeddings/distance matrices go to `cache/{method_id}_{dataset_id}.npz` for rerun-without-recompute.
- **Statistical reporting**: always pair p-values with effect sizes (Cliff's delta or matched-pairs rank-biserial). Full metric suite per experiment, never a cherry-picked number.
- **Reproducibility**: fix seeds, record software versions in `metadata.json`.

## Conventions

- Distance convention: smaller = more similar. For similarity-based methods (Tanimoto, ROCS), convert as `distance = 1 - similarity`.
- Identifiers from the plan (B1, D3, EXP-4, M-BEDROC20, etc.) are load-bearing — use them verbatim in filenames, config keys, and result columns so cross-referencing stays trivial.
- Each MUT variant needs its own stable ID (e.g. `MUT-mean`, `MUT-attention`, ...) so benchmark outputs can distinguish condensing schemes.
- DUD-E (D5) results are **weighted below** WelQrate (D3) and MUV (D4) in final conclusions due to known analog bias — preserve this in any aggregation logic.
- EXP-5 (bioisostere recognition) is the critical hypothesis test. If MUTs fail there, the core hypothesis is weakened regardless of other results — don't bury that outcome in aggregate scores.

## Stale Files

- [README.md](README.md) describes the old SVM/ElektroNN scope and should be rewritten once the new framework takes shape. Don't treat it as current documentation.
