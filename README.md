# EDDDe — Electron Density Derived Descriptors

A benchmarking framework for electron-density-based molecular similarity. The central question: do distances derived from DFT-level electron density representations track functional molecular similarity better than established methods?

[ElektroNN](https://github.com/Hackefleisch/ElektroNN) produces per-atom coefficient matrices of shape `(n_atoms, 127)` — basis-function fits to the electron density computed at DFT level. EDDDe takes those matrices, condenses them into fixed-size embeddings via various schemes ("methods under test", MUTs), and benchmarks the resulting distances against 17 established baselines across 6 experiments covering chemical series smoothness, electronic sensitivity, virtual screening retrieval, activity cliffs, bioisostere recognition, and scaffold hopping.

The full experimental design is in [PROJECT_PLAN.md](PROJECT_PLAN.md). Human-readable rationale is in [experimental_plan.md](experimental_plan.md).

---

## Architecture

Every method — MUT or baseline — exposes the same two-function interface:

```python
method.embed_dataset(stage_data)  # -> dict[mol_id -> embedding]
method.distance(e1, e2)           # -> float  (smaller = more similar)
```

The runner materializes dataset stages (SMILES → conformers → ElektroNN coefficients), caches embeddings per `(method, dataset)`, then runs each registered experiment against each registered method. Everything is content-addressed: an artifact is only recomputed when its producer version or an upstream output hash has changed.

```
eddde/
  cache.py                   # manifest-based staleness checks
  runner.py                  # main() — stages → embeddings → experiments → SUMMARY.md
  data/
    base.py                  # Stage enum, Dataset base class
    conformers.py            # RDKit ETKDGv3 + MMFF94, lowest-energy single conformer
    elektronn_runner.py      # ElektroNN integration, model cache, supported-element set
    pipeline.py              # build_up_to(dataset, stage), SMILES-stage element filter
    sources/                 # one file per dataset (S1–S8 implemented)
  methods/
    base.py                  # Method protocol, embedding cache
    baselines/               # one file per baseline (B1–B7 implemented)
    muts/                    # one file per MUT condensing scheme (MUT-mean implemented)
  experiments/               # one file per experiment (EXP-1, EXP-2 implemented)
```

Each cached artifact has a sidecar `*.manifest.json` storing producer version, input hashes, compute time, and accumulated upstream cost — so the full pipeline cost for any result is an O(1) lookup. `results/SUMMARY.md` is regenerated on every run with per-experiment metric tables and a cross-experiment average-rank leaderboard.

**Element filter.** The SMILES stage applies a project-wide hard filter dropping molecules whose SMILES contains an element outside ElektroNN's supported basis set (`{H, C, N, O, F, S, Cl}`). This ensures every method — including SMILES-only baselines — sees the same molecule set. See [CLAUDE.md](CLAUDE.md) for details.

---

## Installation

Requires Python ≥ 3.10 and [uv](https://github.com/astral-sh/uv). The `elektronn` dependency is fetched via SSH from its private repository, so SSH access to `git@github.com:Hackefleisch/ElektroNN` is required.

```bash
git clone git@github.com:Hackefleisch/EDDDe.git
cd EDDDe
uv venv && source .venv/bin/activate
uv pip install -e .
```

## Running

```bash
python -m eddde
```

The runner checks every stage, embedding, and experiment result for staleness and runs only what needs updating. Adding a new method, dataset, or experiment and re-running produces incremental results without touching anything already cached.

---

## Extending

**Add a baseline or MUT** — create a class in `eddde/methods/baselines/` or `eddde/methods/muts/` with `id`, `version`, `needs` (a `Stage`), `embed_dataset`, and `distance`. Register it in `eddde/methods/__init__.py`.

**Add a dataset** — subclass `Dataset` in `eddde/data/sources/`, implement `build_smiles` (and `build_native_conformers` if the dataset ships 3D structures). Register in `eddde/data/__init__.py`.

**Add an experiment** — implement the `Experiment` protocol in `eddde/experiments/`, declare `datasets` and `metric_direction`. Optionally declare `metric_datasets` to specify which datasets each metric applies to (used by the SUMMARY writer for accurate coverage fractions). Register in `eddde/experiments/__init__.py`.

New dependencies go in `pyproject.toml` — don't silently assume they are present.

---

## Current results

Results below are from running EXP-1 and EXP-2 against B1–B7 and MUT-mean. All results are fully reproducible: `python -m eddde` regenerates them from scratch.

### EXP-1 — Homologous Series Smoothness (S1–S5)

Tests whether distances grow monotonically and smoothly with chain length across five homologous series (n-alkanes, n-alkanols, n-alkanoic acids, n-alkylamines, polyethylene glycols). Metrics: M-MONO (Spearman ρ with chain-length gap, ↑), M-SMOOTH (std dev of consecutive-distance ratios, ↓), M-LIN (R² of d vs chain position, ↑).

| Method | M-MONO | M-SMOOTH | M-LIN | Avg rank | s/mol |
| --- | --- | --- | --- | --- | --- |
| B7 | 0.862±0.033 | 0.679±0.135 | 0.831±0.025 | 2.00 | 0.0024 |
| MUT-mean | 0.678±0.017 | 0.222±0.038 | 0.840±0.010 | 2.00 | 0.558 |
| B5 | 0.711±0.027 | 0.105±0.020 | 0.582±0.157 | 2.33 | 0.000346 |
| B1–B6 | 0.47–0.57 | — | 0.30–0.63 | 5–8 | <0.001 |

MUT-mean achieves the best M-LIN score (0.840) and competitive M-MONO, but distances are smooth rather than linear — M-SMOOTH well below the B7 value. Topological fingerprints (B1–B6) show zero M-SMOOTH signal (their distance functions saturate as local neighborhoods become self-similar in long chains).

### EXP-2 — Functional Group Substitution Sensitivity (S6–S8)

Tests whether embeddings reflect electronic character of substituents across three probes: monosubstituted benzenes (S6, conjugated), monosubstituted cyclohexanes (S7, aliphatic control), and para-substituted benzoic acids with known Hammett σ_para values (S8). Metrics: M-HAMMETT-PAIR (Spearman ρ between |Δσ| and pairwise distance, ↑), M-HAMMETT-ABS (Spearman ρ between |σ| and distance from H-compound, ↑), M-SILHOUETTE-S6 (donor/acceptor/neutral clustering on conjugated scaffold, ↑), M-SILHOUETTE-S7 (same on aliphatic control — want low clustering, ↓).

| Method | M-HAMMETT-PAIR | M-HAMMETT-ABS | M-SILHOUETTE-S6 | M-SILHOUETTE-S7 | Avg rank | s/mol |
| --- | --- | --- | --- | --- | --- | --- |
| B4 | 0.400 | 0.713 | 0.066 | 0.041 | 3.00 | 0.00349 |
| B1 | 0.312 | 0.788 | 0.107 | 0.093 | 3.00 | 0.00666 |
| B2 | 0.260 | 0.788 | 0.081 | 0.067 | 3.25 | 0.00444 |
| B3 | 0.336 | 0.750 | 0.053 | 0.162 | 4.50 | 0.00298 |
| B7 | 0.233 | 0.669 | 0.215 | 0.240 | 5.00 | 0.00355 |
| B6 | −0.048 | 0.773 | 0.015 | 0.061 | 5.25 | 0.00471 |
| B5 | 0.218 | 0.604 | 0.160 | 0.171 | 5.75 | 0.00352 |
| MUT-mean | 0.228 | 0.285 | 0.027 | 0.082 | 6.25 | 0.485 |

EXP-2 results are a first look with only the simplest MUT variant (atom-mean pooling). MUT-mean does not yet show an advantage on electronic sensitivity — topological fingerprints outperform it on M-HAMMETT-PAIR. This is expected to improve as more informative condensing schemes (irrep-weighted, attention-pooled) are implemented. The Hammett scatter plots are the primary diagnostic at this stage.

Note: Br-containing molecules are dropped by the element filter (see above), reducing S6 to 11 and S8 to 9 compounds.

---

## Status

| Component | Status |
|---|---|
| Framework: runner, caching, manifest chain, SUMMARY writer | done |
| **Baselines** | |
| B1 ECFP4, B2 ECFP6, B3 FCFP4 | done |
| B4 MACCS keys, B5 Atom Pair, B6 Topological Torsion | done |
| B7 RDKit 2048-bit 2D descriptors (cosine distance) | done |
| B8–B14 (3D shape: ROCS/USR/USRCAT, pharmacophore, scaffold) | pending |
| B15–B17 (QM-descriptor baselines: Coulomb matrix, SOAP, Mol2Vec) | pending |
| **MUTs** | |
| MUT-mean (atom-mean → 127-d vector, Euclidean distance) | done |
| MUT-mean-cosine, MUT-mean-irrep-weighted, MUT-mean-mahalanobis | planned |
| Attention-pooled and graph-pooled variants | planned |
| **Datasets** | |
| S1–S5 (homologous series: alkanes, alkanols, acids, amines, PEGs) | done |
| S6 monosubstituted benzenes (11 after element filter) | done |
| S7 monosubstituted cyclohexanes (10) | done |
| S8 para-substituted benzoic acids — Hammett series (9 after element filter) | done |
| D3 WelQrate, D4 MUV, D5 DUD-E, D6 MMP-cliffs, D7–D8 bioisosteres, D9 Riniker-Landrum | pending |
| **Experiments** | |
| EXP-1 Homologous series smoothness (M-MONO, M-SMOOTH, M-LIN) | done |
| EXP-2 Functional group substitution sensitivity (M-HAMMETT-PAIR, M-SILHOUETTE) | done |
| EXP-3 Virtual screening retrieval (WelQrate, MUV, DUD-E) | pending |
| EXP-4 Activity cliff sensitivity | pending |
| EXP-5 Bioisostere recognition (critical hypothesis test) | pending |
| EXP-6 Scaffold hopping | pending |
