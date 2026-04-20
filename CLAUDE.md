# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Scope

EDDDe is the **downstream analytics layer** for molecular property prediction. Embedding generation, dataset parsing, and model inference live in a separate repo ([ElektroNN](https://github.com/Hackefleisch/ElektroNN)) pulled in as a git dependency via `uv`. EDDDe consumes ElektroNN outputs (pickled embedding dicts, distance/adjacency matrices) to run SVM classifiers, Gromov-Wasserstein distance analyses, and exploratory notebooks.

This scope change is recent — the repo's history contains files that predate the refactor. When adding features, assume embedding/model code does **not** belong here.

## Environment and Commands

Uses `uv` with Python >= 3.10. The `elektronn` dep is fetched via SSH from `git@github.com:Hackefleisch/ElektroNN`, so install requires SSH access to that repo.

```bash
uv venv && source .venv/bin/activate
uv pip install -e .           # editable dev install
```

No test suite, linter, or formatter is configured — `pyproject.toml` declares `pytest` under optional extras but no tests exist. Don't invent lint/test commands; ask first if one is needed.

Run scripts directly with the venv active:

```bash
python eddde/svm.py                                     # SVM training pipeline
python calc_gw_dist.py                                  # imported as a library, not a CLI
python analyze_qm40_correlations.py                     # QM40 property correlation analysis
python create_qm40_subset.py --target HOMO --size 10000 # stratified QM40 subsetting
python smiles_to_fingerprints.py --csv <path> --output <path>.npy
```

## Architecture

### Packaged module: [eddde/](eddde/)
Only one real file: [eddde/svm.py](eddde/svm.py). End-to-end SVM pipeline — loads `preds_extended.pkl` actives/inactives, mean-pools embeddings along the atom axis, grid-searches (`kernel`, `C`, `gamma`) with Stratified K-Fold on ROC-AUC, and persists model/scaler/results with timestamped filenames.

### Top-level scripts (not importable as a package)
- [calc_gw_dist.py](calc_gw_dist.py) — `calculate_gw_distance(...)` wraps `ot.gromov.fused_gromov_wasserstein2` with an `alpha` trade-off between structural (GW) and feature (W) distance.
- [analyze_qm40_correlations.py](analyze_qm40_correlations.py) — finds the 4-property subset of QM40 with minimal pairwise correlation (brute-force `itertools.combinations`).
- [create_qm40_subset.py](create_qm40_subset.py) — 2D stratified sampler over `(target property, heavy atom count)`.
- [smiles_to_fingerprints.py](smiles_to_fingerprints.py) — RDKit topological fingerprints keyed by `Zinc_id`, saved as a pickled dict inside `.npy`.

### Notebooks
[nb_gromov_wasserstein_analysis.ipynb](nb_gromov_wasserstein_analysis.ipynb), [nb_gromov_wasserstein_rot_conf.ipynb](nb_gromov_wasserstein_rot_conf.ipynb), [nb_q_benchmark_playground.ipynb](nb_q_benchmark_playground.ipynb), [nb_quanti_benchmark_playground.ipynb](nb_quanti_benchmark_playground.ipynb) are exploratory — treat them as scratch space, not reference implementations.

### Data layout ([data/](data/))
- `data/qsar/` — welQrate (binary) and SMRT (continuous) QSAR sets; each molecule has `coefficients.pkl` (N×127), `adjacency.pkl`, `distance.pkl` keyed by molecule ID (see [data/README.md](data/README.md)).
- `data/conformers/`, `data/rotated/` — 3D conformer inputs.
- QM40 is **not** in this repo; scripts point to `/media/iwe20/DataSSD/QM40_dataset/QM40_main.csv`.

## Hard-coded Paths — Expect to Patch

Several paths in the code assume another developer's machine or an external SSD:

- [eddde/svm.py](eddde/svm.py) references `/home/jderiz/EDDDe/data/qsar/embeddings/AID435034/...` for both input pickles and output artifacts. Parameterize or update before running.
- QM40 scripts hard-code `/media/iwe20/DataSSD/QM40_dataset/QM40_main.csv` as a default.

When touching these scripts, prefer accepting the path as a CLI arg (the QM40 scripts already do) rather than hard-coding a new one.

## Conventions

- Data exchange between ElektroNN and EDDDe is via pickled dicts keyed by molecule ID — preserve that contract when adding new consumers.
- Embeddings are per-atom `(N_atoms, D)` matrices; atom-axis mean-pooling before SVM is the current pattern (see [svm.py:27-28](eddde/svm.py#L27-L28)).
- `PROJECT_PLAN.md` and `experimental_plan.md` are working documents for the author's roadmap — read for context, don't treat as authoritative specs.
