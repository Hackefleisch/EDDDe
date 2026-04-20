# EDDDe — Electron Density Derived Descriptors

A benchmarking framework for electron-density-based molecular similarity. The central question: do distances derived from DFT-level electron density representations track functional molecular similarity better than established methods?

[ElektroNN](https://github.com/Hackefleisch/ElektroNN) produces per-atom coefficient matrices of shape `(n_atoms, 127)` — basis-function fits to the electron density computed at DFT level. EDDDe takes those matrices, condenses them into fixed-size embeddings via various schemes ("methods under test", MUTs), and benchmarks the resulting distances against 17 established baselines across 6 experiments covering chemical series smoothness, electronic sensitivity, virtual screening retrieval, activity cliffs, bioisostere recognition, and scaffold hopping.

The full experimental design is in [PROJECT_PLAN.md](PROJECT_PLAN.md).

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
  runner.py                  # main() — stages → embeddings → experiments
  data/
    base.py                  # Stage enum, Dataset base class
    conformers.py            # default: RDKit ETKDGv3 + MMFF94
    elektronn_runner.py      # ElektroNN integration (stub)
    pipeline.py              # build_up_to(dataset, stage)
    sources/                 # one file per dataset
  methods/
    base.py                  # Method protocol, embedding cache
    baselines/               # one file per baseline group
    muts/                    # one file per MUT condensing scheme
  experiments/               # one file per experiment (EXP-1 … EXP-6)
```

Each cached artifact has a sidecar `*.manifest.json` storing producer version, input hashes, compute time, and accumulated upstream cost — so the full pipeline cost for any result is an O(1) lookup.

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

The runner checks every stage, embedding, and experiment result for staleness and runs only what needs updating. Adding a new method, dataset, or experiment and re-running will produce incremental results without touching anything already cached.

---

## Extending

**Add a baseline or MUT** — create a class in `eddde/methods/baselines/` or `eddde/methods/muts/` with `id`, `version`, `needs` (a `Stage`), `embed_dataset`, and `distance`. Register it in `eddde/methods/__init__.py`.

**Add a dataset** — subclass `Dataset` in `eddde/data/sources/`, implement `build_smiles` (and `build_native_conformers` if the dataset ships 3D structures). Register in `eddde/data/__init__.py`.

**Add an experiment** — implement the `Experiment` protocol in `eddde/experiments/`, declare which dataset IDs it runs on. Register in `eddde/experiments/__init__.py`.

New dependencies go in `pyproject.toml` — don't silently assume they are present.

---

## Status

| Component | Status |
|-----------|--------|
| Framework skeleton | done |
| B1 ECFP4 | done |
| S1 n-alkanes (EXP-1 series) | done |
| EXP-1 M-MONO metric | done |
| Remaining baselines (B2–B17) | pending |
| Remaining datasets / series | pending |
| ElektroNN integration | pending |
| MUT implementations | pending |
