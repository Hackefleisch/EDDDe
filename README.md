# EDDDe — Electron Density Derived Descriptors

A benchmarking framework for electron-density-based molecular similarity. The central question: do distances derived from DFT-level electron density representations track functional molecular similarity better than established methods?

[ElektroNN](https://github.com/Hackefleisch/ElektroNN) produces per-atom coefficient matrices of shape `(n_atoms, 127)` — basis-function fits to the electron density computed at DFT level. EDDDe takes those matrices, condenses them into fixed-size embeddings via various schemes ("methods under test", MUTs), and benchmarks the resulting distances against established baselines across six experiments covering chemical-series smoothness, electronic sensitivity, virtual-screening retrieval, activity cliffs, bioisostere recognition, and scaffold hopping.

Source of truth for the experimental design: [PROJECT_PLAN.md](PROJECT_PLAN.md) (structured spec) and [experimental_plan.md](experimental_plan.md) (narrative rationale). Per-strain implementation references: [docs/](docs/). Always-fresh per-method numbers: [results/SUMMARY.md](results/SUMMARY.md).

---

## Installation

Requires Python ≥ 3.10 and [uv](https://github.com/astral-sh/uv). The `elektronn` dependency is fetched via SSH from a private repository, so SSH access to `git@github.com:Hackefleisch/ElektroNN` is required.

```bash
git clone git@github.com:Hackefleisch/EDDDe.git
cd EDDDe
uv venv && source .venv/bin/activate
uv pip install -e .
```

### Per-machine config: `eddde/local_settings.py`

Anything machine-specific (today: paths to optional external binaries) lives in `eddde/local_settings.py`, which is gitignored. Copy the template once after install:

```bash
cp eddde/local_settings.example.py eddde/local_settings.py
```

Everything in this file is optional. When a value is `None` or absent, the dependent method silently does not register — every other method runs unaffected, no empty result columns. Adding new optional external deps follows the `_setting(attr_name)` pattern in [eddde/__init__.py](eddde/__init__.py).

### Optional: BCL toolkit for B18 (BCL::Mol2D)

B18 calls the closed-source [BCL](https://github.com/BCLCommons/bcl) C++ binary. The license forbids redistribution, so it can't ship via uv or be vendored. To enable B18:

1. Download the prebuilt installer from [BCLCommons/bcl releases](https://github.com/BCLCommons/bcl/releases) (`bcl-4.3.1-Linux-x86_64.sh` at the time of writing), make it executable, and run it once.
2. In `eddde/local_settings.py`, set `BCL_BIN = "/abs/path/to/bcl.exe"`.

Skip this step entirely if you don't need B18; startup prints one line noting the skip and proceeds.

---

## Running

```bash
python -m eddde                      # full run
python -m eddde --test-mode          # dev mode: downsample every dataset to ≤1000 mols
python -m eddde --num-workers 8      # cap the CPU process pool (default: cpu_count)
python -m eddde --batch-size 16      # smaller GPU batch if ElektroNN OOMs
```

The runner checks every stage, embedding, and experiment for staleness and rebuilds only what changed. Adding a new method, dataset, or experiment and re-running produces incremental results without touching anything already cached.

**Flags.** `--batch-size N` (ElektroNN GPU batch, default 32). `--dataloader-workers N` (torch DataLoader workers, default 0). `--num-workers N` (process pool for SMILES filtering and conformer generation, default = `cpu_count`). `--test-mode` with optional `--test-size N` (default 1000) — seeded with the project SEED so the subsample is stable; toggling test ↔ full invalidates the SMILES cache.

**Compute expectations.** A cold full run can take up to a day. Conformer generation dominates the CPU cost; ElektroNN inference runs on GPU (default batch needs ~6 GB).

---

## Project layout

```
eddde/
  __init__.py                # project constants (SEED, N_WORKERS, BCL_BIN), local_settings resolver
  local_settings.example.py  # template for per-machine config (copy to local_settings.py)
  cache.py                   # manifest-based staleness checks
  runner.py                  # main() — stages → embeddings → experiments → SUMMARY.md
  data/
    base.py                  # Stage enum, Dataset base class
    conformers.py            # RDKit ETKDGv3 + MMFF94, lowest-energy single conformer
    elektronn_runner.py      # ElektroNN integration, model cache, supported-element set
    pipeline.py              # build_up_to(dataset, stage), project-wide SMILES filters
    sources/                 # one file per dataset
  methods/
    base.py                  # Method ABC, embedding cache, distance benchmark
    distance.py              # pairwise_matrix(): serial / multiprocessing / batched-override dispatch
    baselines/               # one file per baseline
    muts/                    # one file per MUT (five-strain taxonomy; see PROJECT_PLAN.md §3.2)
  experiments/               # one file per experiment; retrieval_common.py shared
scripts/                     # one-off utilities (e.g. draw_dataset.py)
docs/                        # per-strain implementation references (created on demand)
```

Each cached artifact has a sidecar `*.manifest.json` recording producer version, input hashes, compute time, and accumulated upstream cost. `results/SUMMARY.md` is regenerated on every run with per-experiment metric tables and a cross-experiment average-rank leaderboard.

**Element filter.** The SMILES stage applies a project-wide hard filter dropping molecules whose SMILES contains an element outside ElektroNN's supported basis set (`{H, C, N, O, F, S, Cl}`), so every method — including SMILES-only baselines — sees the same molecule set. See [CLAUDE.md](CLAUDE.md) for details.

---

## Extending

**Add a baseline or MUT** — subclass `Method` (from [eddde/methods/base.py](eddde/methods/base.py)) in `eddde/methods/baselines/` or `eddde/methods/muts/`. Set `id`, `version`, `needs` (a `Stage`), implement `embed_dataset`, and pick exactly one of `distance(e1, e2)` (per-pair) or `distances(embs_q, embs_c)` (batched matrix). The framework derives the other automatically and dispatches across a process pool for large per-pair matrices. Register in `eddde/methods/__init__.py`. MUTs belong to one of the five strains spec'd in [PROJECT_PLAN.md §3.2](PROJECT_PLAN.md); strain-specific implementation references (when needed) live in [docs/](docs/).

**Add a dataset** — subclass `Dataset` in `eddde/data/sources/`, implement `build_smiles` (and `build_native_conformers` if the dataset ships 3D structures). Register in `eddde/data/__init__.py`.

**Add an experiment** — implement the `Experiment` protocol in `eddde/experiments/`. Declare `datasets`, `metric_direction`, and optionally `metric_datasets`. Register in `eddde/experiments/__init__.py`. Retrieval-style experiments share helpers via `retrieval_common.py`.

New dependencies go in `pyproject.toml` — don't silently assume they are present.

**Visualize a dataset** — render a 2D molecule grid:

```bash
python scripts/draw_dataset.py <dataset_id> [--out <path>] [--cols <n>] [--mol-size <px>]
```

---

## Status

Authoritative, always-fresh per-method numbers live in [results/SUMMARY.md](results/SUMMARY.md), regenerated on every `python -m eddde` run. Group-level summary:

- **Framework** — done (runner, caching, manifest chain, SUMMARY writer, optional-binary soft-skip).
- **Baselines** — B1–B11 and B18 implemented. B12 (Mol2vec), B13 (Uni-Mol), B14 (Chemprop), B15–B17 (Coulomb matrix, SOAP, ACSF) pending.
- **MUTs** — Five-strain taxonomy spec'd in [PROJECT_PLAN.md §3.2](PROJECT_PLAN.md). Strain A: `MUT-mean` implemented (refactor to 39-d scalarised pending); other Strain A variants and Strains B/C/D/E all planned.
- **Datasets** — Internal series S1–S8 done. D3 WelQrate (9 PubChem AIDs) and D4 MUV (17 AIDs) done. D5 DUD-E deferred (see PROJECT_PLAN.md §5.5). D6 cliffs, D7–D8 bioisosteres, D9 Riniker-Landrum pending.
- **Experiments** — EXP-1, EXP-2 done. EXP-3a, EXP-3b implemented; first end-to-end runs in progress. EXP-3c deferred (see PROJECT_PLAN.md §5.5). EXP-4 cliffs, EXP-5 bioisosteres, EXP-6 scaffold hopping pending.
