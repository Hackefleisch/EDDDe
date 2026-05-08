"""EDDDe: Benchmark framework for electron-density-based molecular similarity."""

import os

__version__ = "0.2.0"

# Project-wide RNG seed. Used wherever the pipeline needs deterministic
# randomness (conformer embedding, test-mode downsampling, etc.) so reruns are
# reproducible and content-hashes stay stable across invocations.
SEED = 0xEDDDE

# Project-wide CPU-bound multiprocessing pool size. Used by every stage that
# parallelises per-molecule work over RDKit (SMILES filtering, conformer
# generation). Overridable via the --conformer-workers CLI flag in
# eddde/__main__.py.
N_WORKERS = os.cpu_count() or 1
