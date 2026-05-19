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


# ----- per-machine optional settings ---------------------------------------
# Settings live in `eddde/local_settings.py` (gitignored, copy from
# local_settings.example.py). When that file is absent or doesn't set a
# given attribute, the value here is None and the dependent method
# silently does not register.
try:
    from . import local_settings as _local_settings  # noqa: F401
except ImportError:
    _local_settings = None


def _setting(attr_name: str, default=None):
    if _local_settings is not None and hasattr(_local_settings, attr_name):
        return getattr(_local_settings, attr_name) or default
    return default


# Path to the BCL executable (for B18 BCL::Mol2D). The BCL binary is closed-
# source and its license forbids redistribution; users install it themselves
# (see CLAUDE.md §Environment) and set BCL_BIN in eddde/local_settings.py.
# When None, B18 silently does not register -- every other method is
# unaffected.
BCL_BIN = _setting("BCL_BIN")
