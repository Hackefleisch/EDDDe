"""EDDDe: Benchmark framework for electron-density-based molecular similarity."""

__version__ = "0.2.0"

# Project-wide RNG seed. Used wherever the pipeline needs deterministic
# randomness (conformer embedding, test-mode downsampling, etc.) so reruns are
# reproducible and content-hashes stay stable across invocations.
SEED = 0xEDDDE
