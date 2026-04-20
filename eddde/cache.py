"""Cache manifests and staleness checks.

Each cached artifact has a sidecar manifest (JSON) recording:
  - version: code version string of the producer
  - inputs: dict of input_name -> hash or version string it depended on
  - output_hash: SHA-256 of the produced artifact
  - compute_time: seconds spent producing this artifact alone
  - upstream_compute_time: sum of compute_time + upstream_compute_time of
      direct inputs, so the full chain cost can be read in O(1)
  - timestamp: unix seconds

An artifact is stale if any of:
  - the artifact file is missing
  - the manifest is missing
  - the stored version != expected version
  - the stored inputs != expected inputs
Input changes cascade: a stage's output_hash ends up in the next stage's
inputs, so bumping a version anywhere invalidates every downstream artifact.
"""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


CHUNK = 1 << 20


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Manifest:
    version: str
    inputs: dict
    output_hash: str
    compute_time: float
    upstream_compute_time: float = 0.0
    timestamp: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> "Manifest | None":
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text()))

    def chain_time(self) -> float:
        return self.compute_time + self.upstream_compute_time


def manifest_path(artifact: Path) -> Path:
    return artifact.parent / (artifact.name + ".manifest.json")


def is_stale(artifact: Path, expected_version: str, expected_inputs: dict) -> bool:
    if not artifact.exists():
        return True
    m = Manifest.load(manifest_path(artifact))
    if m is None:
        return True
    if m.version != expected_version:
        return True
    if m.inputs != expected_inputs:
        return True
    return False


def write_manifest(
    artifact: Path,
    version: str,
    inputs: dict,
    compute_time: float,
    upstream_compute_time: float = 0.0,
) -> Manifest:
    m = Manifest(
        version=version,
        inputs=inputs,
        output_hash=hash_file(artifact),
        compute_time=compute_time,
        upstream_compute_time=upstream_compute_time,
        timestamp=time.time(),
    )
    manifest_path(artifact).write_text(m.to_json())
    return m


@contextmanager
def timed():
    """Usage: with timed() as t: ...; elapsed = t['seconds']"""
    t = {"seconds": 0.0}
    start = time.perf_counter()
    try:
        yield t
    finally:
        t["seconds"] = time.perf_counter() - start
