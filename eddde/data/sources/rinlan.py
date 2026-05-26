"""Riniker-Landrum ChEMBL_II benchmark (D9) for EXP-6: 37 ChEMBL targets curated
for retrospective virtual-screening and scaffold-hopping evaluation.

Paper:  Riniker & Landrum, J. Cheminform. 2013, 5:26. doi:10.1186/1758-2946-5-26
Source: rdkit/benchmarking_platform on GitHub.

Data layout in the upstream repo:
    compounds/ChEMBL_II/Target_no_<N>.pkl
        Pickled defaultdict(scaffold_id -> [[chembl_id, smiles], ...]). Actives
        of one target, pre-clustered by scaffold. We keep the scaffold_id as a
        column in smiles.csv so EXP-6 can compute scaffold-aware metrics
        without re-deriving Bemis-Murcko.
    compounds/ChEMBL/cmp_list_ChEMBL_zinc_decoys.dat.gz
        Shared ~10,000-compound ZINC decoy pool, used as the screening-library
        background for every ChEMBL target (the original 2013 protocol uses it
        the same way; the `firstchembl` flag in their scoring scripts reads it
        once and reuses across targets). We mirror that: download once to a
        shared raw dir, then deterministically sample DECOYS_PER_TARGET=1500
        decoys per target seeded by sha256(SEED|dataset_id).

Note on PROJECT_PLAN.md §5.8: the spec says "88 targets". The repo organizes
into three subsets (per its README):
  - subset I:           88 targets (MUV + DUD + ChEMBL combined)
  - subset I filtered:  69 targets
  - subset II:          37 targets from ChEMBL only — designed for VS use case
Subset II is the right pool for EXP-6 because only its compounds ship with
upstream scaffold clusters; MUV/DUD do not. The 37-target count should be
documented in PROJECT_PLAN.md (see scratch/exp6_implementation_plan.md §11.1).

Each target becomes one Dataset subclass so the runner can track caching and
staleness per target independently — same pattern as muv.py and welqrate.py.

build_smiles() output schema:
    id            chembl_id for actives, "decoy_<external_id>" for decoys
    smiles        canonical SMILES
    activity      1 (active) | 0 (decoy)
    scaffold_id   non-negative int for actives (preserved from upstream),
                  -1 for decoys
"""

from __future__ import annotations

import gzip
import hashlib
import io
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ... import SEED
from ..base import Dataset, CACHE_ROOT


# ---------------------------------------------------------------------------
# Catalogue: ChEMBL target IDs from compounds/ChEMBL_II/ (37 entries).
# Verified via GitHub tree API on 2026-05-26.
# ---------------------------------------------------------------------------

RINLAN_TARGET_IDS: list[str] = [
    "15", "25", "43", "51", "61", "65", "72", "87", "90", "93",
    "100", "107", "108", "114", "121", "126", "130", "165", "259",
    "10188", "10193", "10260", "10280", "10434", "10980",
    "11140", "11365", "11489", "11534", "11575", "11631",
    "12209", "12252", "12952", "13001", "17045", "19905",
]


_RAW_BASE = (
    "https://raw.githubusercontent.com/rdkit/benchmarking_platform/master"
)

# Per-target decoy budget. Original Riniker-Landrum used 10 % of the full ZINC
# decoy pool (~1000); 1500 keeps things in that order of magnitude while
# leaving headroom for the project-wide element filter to drop a few percent.
DECOYS_PER_TARGET = 1500

# Shared raw-data location: the global ZINC decoy file is downloaded once and
# reused for every Target_no_<N>. Per-target raw dirs link out to it rather
# than re-downloading.
_SHARED_RAW_DIR = CACHE_ROOT / "datasets" / "_rinlan_shared" / "raw"
_ZINC_DECOYS_PATH = _SHARED_RAW_DIR / "zinc_decoys.dat.gz"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _curl(url: str, dest: Path) -> None:
    result = subprocess.run(
        ["curl", "-L", "--silent", "--show-error", "-o", str(dest), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed ({url}):\n{result.stderr}")


def _ensure_zinc_decoys() -> Path:
    """Download the shared ZINC decoy file once; reuse for every target."""
    if _ZINC_DECOYS_PATH.exists():
        return _ZINC_DECOYS_PATH
    _SHARED_RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_RAW_BASE}/compounds/ChEMBL/cmp_list_ChEMBL_zinc_decoys.dat.gz"
    print(f"[rinlan/shared] downloading ZINC decoy pool ...")
    _curl(url, _ZINC_DECOYS_PATH)
    return _ZINC_DECOYS_PATH


def _ensure_actives_pkl(target_id: str, raw_dir: Path) -> Path:
    raw_pkl = raw_dir / "actives.pkl"
    if raw_pkl.exists():
        return raw_pkl
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = f"{_RAW_BASE}/compounds/ChEMBL_II/Target_no_{target_id}.pkl"
    print(f"[RinLan_{target_id}] downloading actives ...")
    _curl(url, raw_pkl)
    return raw_pkl


def _read_zinc_decoys(path: Path) -> list[tuple[str, str]]:
    """Parse the shared ZINC decoy file: rows of (external_id, internal_id, smiles).
    Returns list of (internal_id, smiles) — internal_id is unique within the pool.
    """
    with gzip.open(path, "rt") as f:
        text = f.read()
    rows: list[tuple[str, str]] = []
    for raw in io.StringIO(text):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        # external_id, internal_id, smiles
        rows.append((parts[1], parts[2]))
    return rows


def _seeded_rng(target_id: str) -> np.random.Generator:
    """Deterministic RNG per target, derived from project SEED so reruns
    sample the same decoys regardless of machine or order."""
    key = f"{SEED}|RinLan_{target_id}".encode()
    seed_bytes = hashlib.sha256(key).digest()[:8]
    return np.random.default_rng(int.from_bytes(seed_bytes, "big"))


# ---------------------------------------------------------------------------
# Base class — one subclass per target via factory below.
# ---------------------------------------------------------------------------

class _RinLanBase(Dataset):
    has_native_conformers = False

    _target_id: str = ""

    def build_smiles(self, out: Path) -> None:
        raw_dir = CACHE_ROOT / "datasets" / self.id / "raw"
        actives_pkl = _ensure_actives_pkl(self._target_id, raw_dir)
        zinc_pkl = _ensure_zinc_decoys()

        with open(actives_pkl, "rb") as f:
            scaf_to_actives = pickle.load(f, encoding="latin-1")

        rows: list[dict] = []
        for scaf_id, active_list in scaf_to_actives.items():
            sid = int(scaf_id)  # upstream uses Decimal
            for chembl_id, smiles in active_list:
                rows.append({
                    "id": str(chembl_id),
                    "smiles": str(smiles),
                    "activity": 1,
                    "scaffold_id": sid,
                })

        decoys = _read_zinc_decoys(zinc_pkl)
        rng = _seeded_rng(self._target_id)
        budget = min(DECOYS_PER_TARGET, len(decoys))
        chosen = rng.choice(len(decoys), size=budget, replace=False)
        for i in chosen:
            internal_id, smiles = decoys[int(i)]
            rows.append({
                "id": f"decoy_{internal_id}",
                "smiles": str(smiles),
                "activity": 0,
                "scaffold_id": -1,
            })

        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        n_act = int((df["activity"] == 1).sum())
        n_dec = int((df["activity"] == 0).sum())
        n_scaf = df.loc[df["activity"] == 1, "scaffold_id"].nunique()
        print(
            f"[{self.id}] wrote {len(df)} molecules to {out} "
            f"({n_act} actives across {n_scaf} scaffolds, {n_dec} decoys)"
        )

    def test_mode_subsample(self, df: pd.DataFrame, n: int, rng):  # noqa: ANN001
        """Keep every active, fill remainder with random decoys.

        Uniform random would drop most actives at typical hit rates (~10 %),
        leaving too few to compute EF/ScafEF. Mirrors the WelQrate policy.
        """
        actives = df[df["activity"] == 1]
        decoys = df[df["activity"] == 0]
        budget = max(0, n - len(actives))
        if budget < len(decoys):
            decoys = decoys.sample(n=budget, random_state=rng)
        return pd.concat([actives, decoys]).reset_index(drop=True)

    test_mode_version = "v1-keep-actives"


def _make_dataset(target_id: str) -> _RinLanBase:
    ds_id = f"RinLan_{target_id}"
    cls = type(
        ds_id,
        (_RinLanBase,),
        {"id": ds_id, "version": "v1", "_target_id": target_id},
    )
    return cls()


ALL_RINLAN: list[_RinLanBase] = [_make_dataset(t) for t in RINLAN_TARGET_IDS]
RINLAN_DATASET_IDS: list[str] = [ds.id for ds in ALL_RINLAN]
