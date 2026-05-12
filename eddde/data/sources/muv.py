"""MUV datasets (D4) for EXP-3b: 17 PubChem assays curated to defeat
analog-based similarity.

Paper:  Rohrer & Baumann (2009), JCIM 49(2):169-184. doi:10.1021/ci8002649
Source: rdkit/benchmarking_platform on GitHub. Each target ships as a pair
        of gzipped TSVs (actives and decoys) under compounds/MUV/.

Each MUV assay becomes one Dataset subclass so the runner can track caching
and staleness per target independently — same pattern as welqrate.py.

File format (TSV, comment header):
    # PUBCHEM_COMPOUND_CID    ID    SMILES
    <cid>                     <muv_id>   <smiles>

We use the MUV-assigned `ID` column ("MUV_466_A_12", "MUV_466_D_42") as the
canonical row id: it is unique within the dataset and visibly encodes the
active/decoy label, which makes debugging cleaner than reusing the CID.

build_smiles() writes smiles.csv with columns:
    id        MUV-assigned ID (e.g. "MUV_466_A_12")
    smiles    isomeric SMILES
    activity  1 = active, 0 = decoy
"""

from __future__ import annotations

import gzip
import io
import subprocess
from pathlib import Path

import pandas as pd

from ..base import Dataset, CACHE_ROOT


# All 17 MUV targets shipped with the RDKit benchmarking platform.
MUV_AIDS: list[str] = [
    "466", "548", "600", "644", "652", "689", "692", "712", "713",
    "733", "737", "810", "832", "846", "852", "858", "859",
]

_BASE_URL = (
    "https://raw.githubusercontent.com/rdkit/benchmarking_platform/master/"
    "compounds/MUV"
)


def _curl(url: str, dest: Path) -> None:
    result = subprocess.run(
        ["curl", "-L", "--silent", "--show-error", "-o", str(dest), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed ({url}):\n{result.stderr}")


def _ensure_raw(aid: str, raw_dir: Path) -> None:
    """Download actives + decoys .dat.gz files for one target."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("actives", "decoys"):
        out = raw_dir / f"{kind}.dat.gz"
        if out.exists():
            continue
        url = f"{_BASE_URL}/cmp_list_MUV_{aid}_{kind}.dat.gz"
        print(f"[MUV_{aid}] downloading {kind} ...")
        _curl(url, out)


def _read_dat_gz(path: Path) -> pd.DataFrame:
    """Parse a MUV .dat.gz file (TSV with a '#'-prefixed header)."""
    with gzip.open(path, "rt") as f:
        text = f.read()
    # The header line starts with '#'. pandas can read this if we strip the '#'.
    text = text.lstrip()
    if text.startswith("#"):
        text = text[1:]
    return pd.read_csv(io.StringIO(text), sep="\t")


class _MUVBase(Dataset):
    has_native_conformers = False
    # Same rationale as WelQrate: hit rate is ~30/15000, so uniform test-mode
    # sampling drops most or all actives. Keep every active and fill the rest
    # with random decoys instead. Bump after changing the override logic.
    test_mode_version = "v1-keep-actives"

    _aid: str = ""

    def build_smiles(self, out: Path) -> None:
        aid = self._aid
        raw_dir = CACHE_ROOT / "datasets" / f"MUV_{aid}" / "raw"
        _ensure_raw(aid, raw_dir)

        actives = _read_dat_gz(raw_dir / "actives.dat.gz")
        actives["activity"] = 1
        decoys = _read_dat_gz(raw_dir / "decoys.dat.gz")
        decoys["activity"] = 0

        df = pd.concat([actives, decoys], ignore_index=True)
        df = df.rename(columns={"ID": "id", "SMILES": "smiles"})
        df = df[["id", "smiles", "activity"]]
        df["id"] = df["id"].astype(str)
        df.to_csv(out, index=False)
        print(f"[MUV_{aid}] wrote {len(df)} molecules to {out}")

    def test_mode_subsample(self, df, n, rng):
        if len(df) <= n:
            return df
        actives = df[df["activity"] == 1]
        inactives = df[df["activity"] == 0]

        if len(actives) >= n:
            idx = rng.choice(len(actives), size=n, replace=False)
            idx.sort()
            kept = actives.iloc[idx]
        else:
            n_inactives = min(n - len(actives), len(inactives))
            idx = rng.choice(len(inactives), size=n_inactives, replace=False)
            idx.sort()
            kept = pd.concat([actives, inactives.iloc[idx]])

        return kept.sort_index()


def _make_dataset(aid: str) -> _MUVBase:
    cls = type(
        f"MUV_{aid}",
        (_MUVBase,),
        {"id": f"MUV_{aid}", "version": "v1", "_aid": aid},
    )
    return cls()


ALL_MUV: list[_MUVBase] = [_make_dataset(aid) for aid in MUV_AIDS]
