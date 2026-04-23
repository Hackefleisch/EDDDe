"""WelQrate datasets (D3) for EXP-3a: 9 curated virtual-screening targets.

Paper: Liu et al., NeurIPS 2024. arXiv:2411.09820
Data:  vanderbilt.box.com/v/WelQrate-Datasets

Each dataset is a separate Dataset subclass so the runner can track caching
and staleness per target independently.

build_smiles() downloads the raw CSVs and scaffold splits from Box into
  cache/datasets/{AID}/raw/
then writes smiles.csv with columns:
  id            PubChem CID as string
  smiles        isomeric SMILES
  activity      1 = active, 0 = inactive
  scaffold_seed1 .. scaffold_seed5   "train" | "valid" | "test"
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ..base import Dataset, CACHE_ROOT

# ---------------------------------------------------------------------------
# URL catalogue (sourced from the welqrate 0.1.4 package metadata)
# ---------------------------------------------------------------------------

_CATALOGUE: dict[str, dict[str, str]] = {
    "AID1798": {
        "raw_url": "https://vanderbilt.box.com/shared/static/cd2dpdinu8grvi8dye3gi9bzsdt659d1.zip",
        "split_url": "https://vanderbilt.box.com/shared/static/68m9qigxd7kt0xtta3chrx270grd1l4p.zip",
    },
    "AID1843": {
        "raw_url": "https://vanderbilt.box.com/shared/static/4f9wl7t9pmkm5p6695owj6bsxin0hdo1.zip",
        "split_url": "https://vanderbilt.box.com/shared/static/o3hmvwo0surg1vhettflhtxxgn2o6clw.zip",
    },
    "AID2258": {
        "raw_url": "https://vanderbilt.box.com/shared/static/b1cg619c4f35p9mkm42mgu8a9m86t4at.zip",
        "split_url": "https://vanderbilt.box.com/shared/static/2hhz7o3vxwjq6p93yzxfrporeze5zzgf.zip",
    },
    "AID2689": {
        "raw_url": "https://vanderbilt.box.com/shared/static/5no4ut1vusxmxsbwxsmb68o6ix94mxgy.zip",
        "split_url": "https://vanderbilt.box.com/shared/static/hml9qznmwhhb6rcddw18vp7yoh331wlb.zip",
    },
    "AID435008": {
        "raw_url": "https://vanderbilt.box.com/shared/static/mu6hfitlenanp11wgm3z7u5drae2ofsd.zip",
        "split_url": "https://vanderbilt.box.com/shared/static/ybyorbb541gnefjvagbqg1p37jfae98z.zip",
    },
    "AID435034": {
        "raw_url": "https://vanderbilt.box.com/shared/static/ctk2vs70bsjmpznalqoj4uqbbdkdcbgj.zip",
        "split_url": "https://vanderbilt.box.com/shared/static/onistu6vrizl4o2nu0xlfhowcf1q498a.zip",
    },
    "AID463087": {
        "raw_url": "https://vanderbilt.box.com/shared/static/y52n3pzf27ghqtxt0xz899gwuj3yw3m2.zip",
        "split_url": "https://vanderbilt.box.com/shared/static/y9jkhhu4cljq4awlfkvlo4im6iz3kj5t.zip",
    },
    "AID485290": {
        "raw_url": "https://vanderbilt.box.com/shared/static/8w0njtoqmgs8g1c12qj0p52w0d3wwbop.zip",
        "split_url": "https://vanderbilt.box.com/shared/static/2w5byzpf5pk4jqb9rknzdihzztb3u9e6.zip",
    },
    "AID488997": {
        "raw_url": "https://vanderbilt.box.com/shared/static/rguy1gm36x7clq6822riconznoflc4xe.zip",
        "split_url": "https://vanderbilt.box.com/shared/static/dceunwrlotxkz58usfgnxr7meziy3cdr.zip",
    },
}

N_SCAFFOLD_SEEDS = 5


# ---------------------------------------------------------------------------
# Shared download helpers
# ---------------------------------------------------------------------------

def _curl(url: str, dest: Path) -> None:
    result = subprocess.run(
        ["curl", "-L", "--silent", "--show-error", "-o", str(dest), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed ({url}):\n{result.stderr}")


def _ensure_raw(aid: str, raw_url: str, raw_dir: Path) -> None:
    """Download and extract actives/inactives CSVs into raw_dir if not present."""
    if (raw_dir / "actives.csv").exists() and (raw_dir / "inactives.csv").exists():
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "raw.zip"
        print(f"[{aid}] downloading raw data ...")
        _curl(raw_url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                if not member.endswith(".csv"):
                    continue
                name = Path(member).name
                if "inactives" in name:
                    dest = raw_dir / "inactives.csv"
                elif "actives" in name:
                    dest = raw_dir / "actives.csv"
                else:
                    continue
                dest.write_bytes(zf.read(member))
        print(f"[{aid}] raw CSVs saved to {raw_dir}")


def _ensure_splits(aid: str, split_url: str, split_dir: Path, all_cids: list[int]) -> None:
    """Download scaffold splits and persist as JSON {seed: {train/valid/test: [CID, ...]}}."""
    existing = list(split_dir.glob("scaffold_seed*.json"))
    if len(existing) == N_SCAFFOLD_SEEDS:
        return
    split_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "split.zip"
        print(f"[{aid}] downloading split data ...")
        _curl(split_url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            scaffold_pts = sorted(
                n for n in zf.namelist()
                if "scaffold" in n and "_2d_" in n and n.endswith(".pt")
            )
            for member in scaffold_pts:
                # e.g. scaffold/AID1798_2d_scaffold_seed1.pt
                seed = Path(member).stem.split("scaffold_seed")[-1]
                pt_path = Path(tmp) / Path(member).name
                pt_path.write_bytes(zf.read(member))
                raw = torch.load(pt_path, weights_only=False)
                # Some files carry an extra 'timestamp' string key — keep only split subsets.
                split_dict = {k: v for k, v in raw.items() if isinstance(v, list) and k in {"train", "valid", "test"}}
                cid_split = {
                    subset: [all_cids[i] for i in indices]
                    for subset, indices in split_dict.items()
                }
                out = split_dir / f"scaffold_seed{seed}.json"
                out.write_text(json.dumps(cid_split))
        print(f"[{aid}] scaffold splits saved to {split_dir}")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _WelQrateBase(Dataset):
    has_native_conformers = False

    _aid: str = ""
    _raw_url: str = ""
    _split_url: str = ""

    def build_smiles(self, out: Path) -> None:
        aid = self._aid
        raw_dir = CACHE_ROOT / "datasets" / aid / "raw"
        split_dir = CACHE_ROOT / "datasets" / aid / "splits"

        _ensure_raw(aid, self._raw_url, raw_dir)

        actives = pd.read_csv(raw_dir / "actives.csv", usecols=["CID", "SMILES"])
        actives["activity"] = 1
        inactives = pd.read_csv(raw_dir / "inactives.csv", usecols=["CID", "SMILES"])
        inactives["activity"] = 0

        # Ordering must be stable: actives first, then inactives — matches split indices.
        df = pd.concat([actives, inactives], ignore_index=True)
        all_cids: list[int] = df["CID"].tolist()

        _ensure_splits(aid, self._split_url, split_dir, all_cids)

        # Attach scaffold split assignments
        cid_to_idx = {cid: i for i, cid in enumerate(all_cids)}
        for seed in range(1, N_SCAFFOLD_SEEDS + 1):
            split_json = split_dir / f"scaffold_seed{seed}.json"
            split_data: dict[str, list[int]] = json.loads(split_json.read_text())
            col = f"scaffold_seed{seed}"
            assignment: dict[int, str] = {}
            for subset, cids in split_data.items():
                for cid in cids:
                    assignment[cid] = subset
            df[col] = df["CID"].map(assignment)

        df = df.rename(columns={"CID": "id", "SMILES": "smiles"})
        df["id"] = df["id"].astype(str)
        df.to_csv(out, index=False)
        print(f"[{aid}] wrote {len(df)} molecules to {out}")


# ---------------------------------------------------------------------------
# One concrete class per target, generated from the catalogue
# ---------------------------------------------------------------------------

def _make_dataset(aid: str, raw_url: str, split_url: str) -> _WelQrateBase:
    cls = type(
        aid,
        (_WelQrateBase,),
        {"id": aid, "version": "v1", "_aid": aid, "_raw_url": raw_url, "_split_url": split_url},
    )
    return cls()


AID1798  = _make_dataset("AID1798",  **_CATALOGUE["AID1798"])
AID1843  = _make_dataset("AID1843",  **_CATALOGUE["AID1843"])
AID2258  = _make_dataset("AID2258",  **_CATALOGUE["AID2258"])
AID2689  = _make_dataset("AID2689",  **_CATALOGUE["AID2689"])
AID435008 = _make_dataset("AID435008", **_CATALOGUE["AID435008"])
AID435034 = _make_dataset("AID435034", **_CATALOGUE["AID435034"])
AID463087 = _make_dataset("AID463087", **_CATALOGUE["AID463087"])
AID485290 = _make_dataset("AID485290", **_CATALOGUE["AID485290"])
AID488997 = _make_dataset("AID488997", **_CATALOGUE["AID488997"])

ALL_WELQRATE = [
    AID1798, AID1843, AID2258, AID2689,
    AID435008, AID435034, AID463087, AID485290, AID488997,
]
