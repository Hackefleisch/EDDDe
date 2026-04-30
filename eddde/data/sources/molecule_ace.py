"""MoleculeACE benchmark datasets (D6) for EXP-4: 30 pre-curated ChEMBL targets.

Paper: van Tilborg, Alenicheva & Grisoni, J. Chem. Inf. Model. 2022, 62, 5938.
       DOI: 10.1021/acs.jcim.2c01073
Data:  github.com/molML/MoleculeACE/tree/main/MoleculeACE/Data/benchmark_data

Each target ships as a per-target CSV with columns:
  smiles, exp_mean [nM], y, cliff_mol, split, y [pEC50/pKi]

Each target is a separate Dataset subclass so the runner tracks caching and
staleness per target independently.

build_smiles() downloads the raw CSV from MoleculeACE GitHub raw into
  cache/datasets/{target_id}/raw/molecule_ace.csv
then writes smiles.csv with columns:
  id           "{target_id}_{row_idx:04d}"
  smiles       canonical SMILES
  pY           -log10(activity in M), i.e. pKi or pEC50
  cliff_mol    0/1 per-molecule cliff flag from the upstream CSV (diagnostic
               only; EXP-4 reconstructs pair labels from similarity criteria)
  split        "train" or "test" from the upstream QSAR split (diagnostic
               only; EXP-4 uses all molecules)
  assay_type   "Ki" or "EC50"

Pair construction and cliff labelling live in EXP-4, not here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from ..base import Dataset, CACHE_ROOT


# ---------------------------------------------------------------------------
# Catalogue: target ID -> assay type. URL is built from these via _RAW_BASE.
# Filenames mirror MoleculeACE/Data/benchmark_data/CHEMBL{n}_{Ki,EC50}.csv.
# ---------------------------------------------------------------------------

_RAW_BASE = (
    "https://raw.githubusercontent.com/molML/MoleculeACE/main/"
    "MoleculeACE/Data/benchmark_data"
)

_CATALOGUE: dict[str, str] = {
    "CHEMBL1862_Ki":  "Ki",
    "CHEMBL1871_Ki":  "Ki",
    "CHEMBL2034_Ki":  "Ki",
    "CHEMBL2047_EC50": "EC50",
    "CHEMBL204_Ki":   "Ki",
    "CHEMBL2147_Ki":  "Ki",
    "CHEMBL214_Ki":   "Ki",
    "CHEMBL218_EC50": "EC50",
    "CHEMBL219_Ki":   "Ki",
    "CHEMBL228_Ki":   "Ki",
    "CHEMBL231_Ki":   "Ki",
    "CHEMBL233_Ki":   "Ki",
    "CHEMBL234_Ki":   "Ki",
    "CHEMBL235_EC50": "EC50",
    "CHEMBL236_Ki":   "Ki",
    "CHEMBL237_EC50": "EC50",
    "CHEMBL237_Ki":   "Ki",
    "CHEMBL238_Ki":   "Ki",
    "CHEMBL239_EC50": "EC50",
    "CHEMBL244_Ki":   "Ki",
    "CHEMBL262_Ki":   "Ki",
    "CHEMBL264_Ki":   "Ki",
    "CHEMBL2835_Ki":  "Ki",
    "CHEMBL287_Ki":   "Ki",
    "CHEMBL2971_Ki":  "Ki",
    "CHEMBL3979_EC50": "EC50",
    "CHEMBL4005_Ki":  "Ki",
    "CHEMBL4203_Ki":  "Ki",
    "CHEMBL4616_EC50": "EC50",
    "CHEMBL4792_Ki":  "Ki",
}


def _curl(url: str, dest: Path) -> None:
    result = subprocess.run(
        ["curl", "-L", "--silent", "--show-error", "-o", str(dest), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed ({url}):\n{result.stderr}")


def _ensure_raw(target_id: str, raw_dir: Path) -> Path:
    """Download the MoleculeACE per-target CSV into raw_dir if not present."""
    raw_csv = raw_dir / "molecule_ace.csv"
    if raw_csv.exists():
        return raw_csv
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = f"{_RAW_BASE}/{target_id}.csv"
    print(f"[{target_id}] downloading MoleculeACE CSV ...")
    _curl(url, raw_csv)
    return raw_csv


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _MoleculeACEBase(Dataset):
    has_native_conformers = False

    _target_id: str = ""
    _assay: str = ""

    def build_smiles(self, out: Path) -> None:
        target_id = self._target_id
        raw_dir = CACHE_ROOT / "datasets" / target_id / "raw"
        raw_csv = _ensure_raw(target_id, raw_dir)

        raw = pd.read_csv(raw_csv)
        # Required columns from MoleculeACE benchmark CSVs.
        expected = {"smiles", "cliff_mol", "split", "y [pEC50/pKi]"}
        missing = expected - set(raw.columns)
        if missing:
            raise RuntimeError(
                f"[{target_id}] MoleculeACE CSV missing columns: {sorted(missing)}"
            )

        df = pd.DataFrame({
            "id": [f"{target_id}_{i:04d}" for i in range(len(raw))],
            "smiles": raw["smiles"].astype(str),
            "pY": raw["y [pEC50/pKi]"].astype(float),
            "cliff_mol": raw["cliff_mol"].astype(int),
            "split": raw["split"].astype(str),
            "assay_type": self._assay,
        })
        df.to_csv(out, index=False)
        print(f"[{target_id}] wrote {len(df)} molecules to {out}")


# ---------------------------------------------------------------------------
# One concrete class per target, generated from the catalogue
# ---------------------------------------------------------------------------

def _make_dataset(target_id: str, assay: str) -> _MoleculeACEBase:
    cls = type(
        target_id,
        (_MoleculeACEBase,),
        {"id": target_id, "version": "v1", "_target_id": target_id, "_assay": assay},
    )
    return cls()


ALL_MOLECULE_ACE: list[_MoleculeACEBase] = [
    _make_dataset(tid, assay) for tid, assay in _CATALOGUE.items()
]

MOLECULE_ACE_DATASET_IDS: list[str] = list(_CATALOGUE.keys())
