"""Draw a grid of 2D molecule images for a registered dataset.

Usage:
    python scripts/draw_dataset.py <dataset_id> [--out <path>] [--cols <n>]

The script builds the SMILES stage (using the project cache/pipeline) if it
hasn't been built yet, then renders each molecule with RDKit and saves a grid
figure to a PNG.

Examples:
    python scripts/draw_dataset.py S1
    python scripts/draw_dataset.py S6 --out figures/s6_molecules.png --cols 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run from any directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _build_smiles_stage(dataset_id: str) -> Path:
    from eddde.data import DATASETS
    from eddde.data.base import Stage
    from eddde.data.pipeline import build_up_to

    ds = DATASETS.get(dataset_id)
    if ds is None:
        known = ", ".join(sorted(DATASETS))
        raise SystemExit(f"Unknown dataset '{dataset_id}'. Known: {known}")

    build_up_to(ds, Stage.SMILES)

    from eddde.data.base import stage_path
    return stage_path(dataset_id, Stage.SMILES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw 2D molecule grid for a dataset.")
    parser.add_argument("dataset_id", help="Dataset ID (e.g. S1, S6)")
    parser.add_argument("--out", default=None, help="Output PNG path (default: figures/<dataset_id>_molecules.png)")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the grid (default: 4)")
    parser.add_argument("--mol-size", type=int, default=200, help="Pixel size of each molecule tile (default: 200)")
    args = parser.parse_args()

    smiles_csv = _build_smiles_stage(args.dataset_id)

    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    df = pd.read_csv(smiles_csv)
    if df.empty:
        raise SystemExit("Dataset is empty after element filtering.")

    mols, labels = [], []
    for row in df.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None:
            print(f"  warning: could not parse SMILES for {row.id}, skipping")
            continue
        mols.append(mol)
        labels.append(str(row.id))

    if not mols:
        raise SystemExit("No valid molecules to draw.")

    cols = min(args.cols, len(mols))
    rows = (len(mols) + cols - 1) // cols
    tile = args.mol_size
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=cols,
        subImgSize=(tile, tile),
        legends=labels,
    )

    out_path = Path(args.out) if args.out else ROOT / "figures" / f"{args.dataset_id}_molecules.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved {len(mols)}-molecule grid ({rows}×{cols}) to {out_path}")


if __name__ == "__main__":
    main()
