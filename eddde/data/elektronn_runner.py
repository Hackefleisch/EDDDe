"""ElektroNN integration: predict electron density coefficients + graph matrices.

For each molecule with a 3D conformer (including explicit hydrogens), runs the
pretrained ElektroNN 5-model ensemble to produce a coefficient matrix of
shape (n_atoms, 127) — 127 = 14 + 42 + 25 + 28 + 18 basis-function coefficients
per atom from irreps `14x0e + 14x1o + 5x2e + 4x3o + 2x4e`. Alongside the
coefficients we also cache the RDKit adjacency matrix and the topological
distance matrix (bond-count distances), both sized (n_atoms, n_atoms) and
indexed consistently with the coefficient rows.

Input: pickled dict[mol_id -> rdkit.Chem.Mol with exactly one 3D conformer].
Output: pickled dict with three sub-dicts:
    {
        "coefficients": dict[mol_id -> np.ndarray (n_atoms, 127)],
        "adjacencies":  dict[mol_id -> np.ndarray (n_atoms, n_atoms)],
        "distances":    dict[mol_id -> np.ndarray (n_atoms, n_atoms)],
    }

Molecules containing atoms outside ElektroNN's supported basis-function set
are dropped with a warning; their ids will be missing from all three sub-dicts.
"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path


VERSION = "elektronn-ensemble+graph-v1"

BATCH_SIZE = 128
NUM_WORKERS = 0


def generate(conformers_pkl: Path, out: Path) -> None:
    import numpy as np
    import torch
    from rdkit import Chem
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import unbatch

    from elektronn.dataset import MoleculeDataset
    from elektronn.model import ElektroNN

    with open(conformers_pkl, "rb") as f:
        mols: dict[str, Chem.Mol] = pickle.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        mol_dict: dict[str, str] = {}
        for mol_id, mol in mols.items():
            sdf_path = Path(tmpdir) / f"{mol_id}.sdf"
            Chem.MolToMolFile(mol, str(sdf_path))
            mol_dict[mol_id] = str(sdf_path)

        dataset = MoleculeDataset(mol_dict, verbose=True)

    skipped = set(mols.keys()) - set(dataset.names)
    if skipped:
        preview = ", ".join(sorted(skipped)[:10]) + ("..." if len(skipped) > 10 else "")
        print(f"  [elektronn] skipped {len(skipped)} molecule(s) with unsupported atoms: {preview}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ElektroNN()
    model.eval()
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    coefficients: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            predictions = model(batch)
            predictions = unbatch(predictions, batch.batch)
            for name, pred in zip(batch.name, predictions):
                coefficients[name] = pred.detach().cpu().numpy()

    adjacencies: dict[str, np.ndarray] = {}
    distances: dict[str, np.ndarray] = {}
    for name in dataset.names:
        mol = dataset.get_mol_by_name(name)
        adjacencies[name] = Chem.GetAdjacencyMatrix(mol)
        distances[name] = Chem.GetDistanceMatrix(mol)

    payload = {
        "coefficients": coefficients,
        "adjacencies": adjacencies,
        "distances": distances,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(payload, f)
