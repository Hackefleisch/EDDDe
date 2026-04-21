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

Model weights are loaded once per process via a module-level cache; call
`prewarm()` before any `timed()` pipeline stage to keep the ~12s weight-load
cost out of per-dataset `compute_time` measurements.
"""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any


VERSION = "elektronn-ensemble+graph-v2"

BATCH_SIZE = 128
NUM_WORKERS = 0

_MODEL_CACHE: dict[str, Any] = {}
_SUPPORTED_ELEMENTS: frozenset[int] | None = None


def supported_elements() -> frozenset[int]:
    """Atomic numbers ElektroNN has basis functions for. Used by the SMILES-stage
    filter to drop molecules that can't be embedded — see pipeline._build_stage."""
    global _SUPPORTED_ELEMENTS
    if _SUPPORTED_ELEMENTS is None:
        from elektronn.dataset import MoleculeDataset
        _SUPPORTED_ELEMENTS = frozenset(int(z) for z in MoleculeDataset({}).basisfunction_params.keys())
    return _SUPPORTED_ELEMENTS


def _get_model():
    """Return (model, device). Loads + caches on first call; cheap thereafter."""
    if "model" not in _MODEL_CACHE:
        import torch
        from elektronn.model import ElektroNN
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ElektroNN()
        model.eval()
        model.to(device)
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["device"] = device
    return _MODEL_CACHE["model"], _MODEL_CACHE["device"]


def prewarm() -> None:
    """Load ElektroNN weights once. Safe to call multiple times. Call before
    any timed pipeline stage so weight-load cost isn't billed per dataset."""
    _get_model()


def generate(conformers_pkl: Path, out: Path) -> None:
    import numpy as np
    import torch
    from rdkit import Chem
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import unbatch

    from elektronn.dataset import MoleculeDataset

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

    model, device = _get_model()

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
