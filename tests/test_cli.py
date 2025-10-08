#!/usr/bin/env python
"""
Tests for EDDDe CLI interfaces (predict.py and convert.py).

Short tests using real interfaces with mock data and temporary files.
"""

import tempfile
import pickle
from pathlib import Path
from unittest.mock import patch, Mock
from omegaconf import OmegaConf

import pytest
import torch

from eddde.predict import run_predict
from eddde.convert import run_convert


def test_predict_interface(temp_dir, basis_params, mock_models):
    """Test predict.py interface with real components and mock data."""
    # Create mock CSV file
    csv_path = temp_dir / "test_molecules.csv"
    with open(csv_path, "w") as f:
        f.write("CID,SMILES\n")
        f.write("1,CCO\n")  # Ethanol
        f.write("2,c1ccccc1\n")  # Benzene

    # Create mock model directory
    model_dir = temp_dir / "models"
    model_dir.mkdir()
    for i, model in enumerate(mock_models):
        torch.save(model.state_dict(), model_dir / f"model_fold_{i+1}.pth")

    # Create config
    config = OmegaConf.create({
        "paths": {
            "basis_params": "./ElektroNN/basisfunction_params.pkl",
            "model_dir": str(model_dir)
        },
        "data": {
            "csv_file": str(csv_path),
            "smiles_col": "SMILES",
            "id_col": "CID",
            "conformer_seed": 42,
            "filter_bad_atoms": True,
            "verbose": False
        },
        "model": {
            "suppress_warnings": True,
            "use_all_folds": True,
            "num_folds": 3,
            "aggregation": "mean"
        },
        "inference": {
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "prefetch_factor": 2,
            "show_progress": False,
            "return_std": False
        },
        "experiment": {
            "output_dir": str(temp_dir / "output")
        },
        "device": "cpu"
    })

    # Run prediction
    run_predict(config)

    # Check output file was created
    output_path = temp_dir / "output" / "predictions_mean.pkl"
    assert output_path.exists()

    # Verify predictions can be loaded
    with open(output_path, "rb") as f:
        predictions = pickle.load(f)

    assert len(predictions) == 2  # Two molecules


def test_convert_interface(temp_dir, basis_params):
    """Test convert.py interface with real components and mock data."""
    # Create mock CSV file
    csv_path = temp_dir / "test_molecules.csv"
    with open(csv_path, "w") as f:
        f.write("CID,SMILES\n")
        f.write("1,CCO\n")  # Ethanol
        f.write("2,c1ccccc1\n")  # Benzene

    # Create config
    config = OmegaConf.create({
        "paths": {
            "basis_params": "./ElektroNN/basisfunction_params.pkl"
        },
        "data": {
            "csv_file": str(csv_path),
            "smiles_col": "SMILES",
            "id_col": "CID",
            "conformer_seed": 42,
            "filter_bad_atoms": False,
            "verbose": False
        },
        "convert": {
            "output_sdf": str(temp_dir / "output.sdf"),
            "optimize_geometry": False
        }
    })

    # Run conversion
    run_convert(config)

    # Check output SDF was created
    output_path = temp_dir / "output.sdf"
    assert output_path.exists()

    # Verify SDF contains molecules
    from rdkit import Chem
    supplier = Chem.SDMolSupplier(str(output_path))
    molecules = [mol for mol in supplier if mol is not None]
    assert len(molecules) == 2  # Two molecules
