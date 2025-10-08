"""
Tests for data loading and molecular dataset functionality.

Tests cover:
- SMILES to molecular graph conversion
- SDF file loading
- CSV file loading
- Dataset filtering and statistics
- Conformer generation
"""

from pathlib import Path

import pytest
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist

from eddde.data import (
    MoleculeDataset,
    SDFMoleculeDataset,
    load_smiles_csv,
    load_molecules_from_sdf,
    convert_smiles_to_sdf,
    create_dataset_from_csv,
    create_dataset_from_sdf,
)


# ============================================================================
# MoleculeDataset Tests
# ============================================================================

def test_dataset_initialization(basis_params):
    """Test dataset initialization with SMILES."""
    smiles_dict = {
        "mol1": "CCO",  # Ethanol
        "mol2": "c1ccccc1",  # Benzene
        "mol3": "CC(=O)O",  # Acetic acid
    }

    dataset = MoleculeDataset(
        smiles_dict=smiles_dict,
        basisfunction_params=basis_params,
        conformer_seed=42,
    )

    assert len(dataset) == 3
    assert dataset.names == ["mol1", "mol2", "mol3"]


def test_invalid_smiles_filtering(basis_params):
    """Test filtering of invalid SMILES."""
    smiles_dict = {
        "mol1": "CCO",
        "invalid": "INVALID_SMILES",
        "mol2": "c1ccccc1",
    }

    dataset = MoleculeDataset(
        smiles_dict=smiles_dict,
        basisfunction_params=basis_params,
        filter_bad_atoms=True,
        verbose=False,
    )

    # Should filter out the invalid SMILES
    assert len(dataset) == 2
    assert dataset.bad_atom_count == 1


def test_bad_atom_filtering(basis_params):
    """Test filtering of molecules with unsupported atoms."""
    smiles_dict = {
        "mol1": "CCO",  # Valid
        "mol2": "CCl",  # Contains Cl (may not be in basis_params)
        "mol3": "c1ccccc1",  # Valid
    }

    dataset = MoleculeDataset(
        smiles_dict=smiles_dict,
        basisfunction_params=basis_params,
        filter_bad_atoms=True,
        verbose=False,
    )

    # Should have at least 2 valid molecules
    assert len(dataset) >= 2
    assert "mol1" in dataset.names
    assert "mol3" in dataset.names


def test_getitem_returns_data_object(basis_params):
    """Test that __getitem__ returns valid PyTorch Geometric Data."""
    smiles_dict = {"ethanol": "CCO"}

    dataset = MoleculeDataset(
        smiles_dict=smiles_dict,
        basisfunction_params=basis_params,
    )

    data = dataset[0]

    # Check that data object has expected attributes
    assert hasattr(data, "pos")
    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert isinstance(data.pos, torch.Tensor)


def test_dataset_statistics(basis_params):
    """Test dataset statistics calculation."""
    smiles_dict = {
        "mol1": "CCO",
        "mol2": "c1ccccc1",
    }

    dataset = MoleculeDataset(
        smiles_dict=smiles_dict,
        basisfunction_params=basis_params,
    )

    stats = dataset.get_statistics()

    assert stats["num_molecules"] == 2
    assert stats["total_atoms"] > 0
    assert stats["avg_atoms_per_mol"] > 0


def test_conformer_caching(basis_params):
    """Test conformer caching functionality."""
    smiles_dict = {"ethanol": "CCO"}

    dataset = MoleculeDataset(
        smiles_dict=smiles_dict,
        basisfunction_params=basis_params,
        cache_conformers=True,
    )

    # First access
    data1 = dataset[0]

    # Check cache
    assert 0 in dataset.conformer_cache

    # Second access should use cache
    data2 = dataset[0]

    # Both should produce valid data
    assert hasattr(data1, "pos")
    assert hasattr(data2, "pos")


# ============================================================================
# CSV Loading Tests
# ============================================================================

def test_load_smiles_csv(temp_dir):
    """Test loading SMILES from CSV file."""
    csv_path = temp_dir / "molecules.csv"

    # Create test CSV
    with open(csv_path, "w") as f:
        f.write("CID,SMILES\n")
        f.write("1,CCO\n")
        f.write("2,c1ccccc1\n")
        f.write("3,CC(=O)O\n")

    smiles_dict = load_smiles_csv(csv_path, smiles_col="SMILES", id_col="CID")

    assert len(smiles_dict) == 3
    assert smiles_dict["1"] == "CCO"
    assert smiles_dict["2"] == "c1ccccc1"
    assert smiles_dict["3"] == "CC(=O)O"


def test_csv_missing_column(temp_dir):
    """Test error handling for missing columns."""
    csv_path = temp_dir / "molecules.csv"

    with open(csv_path, "w") as f:
        f.write("ID,STRUCTURE\n")
        f.write("1,CCO\n")

    with pytest.raises(KeyError):
        load_smiles_csv(csv_path, smiles_col="SMILES", id_col="CID")


# ============================================================================
# SDF Loading Tests
# ============================================================================

def test_load_molecules_from_sdf(temp_dir, basis_params):
    """Test loading molecules from SDF file."""
    sdf_path = temp_dir / "molecules.sdf"

    # Create test SDF with 3D conformers
    writer = Chem.SDWriter(str(sdf_path))

    smiles_list = ["CCO", "c1ccccc1"]
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        mol.SetProp("_Name", f"mol_{i}")
        writer.write(mol)

    writer.close()

    # Load molecules
    molecules = load_molecules_from_sdf(sdf_path)

    assert len(molecules) == 2
    assert "mol_0" in molecules
    assert "mol_1" in molecules


def test_sdf_dataset(temp_dir, basis_params):
    """Test SDFMoleculeDataset."""
    sdf_path = temp_dir / "molecules.sdf"

    # Create test SDF
    writer = Chem.SDWriter(str(sdf_path))
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mol.SetProp("_Name", "ethanol")
    writer.write(mol)
    writer.close()

    # Create dataset
    dataset = SDFMoleculeDataset(
        sdf_path=sdf_path,
        basisfunction_params=basis_params,
    )

    assert len(dataset) == 1
    assert dataset.names[0] == "ethanol"

    # Get data
    data = dataset[0]
    assert hasattr(data, "pos")


# ============================================================================
# Conversion Tests
# ============================================================================

def test_convert_smiles_to_sdf(temp_dir):
    """Test conversion of SMILES to SDF with 3D conformers."""
    smiles_dict = {
        "ethanol": "CCO",
        "benzene": "c1ccccc1",
    }

    output_path = temp_dir / "output.sdf"

    success, failed = convert_smiles_to_sdf(
        smiles_dict=smiles_dict,
        output_sdf_path=output_path,
        conformer_seed=42,
        verbose=False,
    )

    assert success == 2
    assert failed == 0
    assert output_path.exists()

    # Verify SDF can be loaded
    molecules = load_molecules_from_sdf(output_path)
    assert len(molecules) == 2


def test_conversion_with_filtering(temp_dir, basis_params):
    """Test conversion with atom filtering."""
    smiles_dict = {
        "ethanol": "CCO",  # Valid
        "chloroform": "C(Cl)(Cl)Cl",  # Contains Cl
    }

    output_path = temp_dir / "output.sdf"

    success, failed = convert_smiles_to_sdf(
        smiles_dict=smiles_dict,
        output_sdf_path=output_path,
        filter_bad_atoms=True,
        basisfunction_params=basis_params,
        verbose=False,
    )

    # Ethanol should succeed, chloroform may fail if Cl not in params
    assert success >= 1
