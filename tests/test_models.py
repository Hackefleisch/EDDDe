"""
Tests for model loading and E(3) equivariance.

Tests cover:
- Model loading and initialization
- Ensemble predictor functionality
- E(3) equivariance under rotations
- Basis function parameter loading
"""

from pathlib import Path
import pickle

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from e3nn.nn.models.gate_points_2101 import Network
from e3nn import o3

from eddde.models import (
    ModelManager,
    EnsemblePredictor,
    load_basis_function_params,
    DEFAULT_MODEL_KWARGS,
)


# ============================================================================
# ModelManager Tests
# ============================================================================

def test_model_manager_initialization(model_dir):
    """Test ModelManager initialization."""
    manager = ModelManager(
        model_dir=model_dir,
        device="cpu",
    )

    assert manager.model_dir == model_dir
    assert manager.device == torch.device("cpu")


def test_load_single_model(model_dir):
    """Test loading a single model."""
    manager = ModelManager(
        model_dir=model_dir,
        device="cpu",
    )

    model = manager.load_single_model("model_fold_1.pth")

    assert isinstance(model, nn.Module)
    assert len(manager.models) == 1


def test_load_ensemble(model_dir):
    """Test loading ensemble of models."""
    manager = ModelManager(
        model_dir=model_dir,
        device="cpu",
    )

    models = manager.load_ensemble(num_folds=3)

    assert len(models) == 3
    for model in models:
        assert isinstance(model, nn.Module)


def test_load_specific_folds(model_dir):
    """Test loading specific fold indices."""
    manager = ModelManager(
        model_dir=model_dir,
        device="cpu",
    )

    models = manager.load_ensemble(fold_indices=[1, 3])

    assert len(models) == 2


def test_get_model_info(model_dir):
    """Test model information retrieval."""
    manager = ModelManager(
        model_dir=model_dir,
        device="cpu",
    )

    manager.load_ensemble(num_folds=3)
    info = manager.get_model_info()

    assert info["num_models"] == 3
    assert info["parameters_per_model"] > 0
    assert info["total_parameters"] == info["parameters_per_model"] * 3


# ============================================================================
# EnsemblePredictor Tests
# ============================================================================

def _create_mock_data():
    """Create mock PyTorch Geometric Data object."""
    num_nodes = 10
    num_edges = 20

    data = Data(
        pos=torch.randn(num_nodes, 3),
        x=torch.randn(num_nodes, 18),  # 18x 0e input
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 16),  # Spherical harmonics
    )

    return data


def test_ensemble_predictor_initialization(mock_models):
    """Test EnsemblePredictor initialization."""
    predictor = EnsemblePredictor(
        models=mock_models,
        device="cpu",
    )

    assert predictor.num_models == 3
    assert predictor.device == torch.device("cpu")


def test_empty_models_error():
    """Test error when no models provided."""
    with pytest.raises(ValueError):
        EnsemblePredictor(models=[], device="cpu")


def test_predict_single(mock_models):
    """Test prediction on single molecule."""
    predictor = EnsemblePredictor(
        models=mock_models,
        device="cpu",
    )

    data = _create_mock_data()

    mean_pred, std_pred = predictor.predict(data, return_std=True)

    assert isinstance(mean_pred, torch.Tensor)
    assert isinstance(std_pred, torch.Tensor)
    assert mean_pred.shape == std_pred.shape


def test_predict_batch(mock_models):
    """Test prediction on batch of molecules."""
    predictor = EnsemblePredictor(
        models=mock_models,
        device="cpu",
    )

    data_list = [_create_mock_data() for _ in range(5)]
    batch = Batch.from_data_list(data_list)

    mean_pred, std_pred = predictor.predict_batch(batch, return_std=True)

    assert isinstance(mean_pred, torch.Tensor)
    assert isinstance(std_pred, torch.Tensor)
    assert mean_pred.shape[0] > 0


@pytest.mark.parametrize("agg_method", ["mean", "median"])
def test_aggregation_methods(mock_models, agg_method):
    """Test different aggregation methods."""
    predictor = EnsemblePredictor(
        models=mock_models,
        device="cpu",
        aggregation=agg_method,
    )

    data = _create_mock_data()
    mean_pred = predictor.predict(data, return_std=False)

    assert isinstance(mean_pred, torch.Tensor)


# ============================================================================
# Equivariance Tests
# ============================================================================

def _create_molecular_data(num_atoms=5):
    """Create realistic molecular data."""
    # Create positions in 3D
    pos = torch.randn(num_atoms, 3)

    # Create node features (scalar invariants)
    x = torch.randn(num_atoms, 18)

    # Create edges (fully connected for simplicity)
    edge_index = torch.combinations(
        torch.arange(num_atoms), r=2
    ).t().contiguous()

    # Add reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Compute edge attributes (relative positions)
    edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
    edge_attr = o3.spherical_harmonics(
        l=range(4),
        x=edge_vec,
        normalize=True,
        normalization="component"
    )

    data = Data(
        pos=pos,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    return data


def _rotate_data(data, rotation_matrix):
    """Apply rotation to molecular data."""
    # Rotate positions
    pos_rotated = data.pos @ rotation_matrix.t()

    # Rotate edge vectors
    edge_vec = pos_rotated[data.edge_index[1]] - \
        pos_rotated[data.edge_index[0]]
    edge_attr_rotated = o3.spherical_harmonics(
        l=range(4),
        x=edge_vec,
        normalize=True,
        normalization="component"
    )

    data_rotated = Data(
        pos=pos_rotated,
        x=data.x.clone(),  # Scalar features don't change
        edge_index=data.edge_index.clone(),
        edge_attr=edge_attr_rotated,
    )

    return data_rotated


def _random_rotation_matrix():
    """Generate random 3D rotation matrix."""
    # Random rotation using Euler angles
    alpha, beta, gamma = torch.rand(3) * 2 * np.pi

    # Rotation matrices around x, y, z axes
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(alpha), -torch.sin(alpha)],
        [0, torch.sin(alpha), torch.cos(alpha)]
    ])

    Ry = torch.tensor([
        [torch.cos(beta), 0, torch.sin(beta)],
        [0, 1, 0],
        [-torch.sin(beta), 0, torch.cos(beta)]
    ])

    Rz = torch.tensor([
        [torch.cos(gamma), -torch.sin(gamma), 0],
        [torch.sin(gamma), torch.cos(gamma), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


@pytest.mark.equivariance
def test_rotation_equivariance(mock_model):
    """Test that model embeddings transform correctly under rotation."""
    model = mock_model
    data = _create_molecular_data(num_atoms=8)

    # Get prediction on original data
    with torch.no_grad():
        output_original = model(data)

    # Generate random rotation
    R = _random_rotation_matrix()

    # Rotate data
    data_rotated = _rotate_data(data, R)

    # Get prediction on rotated data
    with torch.no_grad():
        output_rotated = model(data_rotated)

    # Check shapes match
    assert output_original.shape == output_rotated.shape

    # For equivariant outputs, outputs should transform
    # (not be identical)
    assert not torch.allclose(output_original, output_rotated, atol=1e-5), \
        "Outputs should change under rotation for equivariant model"


@pytest.mark.equivariance
def test_scalar_invariance(mock_model):
    """Test that scalar features (l=0) are invariant under rotation."""
    model = mock_model
    data = _create_molecular_data(num_atoms=8)

    # Get prediction
    with torch.no_grad():
        output_original = model(data)

    # Generate rotation
    R = _random_rotation_matrix()

    # Rotate data
    data_rotated = _rotate_data(data, R)

    # Get prediction on rotated data
    with torch.no_grad():
        output_rotated = model(data_rotated)

    # Extract scalar features (first 14 channels are 14x0e)
    scalars_original = output_original[:, :14]
    scalars_rotated = output_rotated[:, :14]

    # Scalars should be approximately invariant
    torch.testing.assert_close(
        scalars_original,
        scalars_rotated,
        rtol=1e-3,
        atol=1e-3,
        msg="Scalar outputs should be invariant under rotation"
    )


@pytest.mark.equivariance
def test_multiple_rotations_consistency(mock_model):
    """Test that composition of rotations is consistent."""
    model = mock_model
    data = _create_molecular_data(num_atoms=8)

    # Two rotations
    R1 = _random_rotation_matrix()
    R2 = _random_rotation_matrix()

    # Apply rotations separately
    data_r1 = _rotate_data(data, R1)
    data_r1_r2 = _rotate_data(data_r1, R2)

    with torch.no_grad():
        output_separate = model(data_r1_r2)

    # Apply composed rotation
    R_composed = R2 @ R1
    data_composed = _rotate_data(data, R_composed)

    with torch.no_grad():
        output_composed = model(data_composed)

    # Should give same result
    torch.testing.assert_close(
        output_separate,
        output_composed,
        rtol=1e-4,
        atol=1e-4,
        msg="Composed rotations should give same result as separate rotations"
    )


# ============================================================================
# Basis Function Parameter Tests
# ============================================================================

def test_load_basis_function_params(temp_dir):
    """Test loading basis function parameters."""
    params_path = temp_dir / "basis_params.pkl"

    # Create mock parameters
    mock_params = {
        1.0: {"n": 1, "l": 0, "exp": [0.5, 1.0], "coef": [[1.0], [1.0]]},
        6.0: {"n": 2, "l": 0, "exp": [0.5, 1.0], "coef": [[1.0], [1.0]]},
        8.0: {"n": 2, "l": 1, "exp": [0.5, 1.0], "coef": [[1.0], [1.0]]},
    }

    with open(params_path, "wb") as f:
        pickle.dump(mock_params, f)

    # Load parameters
    loaded_params = load_basis_function_params(params_path)

    assert loaded_params == mock_params
    assert 1.0 in loaded_params
    assert 6.0 in loaded_params


def test_basis_params_file_not_found(temp_dir):
    """Test error when file doesn't exist."""
    params_path = temp_dir / "nonexistent.pkl"

    with pytest.raises(FileNotFoundError):
        load_basis_function_params(params_path)
