"""
Pytest configuration and shared fixtures for EDDDe tests.
"""

import tempfile
import shutil
from pathlib import Path
from importlib.resources import read_binary
import pickle

import pytest
import torch
from e3nn.nn.models.gate_points_2101 import Network

from eddde.models import load_basis_function_params, DEFAULT_MODEL_KWARGS


@pytest.fixture(scope="session")
def basis_params():
    """Load real basis function parameters from elektronn."""
    basisfunction_params_binary = read_binary("elektronn.models", "basisfunction_params.pkl")
    return pickle.loads(basisfunction_params_binary)


@pytest.fixture
def temp_dir():
    """Create and cleanup temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_model():
    """Create a mock ElektroNN model."""
    model = Network(**DEFAULT_MODEL_KWARGS)
    model.eval()
    return model


@pytest.fixture
def mock_models():
    """Create multiple mock models for ensemble testing."""
    models = []
    for _ in range(3):
        model = Network(**DEFAULT_MODEL_KWARGS)
        model.eval()
        models.append(model)
    return models


@pytest.fixture
def model_dir(temp_dir, mock_model):
    """Create directory with mock model checkpoints."""
    model_dir = temp_dir / "models"
    model_dir.mkdir()

    # Save mock model checkpoints
    for fold in range(1, 4):
        model_path = model_dir / f"model_fold_{fold}.pth"
        torch.save(mock_model.state_dict(), model_path)

    return model_dir
