"""
Tests for inference engine functionality.

Tests cover:
- InferenceEngine initialization
- Batch prediction
- Single molecule prediction
- Benchmarking
- Error handling
"""


from pathlib import Path
import pytest
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import copy
from e3nn import o3
import os

from eddde.inference import (
    InferenceEngine,
    BatchProcessor,
    create_inference_pipeline,
)
from eddde.models import EnsemblePredictor, ModelManager


# ============================================================================
# Mock Dataset
# ============================================================================

class MockMoleculeDataset(Dataset):
    """Mock dataset for testing."""

    def __init__(self, num_molecules=10, num_atoms_per_mol=8):
        self.num_molecules = num_molecules
        self.num_atoms_per_mol = num_atoms_per_mol
        self.names = [f"mol_{i}" for i in range(num_molecules)]

    def __len__(self):
        return self.num_molecules

    def __getitem__(self, idx):
        """Return mock molecular data."""
        num_atoms = self.num_atoms_per_mol
        num_edges = num_atoms * 2

        data = Data(
            pos=torch.randn(num_atoms, 3),
            x=torch.randn(num_atoms, 18),
            edge_index=torch.randint(0, num_atoms, (2, num_edges)),
            edge_attr=torch.randn(num_edges, 16),
            filename=self.names[idx],  # Add filename for tracking
        )

        return data

    def get_statistics(self):
        """Return mock statistics."""
        return {
            "num_molecules": self.num_molecules,
            "num_filtered": 0,
            "total_atoms": self.num_molecules * self.num_atoms_per_mol,
        }


# ============================================================================
# InferenceEngine Tests
# ============================================================================

@pytest.fixture
def predictor(mock_models):
    """Create EnsemblePredictor for testing."""
    return EnsemblePredictor(
        models=mock_models,
        device="cpu",
    )


@pytest.fixture
def elektronn() -> InferenceEngine:
    """Create ElektroNN for testing."""
    model_dir = Path(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../ElektroNN/modelparams/04-20-33/")))
    return create_inference_pipeline(
        model_dir=model_dir,
        device="cuda",
        batch_size=4,
        num_workers=0,
        num_folds=1,
    )


@pytest.fixture
def elektronn_ensemble() -> InferenceEngine:
    """Create ElektroNN for testing."""
    model_dir = Path(os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../ElektroNN/modelparams/04-20-33/")))
    return create_inference_pipeline(
        model_dir=model_dir,
        device="cuda",
        batch_size=4,
        num_workers=0,
        num_folds=2,
    )


def test_inference_engine_initialization(predictor):
    """Test InferenceEngine initialization."""
    engine = InferenceEngine(
        predictor=predictor,
        batch_size=4,
        num_workers=0,
        show_progress=False,
    )

    assert engine.batch_size == 4
    assert engine.num_workers == 0
    assert engine.device == predictor.device


def test_predict_dataset(predictor):
    """Test batch prediction on dataset."""
    engine = InferenceEngine(
        predictor=predictor,
        batch_size=4,
        num_workers=0,
        show_progress=False,
    )

    dataset = MockMoleculeDataset(num_molecules=10)
    predictions = engine.predict_dataset(dataset, return_std=True)

    # Check predictions for all molecules
    assert len(predictions) == 10

    # Check structure of predictions
    for mol_name, pred in predictions.items():
        assert "mean" in pred
        assert "std" in pred
        assert isinstance(pred["mean"], torch.Tensor)
        assert isinstance(pred["std"], torch.Tensor)


def test_predict_dataset_no_std(predictor):
    """Test prediction without uncertainty."""
    engine = InferenceEngine(
        predictor=predictor,
        batch_size=4,
        num_workers=0,
        show_progress=False,
    )

    dataset = MockMoleculeDataset(num_molecules=5)
    predictions = engine.predict_dataset(dataset, return_std=False)

    assert len(predictions) == 5

    # Predictions should be tensors, not dicts
    for mol_name, pred in predictions.items():
        assert isinstance(pred, torch.Tensor)


def test_predict_single(predictor):
    """Test single molecule prediction."""
    engine = InferenceEngine(
        predictor=predictor,
        batch_size=1,
        num_workers=0,
        show_progress=False,
    )

    dataset = MockMoleculeDataset(num_molecules=1)
    data = dataset[0]

    mean, std = engine.predict_single(data, return_std=True)

    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
    assert mean.shape == std.shape


def test_benchmark(predictor):
    """Test benchmarking functionality."""
    engine = InferenceEngine(
        predictor=predictor,
        batch_size=4,
        num_workers=0,
        show_progress=False,
    )

    dataset = MockMoleculeDataset(num_molecules=20)
    metrics = engine.benchmark(dataset, num_samples=10)

    # Check metrics are returned
    assert "total_time_seconds" in metrics
    assert "num_molecules" in metrics
    assert "throughput_mol_per_sec" in metrics
    assert "avg_latency_sec_per_mol" in metrics

    # Check values are reasonable
    assert metrics["num_molecules"] == 10
    assert metrics["total_time_seconds"] > 0
    assert metrics["throughput_mol_per_sec"] > 0


# ============================================================================
# BatchProcessor Tests
# ============================================================================

@pytest.fixture
def inference_engine(predictor):
    """Create InferenceEngine for testing."""
    return InferenceEngine(
        predictor=predictor,
        batch_size=4,
        num_workers=0,
        show_progress=False,
    )


def test_batch_processor_initialization(inference_engine):
    """Test BatchProcessor initialization."""
    processor = BatchProcessor(
        inference_engine=inference_engine,
        error_strategy="skip",
    )

    assert processor.error_strategy == "skip"
    assert len(processor.failed_molecules) == 0


def test_process_dataset(inference_engine):
    """Test dataset processing with error handling."""
    processor = BatchProcessor(
        inference_engine=inference_engine,
        error_strategy="skip",
    )

    dataset = MockMoleculeDataset(num_molecules=5)
    predictions = processor.process_dataset(dataset)

    assert len(predictions) == 5


def test_save_predictions(inference_engine, temp_dir):
    """Test saving predictions to disk."""
    processor = BatchProcessor(
        inference_engine=inference_engine,
        error_strategy="skip",
    )

    dataset = MockMoleculeDataset(num_molecules=5)
    save_path = temp_dir / "predictions.pkl"

    predictions = processor.process_dataset(dataset, save_path=save_path)

    assert save_path.exists()

    # Verify can load saved predictions
    import pickle
    with open(save_path, "rb") as f:
        loaded_predictions = pickle.load(f)

    assert len(loaded_predictions) == len(predictions)


# ============================================================================
# Integration Tests
# ============================================================================

def test_create_inference_pipeline(model_dir):
    """Test creating complete inference pipeline."""
    engine = create_inference_pipeline(
        model_dir=model_dir,
        device="cpu",
        batch_size=8,
        num_workers=0,
        num_folds=3,
    )

    assert isinstance(engine, InferenceEngine)
    assert engine.batch_size == 8
    assert engine.predictor.num_models == 3


# ============================================================================
# Equivariance in Inference Tests
# ============================================================================

def _create_mock_molecular_data(name="test"):
    """Create mock molecular data with proper structure."""
    num_atoms = 8

    pos = torch.randn(num_atoms, 3)
    x = torch.randn(num_atoms, 18)

    # Create edges
    edge_index = torch.combinations(torch.arange(num_atoms), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Compute edge attributes
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
        filename=name,
    )

    return data


def _rotate_data(data, angle_deg):
    """Rotate data around z-axis."""
    import numpy as np

    angle = np.deg2rad(angle_deg)
    R = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ], dtype=torch.float32)

    # Rotate positions
    pos_rot = data.pos @ R.t()

    # Recompute edge attributes
    edge_vec = pos_rot[data.edge_index[1]] - pos_rot[data.edge_index[0]]
    edge_attr_rot = o3.spherical_harmonics(
        l=range(4),
        x=edge_vec,
        normalize=True,
        normalization="component"
    )

    data_rot = Data(
        pos=pos_rot,
        x=data.x.clone(),
        edge_index=data.edge_index.clone(),
        edge_attr=edge_attr_rot,
        filename=data.filename,
    )

    return data_rot


@pytest.mark.equivariance
def test_equivariance_random_rotation(elektronn):
    """Test full equivariance of model predictions under random rotation."""
    from e3nn import o3

    # Prepare a mock molecule
    data = _create_mock_molecular_data()

    predictor = elektronn.predictor
    model = predictor.models[0]
    irreps_out = model.irreps_out if hasattr(
        model, "irreps_out") else model.out_irreps

    # Generate a random rotation matrix
    rand_rot = o3.rand_matrix().to(data.pos.device)
    D_out = o3.Irreps(irreps_out).D_from_matrix(
        rand_rot.cpu()).to(data.pos.device)

    # Rotate molecule positions
    data_rot = copy.deepcopy(data)
    data_rot.pos = data.pos @ rand_rot.T

    # Recompute edge attributes for rotated molecule
    edge_vec = data_rot.pos[data_rot.edge_index[1]] - \
        data_rot.pos[data_rot.edge_index[0]]
    data_rot.edge_attr = o3.spherical_harmonics(
        l=range(4),
        x=edge_vec,
        normalize=True,
        normalization="component"
    )

    pred_rot, _ = elektronn.predict_single(data_rot, return_std=True)

    # Predict on original molecule
    pred_orig, _ = elektronn.predict_single(data, return_std=True)

    # Apply D_out.T to original prediction
    pred_orig_rot = pred_orig @ D_out.T.to(pred_orig.device)

    # Check equivariance: predictions should match after rotation
    assert torch.allclose(
        pred_rot, pred_orig_rot, rtol=1e-4, atol=1e-4
    ), "Model predictions are not equivariant under rotation"


@pytest.mark.equivariance
def test_equivariance_random_rotation_ensemble(elektronn_ensemble):
    """Test full equivariance of model predictions under random rotation."""
    data = _create_mock_molecular_data()

    predictor = elektronn_ensemble.predictor
    model = predictor.models[0]
    irreps_out = model.irreps_out if hasattr(
        model, "irreps_out") else model.out_irreps

    # Generate a random rotation matrix
    rand_rot = o3.rand_matrix().to(data.pos.device)
    D_out = o3.Irreps(irreps_out).D_from_matrix(
        rand_rot.cpu()).to(data.pos.device)

    # Rotate molecule positions
    data_rot = copy.deepcopy(data)
    data_rot.pos = data.pos @ rand_rot.T

    pred_rot, _ = elektronn_ensemble.predict_single(data_rot, return_std=True)

    # Predict on original molecule
    pred_orig, _ = elektronn_ensemble.predict_single(data, return_std=True)

    # Apply D_out.T to original prediction
    pred_orig_rot = pred_orig @ D_out.T.to(pred_orig.device)

    # Check equivariance: predictions should match after rotation
    assert torch.allclose(
        pred_rot, pred_orig_rot, rtol=1e-4, atol=1e-4
    ), "Model predictions are not equivariant under rotation"
