"""
EDDDe - Electron Density Derived Descriptors

A scientific software package for deriving and using molecular descriptors
from ElektroNN embeddings.

Main Components
---------------
- data: Data loading and preprocessing
- models: Model management and ensemble prediction
- inference: Efficient batch inference engine

Quick Start
-----------
>>> from eddde import create_inference_pipeline
>>> from eddde.models import load_basis_function_params
>>> from eddde.data import create_dataset_from_csv
>>>
>>> # Load basis function parameters
>>> basis_params = load_basis_function_params("ElektroNN/basisfunction_params.pkl")
>>>
>>> # Create dataset
>>> dataset = create_dataset_from_csv(
...     "data/molecules.csv",
...     basis_params,
...     smiles_col="SMILES",
...     id_col="CID"
... )
>>>
>>> # Create inference pipeline
>>> engine = create_inference_pipeline(
...     model_dir="ElektroNN/modelparams/04-20-33/",
...     device="cuda",
...     batch_size=16
... )
>>>
>>> # Run predictions
>>> predictions = engine.predict_dataset(dataset)
"""

__version__ = "0.1.0"

# Import main classes and functions for convenient access
from .data import (
    MoleculeDataset,
    SDFMoleculeDataset,
    load_smiles_csv,
    load_molecules_from_sdf,
    create_dataset_from_csv,
    create_dataset_from_sdf,
    convert_smiles_to_sdf,
    convert_csv_to_sdf,
)

from .models import (
    ModelManager,
    EnsemblePredictor,
    load_basis_function_params,
    DEFAULT_MODEL_KWARGS,
)

from .inference import (
    InferenceEngine,
    BatchProcessor,
    create_inference_pipeline,
)

__all__ = [
    # Data
    "MoleculeDataset",
    "SDFMoleculeDataset",
    "load_smiles_csv",
    "load_molecules_from_sdf",
    "create_dataset_from_csv",
    "create_dataset_from_sdf",
    "convert_smiles_to_sdf",
    "convert_csv_to_sdf",
    # Models
    "ModelManager",
    "EnsemblePredictor",
    "load_basis_function_params",
    "DEFAULT_MODEL_KWARGS",
    # Inference
    "InferenceEngine",
    "BatchProcessor",
    "create_inference_pipeline",
]
