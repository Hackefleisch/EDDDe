"""
Model management for ElektroNN ensemble predictions.

This module provides classes for loading, managing, and running predictions
with ElektroNN models, including ensemble prediction with uncertainty quantification.
"""

from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import warnings
import pickle

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from e3nn.nn.models.gate_points_2101 import Network
from e3nn import o3


# Default model architecture configuration
DEFAULT_MODEL_KWARGS = {
    "irreps_in": "18x 0e",
    "irreps_hidden": [
        (mul, (l, p)) for l, mul in enumerate([125, 40, 25, 15]) for p in [-1, 1]
    ],
    "irreps_out": "14x0e + 14x1o + 5x2e + 4x3o + 2x4e",
    "irreps_node_attr": None,
    "irreps_edge_attr": o3.Irreps.spherical_harmonics(3),
    "layers": 3,
    "max_radius": 3.5,
    "number_of_basis": 10,
    "radial_layers": 1,
    "radial_neurons": 128,
    "num_neighbors": 12.2298,
    "num_nodes": 24,
    "reduce_output": False,
}


class ModelManager:
    """
    Manages loading and configuration of ElektroNN models.

    This class handles loading single or ensemble models from disk,
    device management, and provides a consistent interface for model access.

    Parameters
    ----------
    model_dir : Path or str
        Directory containing model checkpoint files
    device : str, optional
        Device to load models on ('cuda', 'cpu', 'mps') (default: 'cuda')
    model_kwargs : Dict, optional
        Model architecture configuration (default: DEFAULT_MODEL_KWARGS)
    suppress_warnings : bool, optional
        Whether to suppress warnings during loading (default: True)

    Attributes
    ----------
    models : List[nn.Module]
        List of loaded model instances
    device : torch.device
        Device where models are loaded
    num_models : int
        Number of loaded models
    """

    def __init__(
        self,
        model_dir: Union[Path, str],
        device: str = "cuda",
        model_kwargs: Optional[Dict] = None,
        suppress_warnings: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.model_kwargs = model_kwargs or DEFAULT_MODEL_KWARGS
        self.models: List[nn.Module] = []
        self.suppress_warnings = suppress_warnings

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}")

    def load_ensemble(
        self,
        num_folds: int = 5,
        fold_indices: Optional[List[int]] = None,
    ) -> List[nn.Module]:
        """
        Load an ensemble of models from k-fold cross-validation.

        Parameters
        ----------
        num_folds : int, optional
            Total number of folds (default: 5)
        fold_indices : List[int], optional
            Specific fold indices to load (1-indexed). If None, loads all folds.
            (default: None)

        Returns
        -------
        List[nn.Module]
            List of loaded model instances

        Examples
        --------
        >>> manager = ModelManager('path/to/models')
        >>> models = manager.load_ensemble(num_folds=5)  # Load all 5 folds
        >>> models = manager.load_ensemble(fold_indices=[1, 2, 3])  # Load specific folds
        """
        if self.suppress_warnings:
            warnings.filterwarnings("ignore")

        # Determine which folds to load
        if fold_indices is None:
            fold_indices = list(range(1, num_folds + 1))

        self.models = []

        for fold_idx in fold_indices:
            model_path = self.model_dir / f"model_fold_{fold_idx}.pth"

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model checkpoint not found: {model_path}")

            # Create model instance
            model = Network(**self.model_kwargs)
            model.to(self.device)

            # Load weights
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()  # Set to evaluation mode

            self.models.append(model)
            print(f"Loaded model_fold_{fold_idx}.pth")

        return self.models

    def load_single_model(
        self,
        model_filename: str = "model_fold_1.pth",
    ) -> nn.Module:
        """
        Load a single model checkpoint.

        Parameters
        ----------
        model_filename : str, optional
            Name of the model checkpoint file (default: 'model_fold_1.pth')

        Returns
        -------
        nn.Module
            Loaded model instance
        """
        if self.suppress_warnings:
            warnings.filterwarnings("ignore")

        model_path = self.model_dir / model_filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}")

        # Create and load model
        model = Network(**self.model_kwargs)
        model.to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()

        self.models = [model]
        print(f"Loaded {model_filename}")

        return model

    def to_device(self, device: str):
        """
        Move all loaded models to a different device.

        Parameters
        ----------
        device : str
            Target device ('cuda', 'cpu', 'mps')
        """
        self.device = torch.device(device)
        for model in self.models:
            model.to(self.device)

    def get_model_info(self) -> Dict:
        """
            Get information about loaded models.

            Returns
            -------
            Dict
                Dictionary containing model information including number of models,
                device, and total parameters
        """
        if not self.models:
            return {
                "num_models": 0,
                "device": str(self.device),
                "total_parameters": 0,
            }

        # Count parameters in first model (all should be identical)
        num_params = sum(p.numel() for p in self.models[0].parameters())

        return {
            "num_models": len(self.models),
            "device": str(self.device),
            "parameters_per_model": num_params,
            "total_parameters": num_params * len(self.models),
        }


class EnsemblePredictor:
    """
        Handles ensemble predictions with uncertainty quantification.
        This class wraps multiple models and provides ensemble prediction
        with mean and standard deviation across model outputs.

        Parameters
        ----------
        models : List[nn.Module]
            List of model instances for ensemble prediction
        device : str or torch.device, optional
            Device for inference (default: 'cuda')
        aggregation : str, optional
            Aggregation method ('mean', 'median') (default: 'mean')

        Attributes
        ----------
        models : List[nn.Module]
            Model instances
        device : torch.device
            Inference device
        num_models : int
            Number of models in ensemble
    """

    def __init__(
        self,
        models: List[nn.Module],
        device: Union[str, torch.device] = "cuda",
        aggregation: str = "mean",
    ):
        if not models:
            raise ValueError(
                "At least one model required for ensemble prediction")

        self.models = models
        self.device = torch.device(device) if isinstance(
            device, str) else device
        self.aggregation = aggregation
        self.num_models = len(models)

        # Ensure all models are in eval mode
        for model in self.models:
            model.eval()

    @torch.no_grad()
    def predict(
        self,
        data: Data,
        return_std: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform ensemble prediction on a single molecule.

        Parameters
        ----------
        data : Data
            PyTorch Geometric Data object representing the molecule
        return_std : bool, optional
            Whether to return standard deviation (default: True)

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            If return_std is True: (mean_prediction, std_prediction)
            If return_std is False: mean_prediction only
        """
        # Move data to device
        data = data.to(self.device)

        # Collect predictions from all models
        predictions = []
        for model in self.models:
            pred = model(data)
            predictions.append(pred)

        # Stack predictions [num_models, num_atoms, num_features]
        predictions = torch.stack(predictions, dim=0)

        # Compute mean and std
        if self.aggregation == "mean":
            mean_pred = predictions.mean(dim=0)
        elif self.aggregation == "median":
            mean_pred = predictions.median(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        if return_std:
            std_pred = predictions.std(dim=0)
            return mean_pred, std_pred
        else:
            return mean_pred

    @torch.no_grad()
    def predict_batch(
        self,
        batch: Batch,
        return_std: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform ensemble prediction on a batch of molecules.

        Parameters
        ----------
        batch : Batch
            PyTorch Geometric Batch object
        return_std : bool, optional
            Whether to return standard deviation (default: True)

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            If return_std is True: (mean_prediction, std_prediction)
            If return_std is False: mean_prediction only
        """
        # Move batch to device
        batch = batch.to(self.device)

        # Collect predictions from all models
        predictions = []
        for model in self.models:
            pred = model(batch)
            predictions.append(pred)

        # Stack predictions [num_models, total_atoms_in_batch, num_features]
        predictions = torch.stack(predictions, dim=0)

        # Compute mean and std
        if self.aggregation == "mean":
            mean_pred = predictions.mean(dim=0)
        elif self.aggregation == "median":
            mean_pred = predictions.median(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        if return_std:
            std_pred = predictions.std(dim=0)
            return mean_pred, std_pred
        else:
            return mean_pred


def load_basis_function_params(params_path: Union[Path, str]) -> Dict:
    """
    Load basis function parameters from pickle file.

    Parameters
    ----------
    params_path : Path or str
        Path to basisfunction_params.pkl file

    Returns
    -------
    Dict
        Basis function parameters dictionary

    Raises
    ------
    FileNotFoundError
        If the parameters file does not exist
    """
    params_path = Path(params_path)
    if not params_path.exists():
        raise FileNotFoundError(
            f"Basis function parameters not found: {params_path}")

    with open(params_path, "rb") as f:
        basisfunction_params = pickle.load(f)

    return basisfunction_params
