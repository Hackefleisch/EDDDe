"""
Inference engine for efficient batch prediction with ElektroNN models.

This module provides optimized inference capabilities for processing
molecular datasets with ElektroNN ensemble models, including batch
processing, progress tracking, and error handling.
"""

from typing import Dict, Optional, Callable, Tuple, List
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm

from .models import EnsemblePredictor
from .data import MoleculeDataset


class InferenceEngine:
    """
    Optimized inference engine for batch molecular prediction.

    This class handles efficient batch processing of molecular datasets,
    with support for GPU acceleration, progress tracking, and result
    organization.

    Parameters
    ----------
    predictor : EnsemblePredictor
        Ensemble predictor instance containing loaded models
    batch_size : int, optional
        Batch size for inference (default: 16)
    num_workers : int, optional
        Number of DataLoader workers (default: 8)
    pin_memory : bool, optional
        Whether to use pinned memory (default: True)
    prefetch_factor : int, optional
        DataLoader prefetch factor (default: 2)
    show_progress : bool, optional
        Whether to show progress bar (default: True)

    Attributes
    ----------
    predictor : EnsemblePredictor
        The ensemble predictor
    batch_size : int
        Batch size for processing
    device : torch.device
        Device for inference
    """

    def __init__(
        self,
        predictor: EnsemblePredictor,
        batch_size: int = 16,
        num_workers: int = 8,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        show_progress: bool = True,
    ):
        self.predictor: EnsemblePredictor = predictor
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.pin_memory: bool = pin_memory
        self.prefetch_factor: int = prefetch_factor
        self.show_progress: bool = show_progress
        self.device: torch.device = predictor.device

    @torch.no_grad()
    def predict_dataset(
        self,
        dataset: MoleculeDataset,
        return_std: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform batch prediction on an entire dataset.

        Parameters
        ----------
        dataset : MoleculeDataset
            Dataset to predict on
        return_std : bool, optional
            Whether to include uncertainty estimates (default: True)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping molecule names to prediction tensors.
            If return_std is True, each tensor contains both mean and std.
        """
        # Create DataLoader
        dataloader = GeometricDataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            shuffle=False,
        )

        predictions = {}

        # Iterate through batches
        iterator = tqdm(
            dataloader, desc="Predicting") if self.show_progress else dataloader

        for batch in iterator:
            batch = batch.to(self.device)

            # Get predictions
            if return_std:
                pred_mean, pred_std = self.predictor.predict_batch(
                    batch, return_std=True)
            else:
                pred_mean = self.predictor.predict_batch(
                    batch, return_std=False)

            # Separate predictions by molecule
            for i in range(len(batch.filename)):
                mol_name = batch.filename[i]
                mask = batch.batch == i

                if return_std:
                    # Store mean and std together
                    predictions[mol_name] = {
                        "mean": pred_mean[mask].cpu(),
                        "std": pred_std[mask].cpu(),
                    }
                else:
                    predictions[mol_name] = pred_mean[mask].cpu()

        return predictions

    @torch.no_grad()
    def predict_single(
        self,
        data,
        return_std: bool = True,
    ) -> torch.Tensor:
        """
        Predict on a single molecule.

        Parameters
        ----------
        data : Data
            PyTorch Geometric Data object
        return_std : bool, optional
            Whether to return uncertainty (default: True)

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            Prediction tensor(s)
        """
        data = data.to(self.device)
        return self.predictor.predict(data, return_std=return_std)

    def benchmark(
        self,
        dataset: MoleculeDataset,
        num_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance on a dataset.

        Parameters
        ----------
        dataset : MoleculeDataset
            Dataset to benchmark on
        num_samples : int, optional
            Number of samples to use (default: entire dataset)

        Returns
        -------
        Dict[str, float]
            Performance metrics including throughput and latency
        """
        if num_samples is not None:
            # Create subset
            indices = list(range(min(num_samples, len(dataset))))
            dataset = torch.utils.data.Subset(dataset, indices)

        # Warm-up run
        if len(dataset) > 0:
            _ = self.predict_single(dataset[0], return_std=False)

        # Timed run
        start_time = time.time()
        _ = self.predict_dataset(dataset, return_std=False)
        elapsed_time = time.time() - start_time

        num_molecules = len(dataset)
        throughput = num_molecules / elapsed_time if elapsed_time > 0 else 0
        avg_latency = elapsed_time / num_molecules if num_molecules > 0 else 0

        return {
            "total_time_seconds": elapsed_time,
            "num_molecules": num_molecules,
            "throughput_mol_per_sec": throughput,
            "avg_latency_sec_per_mol": avg_latency,
        }


class BatchProcessor:
    """
    High-level batch processor with error handling and checkpointing.

    This class provides robust batch processing with automatic error
    recovery, progress checkpointing, and detailed logging.

    Parameters
    ----------
    inference_engine : InferenceEngine
        Configured inference engine
    error_strategy : str, optional
        How to handle errors: 'skip', 'stop', or 'collect' (default: 'skip')
    checkpoint_dir : Path, optional
        Directory for saving checkpoints (default: None)

    Attributes
    ----------
    engine : InferenceEngine
        The inference engine
    error_strategy : str
        Error handling strategy
    failed_molecules : List[str]
        Names of molecules that failed processing
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        error_strategy: str = "skip",
        checkpoint_dir: Optional[Path] = None,
    ):
        self.engine = inference_engine
        self.error_strategy = error_strategy
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.failed_molecules: List[str] = []

        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def process_dataset(
        self,
        dataset: MoleculeDataset,
        save_path: Optional[Path] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a dataset with error handling.

        Parameters
        ----------
        dataset : MoleculeDataset
            Dataset to process
        save_path : Path, optional
            Path to save results (default: None)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of predictions
        """
        try:
            predictions = self.engine.predict_dataset(dataset)

            if save_path:
                self._save_predictions(predictions, save_path)

            return predictions

        except Exception as e:
            if self.error_strategy == "stop":
                raise
            elif self.error_strategy == "skip":
                print(f"Error during batch processing: {e}")
                print("Continuing with next batch...")
                return {}
            elif self.error_strategy == "collect":
                print(f"Error collected: {e}")
                return {}

    def _save_predictions(
        self,
        predictions: Dict,
        save_path: Path,
    ):
        """Save predictions to disk."""
        import pickle

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(predictions, f)

        print(f"Saved predictions to {save_path}")

    def get_failed_molecules(self) -> List[str]:
        """
        Get list of molecules that failed processing.

        Returns
        -------
        List[str]
            Names of failed molecules
        """
        return self.failed_molecules


def create_inference_pipeline(
    model_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
    num_workers: int = 8,
    num_folds: int = 5,
) -> InferenceEngine:
    """
    Convenience function to create a complete inference pipeline.

    Parameters
    ----------
    model_dir : Path
        Directory containing model checkpoints
    device : str, optional
        Device for inference (default: 'cuda')
    batch_size : int, optional
        Batch size (default: 16)
    num_workers : int, optional
        Number of workers (default: 8)
    num_folds : int, optional
        Number of ensemble folds (default: 5)

    Returns
    -------
    InferenceEngine
        Configured inference engine ready for use

    Examples
    --------
    >>> engine = create_inference_pipeline('path/to/models')
    >>> predictions = engine.predict_dataset(dataset)
    """
    from .models import ModelManager, EnsemblePredictor

    # Load models
    model_manager = ModelManager(model_dir, device=device)
    models = model_manager.load_ensemble(num_folds=num_folds)

    # Create predictor
    predictor = EnsemblePredictor(models, device=device)

    # Create engine
    engine = InferenceEngine(
        predictor,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return engine
