#!/usr/bin/env python
"""
EDDDe Predict Command

Run molecular descriptor predictions using ElektroNN ensemble.
"""

import sys
from pathlib import Path
import pickle
import warnings

import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf

from .data import (
    create_dataset_from_csv,
    create_dataset_from_sdf,
    ensure_conformers_pregenerated,
)
from .models import load_basis_function_params, ModelManager, EnsemblePredictor
from .inference import InferenceEngine


def run_predict(cfg: DictConfig):
    """Run molecular descriptor predictions."""
    print("=" * 60)
    print("EDDDe - Molecular Descriptor Prediction")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Suppress warnings if configured
    if cfg.model.get("suppress_warnings", True):
        warnings.filterwarnings("ignore")

    # Load basis parameters
    print(f"\nLoading basis parameters from {cfg.paths.basis_params}...")
    basis_params = load_basis_function_params(cfg.paths.basis_params)
    print(f"  Loaded parameters for {len(basis_params)} atom types")

    # Determine SDF path for pre-generated conformers
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use provided SDF or generate from CSV
    sdf_file = cfg.data.get("sdf_file")
    csv_file = cfg.data.get("csv_file")
    
    # If CSV is provided, ensure conformers are pre-generated
    if csv_file is not None:
        # Determine output SDF path
        if sdf_file is None:
            # Auto-generate SDF path based on CSV filename
            csv_path = Path(csv_file)
            sdf_filename = csv_path.stem + "_conformers.sdf"
            sdf_file = output_dir / sdf_filename
        else:
            sdf_file = Path(sdf_file)
        
        # Ensure conformers are pre-generated (will use existing SDF if it exists)
        sdf_file = ensure_conformers_pregenerated(
            csv_path=Path(csv_file),
            sdf_path=sdf_file if sdf_file.exists() else None,
            output_sdf_path=sdf_file,
            basisfunction_params=basis_params,
            smiles_col=cfg.data.get("smiles_col", "SMILES"),
            id_col=cfg.data.get("id_col", "CID"),
            conformer_seed=cfg.data.get("conformer_seed", 42069),
            filter_bad_atoms=cfg.data.get("filter_bad_atoms", True),
            verbose=True,
        )
    elif sdf_file is not None:
        # Only SDF provided, use it directly
        sdf_file = Path(sdf_file)
        if not sdf_file.exists():
            print(f"Error: SDF file not found: {sdf_file}")
            sys.exit(1)
    
    # Create dataset from SDF (now guaranteed to exist)
    print("\nLoading molecular dataset...")
    if sdf_file is not None:
        print(f"  Source: SDF file ({sdf_file})")
        dataset = create_dataset_from_sdf(
            sdf_path=sdf_file,
            basisfunction_params=basis_params,
            filter_bad_atoms=cfg.data.get("filter_bad_atoms", True),
            verbose=cfg.data.get("verbose", False),
        )
    else:
        print("Error: Must provide either csv_file or sdf_file in data configuration")
        sys.exit(1)

    stats = dataset.get_statistics()
    print(f"  Loaded {stats['num_molecules']} molecules")
    print(f"  Filtered {stats['num_filtered']} molecules")
    if stats.get('cache_enabled', False):
        print(
            f"  Conformer caching: enabled ({stats['cached_conformers']} cached)")

    # Load models
    print(f"\nLoading models from {cfg.paths.model_dir}...")
    model_manager = ModelManager(
        model_dir=cfg.paths.model_dir,
        device=cfg.device,
        suppress_warnings=cfg.model.get("suppress_warnings", True),
    )

    if cfg.model.use_all_folds:
        models = model_manager.load_ensemble(num_folds=cfg.model.num_folds)
    elif cfg.model.specific_folds:
        models = model_manager.load_ensemble(
            num_folds=cfg.model.num_folds,
            fold_indices=cfg.model.specific_folds,
        )
    else:
        models = [model_manager.load_single_model()]

    print(f"  Loaded {len(models)} models")

    # Wrap models in DataParallel if requested
    if cfg.get("ddp", False) and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"\nEnabling DataParallel across {num_gpus} GPUs...")
            gpu_ids = cfg.get("gpu_ids", None)
            if gpu_ids is not None:
                device_ids = list(gpu_ids)
                print(f"  Using specific GPUs: {device_ids}")
            else:
                device_ids = None
                print(f"  Using all available GPUs: {list(range(num_gpus))}")

            # Wrap each model in DataParallel
            wrapped_models = []
            for model in models:
                wrapped_model = nn.DataParallel(model, device_ids=device_ids)
                wrapped_models.append(wrapped_model)
            models = wrapped_models
        else:
            print("\nDataParallel requested but only 1 GPU available, skipping...")
    elif cfg.get("use_data_parallel", False):
        print("\nWarning: DataParallel requested but CUDA not available, skipping...")

    # Create predictor
    agg_func = cfg.model.get("aggregation", "mean")
    predictor = EnsemblePredictor(
        models=models,
        device=cfg.device,
        aggregation=agg_func,
    )

    # Create inference engine
    print("\nRunning predictions...")
    engine = InferenceEngine(
        predictor=predictor,
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.inference.num_workers,
        pin_memory=cfg.inference.pin_memory,
        prefetch_factor=cfg.inference.prefetch_factor,
        show_progress=cfg.inference.show_progress,
    )

    # Run predictions
    error_strategy = cfg.inference.get("error_strategy", "skip")
    predictions = engine.predict_dataset(
        dataset,
        return_std=cfg.inference.return_std,
        error_strategy=error_strategy,
    )

    # Save results
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{agg_func}.pkl"

    with open(output_path, "wb") as f:
        pickle.dump(predictions, f)

    print("\n" + "=" * 60)
    print(f"Predictions complete: {len(predictions)} molecules")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Run molecular descriptor predictions."""
    run_predict(cfg)


if __name__ == "__main__":
    main()
