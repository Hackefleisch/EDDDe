"""
Example usage of EDDDe with Hydra configuration.

This script demonstrates how to use Hydra for configuration management,
allowing you to easily switch between different settings without changing code.

Usage:
    python example_with_hydra.py
    python example_with_hydra.py data=smrt
    python example_with_hydra.py data=wellqrate
    python example_with_hydra.py data.csv_file=data/qsar/AID1798_actives.csv
    python example_with_hydra.py data.sdf_file=data/SMRT/SMRT_dataset.sdf
"""

import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from eddde.models import ModelManager, EnsemblePredictor, load_basis_function_params
from eddde.data import create_dataset_from_csv, create_dataset_from_sdf
from eddde.inference import InferenceEngine


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function with Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    """
    # Print configuration
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Suppress warnings if needed
    if cfg.get("suppress_warnings", True):
        warnings.filterwarnings("ignore")

    # Step 1: Load basis function parameters
    print("\n[1/5] Loading basis function parameters...")
    basis_params = load_basis_function_params(cfg.paths.basis_params)
    print(f"  Loaded parameters for {len(basis_params)} atom types")

    # Step 2: Create dataset
    print("\n[2/5] Creating molecular dataset...")

    # Determine data source and create appropriate dataset
    if cfg.data.get("csv_file") is not None:
        # Load from CSV file with SMILES
        print(f"  Loading from CSV: {cfg.data.csv_file}")
        dataset = create_dataset_from_csv(
            csv_path=cfg.data.csv_file,
            basisfunction_params=basis_params,
            smiles_col=cfg.data.get("smiles_col", "SMILES"),
            id_col=cfg.data.get("id_col", "CID"),
            conformer_seed=cfg.data.get("conformer_seed", 42069),
            filter_bad_atoms=cfg.data.get("filter_bad_atoms", True),
            verbose=cfg.data.get("verbose", False),
            cache_conformers=cfg.data.get("cache_conformers", False),
            pregenerate_conformers=cfg.data.get(
                "pregenerate_conformers", False),
        )
    elif cfg.data.get("sdf_file") is not None:
        # Load from SDF file (already has 3D conformers)
        print(f"  Loading from SDF: {cfg.data.sdf_file}")
        dataset = create_dataset_from_sdf(
            sdf_path=cfg.data.sdf_file,
            basisfunction_params=basis_params,
            filter_bad_atoms=cfg.data.get("filter_bad_atoms", True),
            verbose=cfg.data.get("verbose", False),
        )
    else:
        raise ValueError(
            "No data source specified. Please provide either 'csv_file' or 'sdf_file' in the configuration."
        )

    stats = dataset.get_statistics()
    print(f"  Loaded {stats['num_molecules']} molecules")
    print(f"  Filtered {stats['num_filtered']} molecules")
    if stats.get('cache_enabled', False):
        print(
            f"  Conformer caching: enabled ({stats['cached_conformers']} cached)")

    # Step 3: Load models
    print("\n[3/5] Loading ElektroNN models...")
    model_manager = ModelManager(
        model_dir=cfg.paths.model_dir,
        device=cfg.device,
        suppress_warnings=cfg.model.suppress_warnings,
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

    model_info = model_manager.get_model_info()
    print(f"  Loaded {model_info['num_models']} models")

    # Step 4: Create inference engine
    print("\n[4/5] Setting up inference engine...")
    predictor = EnsemblePredictor(
        models=models,
        device=cfg.device,
        aggregation=cfg.model.get("aggregation", "mean"),
    )

    engine = InferenceEngine(
        predictor=predictor,
        batch_size=cfg.inference.batch_size,
        num_workers=cfg.inference.num_workers,
        pin_memory=cfg.inference.pin_memory,
        prefetch_factor=cfg.inference.prefetch_factor,
        show_progress=cfg.inference.show_progress,
    )

    # Step 5: Run predictions
    print("\n[5/5] Running batch predictions...")
    predictions = engine.predict_dataset(
        dataset,
        return_std=cfg.inference.return_std,
    )

    print(f"\n{'=' * 60}")
    print("Results Summary")
    print(f"{'=' * 60}")
    print(f"  Total molecules processed: {len(predictions)}")

    # Save results
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "predictions.pkl"
    import pickle
    with open(output_path, "wb") as f:
        pickle.dump(predictions, f)

    print(f"\n  Saved predictions to: {output_path}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}\n")

    return predictions


if __name__ == "__main__":
    main()
