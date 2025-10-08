#!/usr/bin/env python
"""
EDDDe Convert Command

Convert CSV with SMILES to SDF with 3D conformers.
"""

import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from .data import convert_csv_to_sdf
from .models import load_basis_function_params


def run_convert(cfg: DictConfig):
    """
    Execute the conversion task.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration
    """
    print("=" * 60)
    print("EDDDe - SMILES to SDF Converter")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    if not cfg.data.get("csv_file"):
        print("Error: csv_file must be specified in configuration")
        sys.exit(1)

    if not cfg.convert.get("output_sdf"):
        print("Error: convert.output_sdf must be specified")
        sys.exit(1)

    # Load basis params if filtering
    basis_params = None
    if cfg.data.get("filter_bad_atoms", False):
        print(f"\nLoading basis parameters from {cfg.paths.basis_params}...")
        basis_params = load_basis_function_params(cfg.paths.basis_params)
        print(f"  Loaded parameters for {len(basis_params)} atom types")

    # Convert
    print("\nConverting molecules...")
    success, failed = convert_csv_to_sdf(
        csv_path=cfg.data.csv_file,
        output_sdf_path=cfg.convert.output_sdf,
        smiles_col=cfg.data.get("smiles_col", "SMILES"),
        id_col=cfg.data.get("id_col", "CID"),
        conformer_seed=cfg.data.get("conformer_seed", 42069),
        add_hydrogens=True,
        optimize_geometry=cfg.convert.get("optimize_geometry", False),
        verbose=True,
        filter_bad_atoms=cfg.data.get("filter_bad_atoms", False),
        basisfunction_params=basis_params,
    )

    print("\n" + "=" * 60)
    print(f"Conversion complete: {success} success, {failed} failed")
    print("=" * 60)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """Convert CSV with SMILES to SDF with 3D conformers."""
    run_convert(cfg)


if __name__ == "__main__":
    main()
