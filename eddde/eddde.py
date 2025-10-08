#!/usr/bin/env python
"""
EDDDe Command-Line Interface

Main CLI entry point for EDDDe with Hydra configuration.
"""

import sys

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main CLI entry point with Hydra configuration.

    By default, runs prediction. Use task=convert to run conversion.
    """
    # Determine which task to run
    task = cfg.get("task", "predict")

    if task == "convert":
        # Run conversion task
        print("Running conversion task...")
        from .convert import run_convert as convert_main
        convert_main(cfg)
    elif task == "predict":
        # Run prediction task
        print("Running prediction task...")
        from .predict import run_predict as predict_main
        predict_main(cfg)
    else:
        print(f"Error: Unknown task '{task}'")
        print("Valid tasks: 'convert', 'predict'")
        sys.exit(1)


def cli_help():
    """Print CLI help information."""
    print("EDDDe - Electron Density Derived Descriptors")
    print("\nAvailable commands:")
    print("  eddde          - Run with Hydra configuration (default: predict)")
    print("  eddde-convert  - Convert CSV with SMILES to SDF with 3D conformers")
    print("  eddde-predict  - Run molecular descriptor predictions")
    print("\nExamples:")
    print("  eddde data=smrt                              # Run prediction")
    print("  eddde task=convert data=wellqrate convert.output_sdf=output.sdf")
    print("  eddde-convert data=wellqrate convert.output_sdf=output.sdf")
    print("  eddde-predict data=smrt inference.batch_size=1024")
    print("\nFor Hydra options, run commands with --help")


if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]):
        cli_help()
    else:
        main()
