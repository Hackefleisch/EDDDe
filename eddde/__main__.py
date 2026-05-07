import argparse

from .data import conformers, elektronn_runner
from .runner import main


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m eddde", description="Run the EDDDe benchmark pipeline.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=elektronn_runner.BATCH_SIZE,
        help=f"ElektroNN GPU batch size (default: {elektronn_runner.BATCH_SIZE}).",
    )
    p.add_argument(
        "--dataloader-workers",
        type=int,
        default=elektronn_runner.NUM_WORKERS,
        help=(
            f"torch DataLoader workers for ElektroNN inference "
            f"(default: {elektronn_runner.NUM_WORKERS}). 0 disables prefetch."
        ),
    )
    p.add_argument(
        "--conformer-workers",
        type=int,
        default=conformers.N_WORKERS,
        help=(
            f"Process pool size for conformer generation "
            f"(default: cpu_count = {conformers.N_WORKERS})."
        ),
    )
    return p.parse_args()


args = _parse_args()
elektronn_runner.BATCH_SIZE = args.batch_size
elektronn_runner.NUM_WORKERS = args.dataloader_workers
conformers.N_WORKERS = args.conformer_workers

main()
