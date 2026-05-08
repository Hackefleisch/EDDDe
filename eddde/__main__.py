import argparse

import eddde
from .data import elektronn_runner, pipeline
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
        "--num-workers",
        type=int,
        default=eddde.N_WORKERS,
        help=(
            f"Process pool size for CPU-bound stages (SMILES filtering, "
            f"conformer generation). Default: cpu_count = {eddde.N_WORKERS}."
        ),
    )
    p.add_argument(
        "--test-mode",
        action="store_true",
        help=(
            "Dev acceleration: randomly downsample every dataset's SMILES stage "
            "to at most --test-size rows (seeded with the project SEED). "
            "Cache invalidates against full-mode artifacts when toggled."
        ),
    )
    p.add_argument(
        "--test-size",
        type=int,
        default=1000,
        help="Max rows per dataset under --test-mode (default: 1000).",
    )
    return p.parse_args()


args = _parse_args()
elektronn_runner.BATCH_SIZE = args.batch_size
elektronn_runner.NUM_WORKERS = args.dataloader_workers
eddde.N_WORKERS = args.num_workers
if args.test_mode:
    pipeline.TEST_MODE_SIZE = args.test_size

main()
