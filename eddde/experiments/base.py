"""Experiment protocol and on-disk layout.

Per PROJECT_PLAN.md §7, each experiment materializes results under
`results/EXP-X/`. We keep per-(method, dataset) raw output in
`results/EXP-X/{method_id}/{dataset_id}/` so staleness can be tracked
independently; aggregated CSVs across methods are produced separately.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


RESULTS_ROOT = Path("results")


def result_dir(exp_id: str, method_id: str, dataset_id: str) -> Path:
    return RESULTS_ROOT / exp_id / method_id / dataset_id


class Experiment(Protocol):
    id: str
    version: str
    datasets: list[str]

    def run(
        self,
        method: Any,
        stage_data: dict,
        embeddings: dict,
        dataset_id: str,
        out: Path,
    ) -> dict: ...
