"""Experiment registry. Import an experiment module and call `_register`."""
from __future__ import annotations

from .base import Experiment
from .exp1_homologous import Exp1Homologous
from .exp2_functional_group import Exp2FunctionalGroup


EXPERIMENTS: dict[str, Experiment] = {}


def _register(e: Experiment) -> None:
    if e.id in EXPERIMENTS:
        raise ValueError(f"duplicate experiment id: {e.id}")
    EXPERIMENTS[e.id] = e


_register(Exp1Homologous())
_register(Exp2FunctionalGroup())
