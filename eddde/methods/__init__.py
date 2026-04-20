"""Method registry. Import a method module and call `_register` to add it."""
from __future__ import annotations

from .base import Method
from .baselines.ecfp import ECFP4, ECFP6, FCFP4


METHODS: dict[str, Method] = {}


def _register(m: Method) -> None:
    if m.id in METHODS:
        raise ValueError(f"duplicate method id: {m.id}")
    METHODS[m.id] = m


_register(ECFP4())
_register(ECFP6())
_register(FCFP4())
