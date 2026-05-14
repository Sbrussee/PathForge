"""Backward-compatible loss imports.

Historically, tests and callers imported concrete loss classes from
``pathbench.core.losses.impl``. The current loss implementation is split across
task-specific modules, so this file re-exports the concrete classes to preserve
that import path.
"""

from __future__ import annotations

from .classification import CrossEntropyLoss
from .regression import MSELoss
from .survival_continuous import CoxPHLoss
from .survival_discrete import DiscreteTimeNLLLoss

__all__ = [
    "CrossEntropyLoss",
    "MSELoss",
    "CoxPHLoss",
    "DiscreteTimeNLLLoss",
]
