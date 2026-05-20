from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """Root abstract base class for all PathBench losses."""

    def __init__(self, task_type: str):
        super().__init__()
        self.task_type = task_type

    @abstractmethod
    def forward(
        self,
        preds: torch.Tensor,
        target: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return a scalar loss tensor for one prediction/target batch."""
