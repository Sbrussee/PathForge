from __future__ import annotations
from typing import Any
import torch

from pathbench.core.losses.base import SurvivalContinuousLoss
from pathbench.utils.registries import LOSSES

try:
    from pycox.models.loss import CoxPHLoss as _PyCoxPHLoss
except ImportError:
    _PyCoxPHLoss = None

@LOSSES.register("CoxPHLoss")
class CoxPHLoss(SurvivalContinuousLoss):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        self.impl = _PyCoxPHLoss() if _PyCoxPHLoss is not None else None

    def calculate_loss(self, preds: torch.Tensor, time: torch.Tensor, event: torch.Tensor, **_: Any) -> torch.Tensor:
        if self.impl is None:
            raise RuntimeError("pycox is not installed or CoxPHLoss is unavailable.")
        
        return self.impl(preds, time, event)