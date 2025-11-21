from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn

from pathbench.core.losses.base import SurvivalDiscreteLoss
from pathbench.utils.registries import LOSSES

try:
    from pycox.models.loss import NLLLogisticHazard as _PyCoxLogHazNLL
except ImportError:
    _PyCoxLogHazNLL = None

@LOSSES.register("DiscreteTimeNLLLoss")
class DiscreteTimeNLLLoss(SurvivalDiscreteLoss):
    def __init__(self):
        super().__init__()
        self.impl = _PyCoxLogHazNLL() if _PyCoxLogHazNLL is not None else None
        
        if self.impl is None:
            self.bcel = nn.BCEWithLogitsLoss(reduction="none")

    def calculate_loss(self, preds: torch.Tensor, time: torch.Tensor, event: torch.Tensor, **_: Any) -> torch.Tensor:
        if self.impl is not None:
            return self.impl(preds, time, event)
        
        raise NotImplementedError("Native fallback for DiscreteTimeNLLLoss is pending.")