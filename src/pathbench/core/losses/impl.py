from __future__ import annotations
from typing import Any
import torch
from torch import nn
from .base import TaskLoss

from pycox.models.loss import CoxPHLoss as _PyCoxPHLoss
from pycox.models.loss import NLLLogisticHazard as _PyCoxLogHazNLL


# --- Classification ---
class CrossEntropyLoss(TaskLoss):
    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__("classification")
        self.loss = nn.CrossEntropyLoss(weight=weight)
    def forward(self, preds: torch.Tensor, target: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.loss(preds, target)

# --- Regression ---
class MSELoss(TaskLoss):
    def __init__(self):
        super().__init__("regression")
        self.loss = nn.MSELoss()
    def forward(self, preds: torch.Tensor, target: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.loss(preds, target)

# --- Survival (continuous time Cox partial likelihood) ---
class CoxPHLoss(TaskLoss):
    """Cox partial negative log-likelihood.
    Uses **pycox** when available; otherwise falls back to a stable native implementation.
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__("survival")
        self.eps = eps
        self.impl = _PyCoxPHLoss() 

    def forward(self, preds: torch.Tensor, target: dict[str, torch.Tensor], **_: Any) -> torch.Tensor:
        if self.impl is not None:
            time = target["time"].reshape(-1)
            event = target["event"].reshape(-1).float()
            return self.impl(preds.reshape(-1), time, event)

# --- Discrete-time survival (logistic hazard / NLL) ---
class DiscreteTimeNLLLoss(TaskLoss):
    """NLL for logistic hazard (a.k.a. Nnet-Survival).
    Uses **pycox** `NLLLogisticHazard` if available; else native BCE-with-logits.
    Expect `preds`: (N, T) logits and target: {"time": int bin, "event": {0,1}}.
    """
    def __init__(self):
        super().__init__("survival_discrete")
        self.impl = _PyCoxLogHazNLL()
        if self.impl is None:
            self.bcel = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds: torch.Tensor, target: dict[str, torch.Tensor], **_: Any) -> torch.Tensor:
        if self.impl is not None:
            t = target["time"].long()
            e = target["event"].float()
            return self.impl(preds, t, e)