from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base import TaskLoss

try:
    from pycox.models.loss import CoxPHLoss as _PyCoxPHLoss
    from pycox.models.loss import NLLLogisticHazard as _PyCoxLogHazNLL
except Exception:  # pragma: no cover - optional dependency
    _PyCoxPHLoss = None
    _PyCoxLogHazNLL = None


class CrossEntropyLoss(TaskLoss):
    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__("classification")
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, preds: torch.Tensor, target: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.loss(preds, target)


class MSELoss(TaskLoss):
    def __init__(self):
        super().__init__("regression")
        self.loss = nn.MSELoss()

    def forward(self, preds: torch.Tensor, target: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.loss(preds, target)


class CoxPHLoss(TaskLoss):
    """Cox partial negative log-likelihood with optional pycox backend."""

    def __init__(self, eps: float = 1e-7):
        super().__init__("survival")
        self.eps = float(eps)
        self.impl = _PyCoxPHLoss() if _PyCoxPHLoss is not None else None

    def forward(self, preds: torch.Tensor, target: dict[str, torch.Tensor], **_: Any) -> torch.Tensor:
        scores = preds.reshape(-1).float()
        time = target["time"].reshape(-1).float()
        event = target["event"].reshape(-1).float()

        if self.impl is not None:
            return self.impl(scores, time, event)

        order = torch.argsort(time, descending=True)
        sorted_scores = scores[order]
        sorted_events = event[order]
        log_cumsum_exp = torch.logcumsumexp(sorted_scores, dim=0)
        partial = sorted_scores - log_cumsum_exp
        observed = partial * sorted_events
        denom = sorted_events.sum().clamp_min(self.eps)
        return -(observed.sum() / denom)


class DiscreteTimeNLLLoss(TaskLoss):
    """NLL for logistic hazard with optional pycox backend."""

    def __init__(self):
        super().__init__("survival_discrete")
        self.impl = _PyCoxLogHazNLL() if _PyCoxLogHazNLL is not None else None

    def forward(self, preds: torch.Tensor, target: dict[str, torch.Tensor], **_: Any) -> torch.Tensor:
        logits = preds.float()
        time = target["time"].long()
        event = target["event"].float()

        if self.impl is not None:
            return self.impl(logits, time, event)

        n, t_max = logits.shape
        time = time.clamp_min(0).clamp_max(t_max - 1)

        hazard = torch.sigmoid(logits)
        surv_prefix = torch.cumprod(1.0 - hazard + 1e-7, dim=1)
        prev_surv = torch.ones((n,), device=logits.device, dtype=logits.dtype)
        idx = time > 0
        prev_surv[idx] = surv_prefix[idx, time[idx] - 1]
        h_t = hazard[torch.arange(n, device=logits.device), time]

        likelihood = torch.where(
            event > 0.5,
            prev_surv * h_t,
            prev_surv * (1.0 - h_t),
        )
        return -torch.log(likelihood.clamp_min(1e-7)).mean()
