from __future__ import annotations

from typing import Protocol

import torch

from pathforge.utils.optional.torchmil import require_torchsurv


class SurvivalMetricsBackend(Protocol):
    """Backend protocol for continuous survival metrics.

    Args:
        risk: Finite risk/log-hazard tensor shaped ``[B]``.
        time: Event/censoring time tensor shaped ``[B]`` in study time units.
        event: Binary event indicator shaped ``[B]`` where one means observed
            event and zero means censored.
    """

    def concordance_index(
        self,
        risk: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        ...


class SurvivalLossBackend(Protocol):
    """Backend protocol for continuous survival losses."""

    def neg_partial_log_likelihood(
        self,
        risk: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        ...


class TorchSurvBackend:
    """TorchSurv-backed survival metrics and losses adapter.

    Example:
        .. code-block:: python

            backend = TorchSurvBackend()
            risk = torch.tensor([0.1, -0.2])
            time = torch.tensor([5.0, 8.0])
            event = torch.tensor([1.0, 0.0])
            loss = backend.neg_partial_log_likelihood(risk, time, event)

    """

    def __init__(self) -> None:
        require_torchsurv()

    def concordance_index(
        self,
        risk: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        require_torchsurv()
        from torchsurv.metrics.cindex import ConcordanceIndex

        risk, time, event = _validate_survival_tensors(risk, time, event)
        return ConcordanceIndex()(risk, event.bool(), time)

    def neg_partial_log_likelihood(
        self,
        risk: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        require_torchsurv()
        from torchsurv.loss.cox import neg_partial_log_likelihood

        risk, time, event = _validate_survival_tensors(risk, time, event)
        return neg_partial_log_likelihood(risk, event.bool(), time)


def _validate_survival_tensors(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    risk = risk.float().reshape(-1)
    time = time.float().reshape(-1)
    event = event.float().reshape(-1)
    assert risk.shape == time.shape == event.shape, "Survival tensors must all have shape [B]."
    assert torch.isfinite(risk).all(), "Survival risk contains NaN or Inf."
    assert torch.isfinite(time).all(), "Survival time contains NaN or Inf."
    assert torch.isfinite(event).all(), "Survival event contains NaN or Inf."
    return risk, time, event
