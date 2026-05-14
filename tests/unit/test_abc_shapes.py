"""Unit tests for abstract MIL model shape contracts."""

from __future__ import annotations

import torch

from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.models.slide_base import SlideLevelModel


class _ToyMILModel(MILModelBase):
    """Minimal concrete MIL model used to validate the base contract."""

    def __init__(self, input_dim: int = 4, output_dim: int = 2) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    @property
    def bag_size(self) -> int | None:
        return None

    def forward_bag(
        self,
        bag: torch.Tensor,
        mask: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        loss_fn: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        _ = (coords, label, loss_fn)
        if mask is None:
            pooled = bag.mean(dim=1)
        else:
            weights = mask.float().unsqueeze(-1)
            pooled = (bag * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)
        return self.linear(pooled)


class _ToySlideModel(SlideLevelModel):
    """Minimal slide-level model used to validate the shared torch base."""

    def __init__(self, input_dim: int = 4, output_dim: int = 2) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward_slide(
        self,
        x: torch.Tensor,
        label: torch.Tensor | None = None,
        loss_fn: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        _ = (label, loss_fn)
        return self.linear(x)


def test_mil_forward_contract_returns_batched_logits() -> None:
    """Concrete MIL models should accept bags shaped ``[B, N, D]``."""
    model = _ToyMILModel(input_dim=4, output_dim=2)
    bag = torch.zeros((2, 8, 4), dtype=torch.float32)

    out = model(bag)

    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


def test_mil_forward_contract_accepts_mask() -> None:
    """Optional bag masks should preserve the batch/output contract."""
    model = _ToyMILModel(input_dim=4, output_dim=3)
    bag = torch.ones((1, 5, 4), dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)

    out = model.forward_bag(bag, mask=mask)

    assert out.shape == (1, 3)
    assert torch.isfinite(out).all()


def test_slide_forward_contract_uses_torch_module_call_semantics() -> None:
    """Slide-level models should remain callable via the shared torch base."""
    model = _ToySlideModel(input_dim=4, output_dim=2)
    features = torch.ones((3, 4), dtype=torch.float32)

    out = model(features)

    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()


def test_shared_torch_initialize_is_not_a_no_op() -> None:
    """Shared torch-backed model initialization should reset parameters."""
    model = _ToySlideModel(input_dim=4, output_dim=2)
    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(1.0)

    model.initialize()

    assert not torch.allclose(model.linear.weight, torch.ones_like(model.linear.weight))
    assert not torch.allclose(model.linear.bias, torch.ones_like(model.linear.bias))
