import torch
import pytest
from pathbench.core.losses.impl import (
    CrossEntropyLoss,
    MSELoss,
    CoxPHLoss,
    DiscreteTimeNLLLoss,
)


def test_cross_entropy():
    loss = CrossEntropyLoss()
    logits = torch.randn(4, 3)
    y = torch.tensor([0, 1, 2, 1])
    v = loss(logits, y)
    assert v.dim() == 0


def test_mse():
    loss = MSELoss()
    x = torch.randn(8, 1)
    y = torch.randn(8, 1)
    v = loss(x, y)
    assert v.item() >= 0


def test_coxph_shapes():
    loss = CoxPHLoss()
    if loss.impl is None:
        pytest.skip("pycox is not installed in this environment")
    z = torch.randn(10, 1)
    target = {
        "time": torch.linspace(1, 10, 10),
        "event": torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1]),
    }
    v = loss(z, target)
    assert torch.isfinite(v)


def test_discrete_time_nll():
    loss = DiscreteTimeNLLLoss()
    if loss.impl is None:
        pytest.skip("pycox logistic hazard loss is not installed in this environment")
    logits = torch.randn(5, 4)
    target = {
        "time": torch.tensor([0, 1, 2, 3, 1]),
        "event": torch.tensor([1, 0, 1, 0, 1]).float(),
    }
    v = loss(logits, target)
    assert v >= 0
