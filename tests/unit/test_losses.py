import torch
from pathbench.utils.registries import LOSSES


def test_cross_entropy():
    loss = LOSSES.get("CrossEntropyLoss")()
    logits = torch.randn(4, 3)
    y = torch.tensor([0, 1, 2, 1])
    v = loss(logits, y)
    assert v.dim() == 0


def test_mse():
    loss = LOSSES.get("MSELoss")()
    x = torch.randn(8, 1)
    y = torch.randn(8, 1)
    v = loss(x, y)
    assert v.item() >= 0


def test_bce_with_logits_binary_classification() -> None:
    loss = LOSSES.get("BCEWithLogitsLoss")()
    logits = torch.randn(6, 1)
    target = torch.randint(0, 2, (6,), dtype=torch.long)
    value = loss(logits, target)
    assert value.dim() == 0


def test_coxph_shapes():
    loss = LOSSES.get("CoxPHLoss")()
    z = torch.randn(10, 1)
    target = {
        "time": torch.linspace(1, 10, 10),
        "event": torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1]),
    }
    v = loss(z, target)
    assert torch.isfinite(v)


def test_discrete_time_nll():
    loss = LOSSES.get("DiscreteTimeNLLLoss")()
    logits = torch.randn(5, 4)
    target = {
        "time": torch.tensor([0, 1, 2, 3, 1]),
        "event": torch.tensor([1, 0, 1, 0, 1]).float(),
    }
    v = loss(logits, target)
    assert v >= 0


def test_registry_exposes_torchsurv_loss_functions() -> None:
    for name in (
        "neg_partial_log_likelihood",
        "neg_log_likelihood",
        "neg_log_likelihood_weibull",
    ):
        loss = LOSSES.get(name)()
        assert loss is not None
