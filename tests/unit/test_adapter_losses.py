from __future__ import annotations

import torch
import torch.nn as nn

from pathforge.adapters.losses import (
    TorchClassificationLoss,
    TorchRegressionLoss,
    TorchSurvivalLoss,
)
from pathforge.utils.registry import Registry


def test_torch_classification_loss_squeezes_binary_logits_and_targets() -> None:
    loss = TorchClassificationLoss(
        nn.BCEWithLogitsLoss,
        target_dtype="float",
        squeeze_binary=True,
    )

    logits = torch.tensor([[0.2], [-0.1]], dtype=torch.float32)
    target = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    value = loss(logits, target)

    assert value.ndim == 0
    assert torch.isfinite(value)


def test_torch_classification_loss_casts_multiclass_targets_to_long() -> None:
    loss = TorchClassificationLoss(nn.CrossEntropyLoss, target_dtype="long")

    logits = torch.randn(3, 4, dtype=torch.float32)
    target = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float32)

    value = loss(logits, target)

    assert value.ndim == 0
    assert torch.isfinite(value)


def test_torch_regression_loss_casts_target_to_float() -> None:
    loss = TorchRegressionLoss(nn.MSELoss)

    preds = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    target = torch.tensor([[0], [2]], dtype=torch.long)

    value = loss(preds, target)

    assert value.ndim == 0
    assert float(value) >= 0.0


def test_torch_survival_loss_requires_time_and_event_keys() -> None:
    loss = TorchSurvivalLoss(lambda preds, event, time: preds.sum(), task_type="survival")

    with torch.no_grad():
        try:
            loss(torch.tensor([[0.2]], dtype=torch.float32), {"time": torch.tensor([1.0])})
        except ValueError as exc:
            assert "time" in str(exc) and "event" in str(exc)
        else:
            raise AssertionError("Expected survival target validation to fail.")


def test_torch_survival_loss_rejects_invalid_discrete_prediction_rank() -> None:
    loss = TorchSurvivalLoss(
        lambda preds, event, time, eval_time=None: preds.sum(),
        task_type="survival_discrete",
        uses_eval_time=True,
    )

    try:
        loss(
            torch.tensor([0.2, 0.3], dtype=torch.float32),
            {
                "time": torch.tensor([0.0, 1.0], dtype=torch.float32),
                "event": torch.tensor([1.0, 0.0], dtype=torch.float32),
            },
        )
    except ValueError as exc:
        assert "[B, T]" in str(exc)
    else:
        raise AssertionError("Expected discrete survival rank validation to fail.")


def test_torch_survival_loss_passes_eval_time_for_discrete_losses() -> None:
    captured: dict[str, torch.Tensor] = {}

    def _loss_fn(
        preds: torch.Tensor,
        event: torch.Tensor,
        time: torch.Tensor,
        *,
        eval_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert eval_time is not None
        captured["preds"] = preds
        captured["event"] = event
        captured["time"] = time
        captured["eval_time"] = eval_time
        return preds.sum()

    loss = TorchSurvivalLoss(
        _loss_fn,
        task_type="survival_discrete",
        uses_eval_time=True,
    )

    preds = torch.randn(2, 4, dtype=torch.float32)
    target = {
        "time": torch.tensor([0, 2], dtype=torch.long),
        "event": torch.tensor([1.0, 0.0], dtype=torch.float32),
    }

    value = loss(preds, target)

    assert value.ndim == 0
    assert captured["preds"].shape == (2, 4)
    assert captured["event"].dtype == torch.bool
    assert captured["time"].dtype == torch.float32
    assert captured["eval_time"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_register_builtin_loss_factories_populates_common_loss_names() -> None:
    registry = Registry()

    from pathforge.adapters.losses import register_builtin_loss_factories

    register_builtin_loss_factories(registry)

    assert registry.get("CrossEntropyLoss") is not None
    assert registry.get("MSELoss") is not None
