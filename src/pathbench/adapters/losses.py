"""Backend-specific loss adapters and registry helpers.

This module contains concrete PathBench loss adapters backed by optional or
framework-provided dependencies. The domain layer should depend only on
``BaseLoss`` and the registry interface, while concrete implementations are
registered from this adapter layer.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from pathbench.core.losses.base import BaseLoss

ClassificationLossCtor = Callable[..., nn.Module]
TorchTensorLossFn = Callable[..., torch.Tensor]

_TORCH_CLASSIFICATION_LOSSES: dict[str, tuple[ClassificationLossCtor, str, bool]] = {
    "BCELoss": (nn.BCELoss, "float", True),
    "BCEWithLogitsLoss": (nn.BCEWithLogitsLoss, "float", True),
    "CrossEntropyLoss": (nn.CrossEntropyLoss, "long", False),
    "NLLLoss": (nn.NLLLoss, "long", False),
}

_TORCH_REGRESSION_LOSSES: dict[str, ClassificationLossCtor] = {
    "HuberLoss": nn.HuberLoss,
    "L1Loss": nn.L1Loss,
    "MSELoss": nn.MSELoss,
    "SmoothL1Loss": nn.SmoothL1Loss,
}


class TorchClassificationLoss(BaseLoss):
    """Classification loss adapter backed by one ``torch.nn`` loss module.

    Args:
        loss_ctor: ``torch.nn`` loss constructor.
        target_dtype: Target dtype expected by the wrapped loss. Use ``"long"``
            for class-index targets and ``"float"`` for binary/soft targets.
        squeeze_binary: Whether to squeeze ``[B, 1]`` logits and targets to
            ``[B]`` before evaluation.
        **loss_kwargs: Forwarded to ``loss_ctor``.

    Example:
        ```python
        loss = TorchClassificationLoss(nn.CrossEntropyLoss, target_dtype="long")
        value = loss(torch.zeros(2, 3), torch.tensor([0, 1]))
        assert value.ndim == 0
        ```
    """

    def __init__(
        self,
        loss_ctor: ClassificationLossCtor,
        *,
        target_dtype: str,
        squeeze_binary: bool = False,
        **loss_kwargs: Any,
    ) -> None:
        super().__init__("classification")
        self.loss = loss_ctor(**loss_kwargs)
        self.impl = self.loss
        self.target_dtype = target_dtype
        self.squeeze_binary = squeeze_binary

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        preds = preds.float()
        target_tensor = target
        if self.squeeze_binary and preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.reshape(-1)
        if (
            self.squeeze_binary
            and target_tensor.ndim == 2
            and target_tensor.shape[1] == 1
        ):
            target_tensor = target_tensor.reshape(-1)
        if self.target_dtype == "long":
            target_tensor = target_tensor.long().reshape(-1)
        else:
            target_tensor = target_tensor.float()
        return self.loss(preds, target_tensor)


class TorchRegressionLoss(BaseLoss):
    """Regression loss adapter backed by one ``torch.nn`` loss module.

    Args:
        loss_ctor: ``torch.nn`` loss constructor.
        **loss_kwargs: Forwarded to ``loss_ctor``.
    """

    def __init__(
        self,
        loss_ctor: ClassificationLossCtor,
        **loss_kwargs: Any,
    ) -> None:
        super().__init__("regression")
        self.loss = loss_ctor(**loss_kwargs)
        self.impl = self.loss

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        preds = preds.float()
        target = target.float()
        return self.loss(preds, target)


class TorchSurvivalLoss(BaseLoss):
    """Survival loss adapter backed by one ``torchsurv.loss`` function.

    Args:
        loss_fn: TorchSurv loss callable.
        task_type: One of ``"survival"`` or ``"survival_discrete"``.
        uses_eval_time: Whether the wrapped loss requires an ``eval_time`` grid.

    Example:
        ```python
        from torchsurv.loss.cox import neg_partial_log_likelihood

        loss = TorchSurvivalLoss(neg_partial_log_likelihood, task_type="survival")
        target = {"time": torch.tensor([1.0]), "event": torch.tensor([1.0])}
        value = loss(torch.tensor([[0.2]]), target)
        assert torch.isfinite(value)
        ```
    """

    def __init__(
        self,
        loss_fn: TorchTensorLossFn,
        *,
        task_type: str,
        uses_eval_time: bool = False,
    ) -> None:
        super().__init__(task_type)
        self.impl = loss_fn
        self.uses_eval_time = uses_eval_time

    def forward(
        self,
        preds: torch.Tensor,
        target: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        if (
            not isinstance(target, dict)
            or "time" not in target
            or "event" not in target
        ):
            raise ValueError(
                "Survival losses expect target to be a dict with 'time' and 'event'."
            )
        event = target["event"].reshape(-1).bool()
        if self.task_type == "survival_discrete":
            preds = preds.float()
            if preds.ndim != 2:
                raise ValueError(
                    "Discrete survival losses expect preds with shape [B, T]."
                )
            time = target["time"].reshape(-1).to(device=preds.device, dtype=torch.float32)
            event = event.to(device=preds.device)
            eval_time = torch.arange(
                preds.shape[1],
                device=preds.device,
                dtype=torch.float32,
            )
            return self.impl(preds, event, time, eval_time=eval_time, **kwargs)

        preds = preds.float()
        time = target["time"].reshape(-1).to(device=preds.device, dtype=torch.float32)
        event = event.to(device=preds.device)
        if self.uses_eval_time:
            raise RuntimeError(
                "Continuous survival adapter received a loss that requires eval_time."
            )
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds.reshape(-1)
        return self.impl(preds, event, time, **kwargs)


def register_builtin_loss_factories(registry: Any) -> None:
    """Populate one registry with supported ``torch.nn`` and ``torchsurv`` losses.

    Args:
        registry: PathBench registry instance used for loss resolution.

    Example:
        ```python
        from pathbench.utils.registry import Registry

        registry = Registry()
        register_builtin_loss_factories(registry)
        assert registry.get("CrossEntropyLoss") is not None
        ```
    """

    for name, (
        loss_ctor,
        target_dtype,
        squeeze_binary,
    ) in _TORCH_CLASSIFICATION_LOSSES.items():
        if registry.is_available(name):
            continue

        def _classification_factory(
            _loss_ctor: ClassificationLossCtor = loss_ctor,
            _target_dtype: str = target_dtype,
            _squeeze_binary: bool = squeeze_binary,
            **loss_kwargs: Any,
        ) -> TorchClassificationLoss:
            return TorchClassificationLoss(
                _loss_ctor,
                target_dtype=_target_dtype,
                squeeze_binary=_squeeze_binary,
                **loss_kwargs,
            )

        registry.register(name)(_classification_factory)

    for name, loss_ctor in _TORCH_REGRESSION_LOSSES.items():
        if registry.is_available(name):
            continue

        def _regression_factory(
            _loss_ctor: ClassificationLossCtor = loss_ctor,
            **loss_kwargs: Any,
        ) -> TorchRegressionLoss:
            return TorchRegressionLoss(_loss_ctor, **loss_kwargs)

        registry.register(name)(_regression_factory)

    try:
        neg_partial_log_likelihood = getattr(
            importlib.import_module("torchsurv.loss.cox"),
            "neg_partial_log_likelihood",
        )
        neg_log_likelihood = getattr(
            importlib.import_module("torchsurv.loss.survival"),
            "neg_log_likelihood",
        )
        neg_log_likelihood_weibull = getattr(
            importlib.import_module("torchsurv.loss.weibull"),
            "neg_log_likelihood_weibull",
        )
    except Exception:
        return

    survival_specs: dict[str, tuple[TorchTensorLossFn, str, bool]] = {
        "CoxPHLoss": (neg_partial_log_likelihood, "survival", False),
        "neg_partial_log_likelihood": (neg_partial_log_likelihood, "survival", False),
        "DiscreteTimeNLLLoss": (neg_log_likelihood, "survival_discrete", True),
        "neg_log_likelihood": (neg_log_likelihood, "survival_discrete", True),
        "neg_log_likelihood_weibull": (
            neg_log_likelihood_weibull,
            "survival",
            False,
        ),
    }
    for name, (loss_fn, task_type, uses_eval_time) in survival_specs.items():
        if registry.is_available(name):
            continue

        def _survival_factory(
            _loss_fn: TorchTensorLossFn = loss_fn,
            _task_type: str = task_type,
            _uses_eval_time: bool = uses_eval_time,
        ) -> TorchSurvivalLoss:
            return TorchSurvivalLoss(
                _loss_fn,
                task_type=_task_type,
                uses_eval_time=_uses_eval_time,
            )

        registry.register(name)(_survival_factory)


__all__ = [
    "TorchClassificationLoss",
    "TorchRegressionLoss",
    "TorchSurvivalLoss",
    "register_builtin_loss_factories",
]
