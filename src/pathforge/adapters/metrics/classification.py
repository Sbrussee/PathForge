from __future__ import annotations

from typing import Protocol

import torch

from pathforge.utils.optional.torchmil import require_torchmetrics


class ClassificationMetricsBackend(Protocol):
    """Backend protocol for classification metrics.

    Args:
        logits: Tensor shaped ``[B, C]`` or binary logits shaped ``[B]``.
        target: Integer class labels shaped ``[B]``.

    Returns:
        Dictionary of scalar finite metric tensors.
    """

    def compute(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        ...


class TorchMetricsClassificationBackend:
    """TorchMetrics-backed classification metrics adapter.

    Example:
        ```python
        backend = TorchMetricsClassificationBackend(num_classes=2)
        metrics = backend.compute(torch.zeros(4, 2), torch.tensor([0, 1, 0, 1]))
        assert "accuracy" in metrics
        ```
    """

    def __init__(self, *, num_classes: int, task: str | None = None) -> None:
        require_torchmetrics()
        self.num_classes = num_classes
        self.task = task or ("binary" if num_classes == 2 else "multiclass")

    def compute(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        require_torchmetrics()
        from torchmetrics.functional.classification import accuracy, auroc, f1_score

        assert logits.ndim in {1, 2}, "Classification logits must be shaped [B] or [B, C]."
        target = target.long().reshape(-1)
        preds = logits.reshape(-1) if logits.ndim == 1 else logits
        metrics = {
            "accuracy": accuracy(preds, target, task=self.task, num_classes=self.num_classes),
            "f1": f1_score(preds, target, task=self.task, num_classes=self.num_classes),
        }
        try:
            metrics["auroc"] = auroc(preds, target, task=self.task, num_classes=self.num_classes)
        except ValueError:
            pass
        return metrics
