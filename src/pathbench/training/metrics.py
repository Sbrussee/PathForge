from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple
import inspect
import numpy as np
import torch
from sklearn import metrics as sk_metrics

try:
    from pycox.evaluation.concordance import concordance_td
except ImportError:  # pragma: no cover - optional dependency
    concordance_td = None

try:
    import torchmetrics.functional as tmf
except ImportError:  # pragma: no cover - optional dependency
    tmf = None

#TODO: Support any metric in torchmetrics / pycox?
def evaluate_predictions(
    logits: torch.Tensor,
    labels: Any,
    task: str,
    metrics: Iterable[str],
    average: str,
    positive_label: int,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for MIL predictions.

    Args:
        logits: Model outputs as a tensor.
        labels: Ground-truth labels aligned with logits.
        task: Task type (classification or regression).
        metrics: Metric names to compute.
        average: Averaging strategy for multi-class metrics.
        positive_label: Positive label index for binary metrics.
    """
    if task == "classification":
        return _evaluate_classification(logits, labels, metrics, average, positive_label)
    if task == "regression":
        return _evaluate_regression(logits, labels, metrics)
    if task == "survival":
        return _evaluate_survival_continuous(logits, labels, metrics)
    if task == "survival_discrete":
        return _evaluate_survival_discrete(logits, labels, metrics)
    raise ValueError(f"Unsupported task for evaluation: {task}")


def _evaluate_classification(
    logits: torch.Tensor,
    labels: Any,
    metrics: Iterable[str],
    average: str,
    positive_label: int,
) -> Dict[str, float]:
    logits_np = logits.detach().cpu().numpy()
    labels_np = np.asarray(labels, dtype=int)

    if logits_np.ndim == 1 or logits_np.shape[1] == 1:
        probs = 1 / (1 + np.exp(-logits_np.reshape(-1)))
        preds = (probs >= 0.5).astype(int)
    else:
        probs = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        preds = probs.argmax(axis=1)

    results: Dict[str, float] = {}
    for metric in metrics:
        if metric == "accuracy":
            results[metric] = sk_metrics.accuracy_score(labels_np, preds)
        elif metric == "balanced_accuracy":
            results[metric] = sk_metrics.balanced_accuracy_score(labels_np, preds)
        elif metric == "f1":
            results[metric] = sk_metrics.f1_score(labels_np, preds, average=average)
        elif metric == "precision":
            results[metric] = sk_metrics.precision_score(labels_np, preds, average=average)
        elif metric == "recall":
            results[metric] = sk_metrics.recall_score(labels_np, preds, average=average)
        elif metric == "roc_auc":
            if probs.ndim == 1 or probs.shape[1] == 1:
                results[metric] = sk_metrics.roc_auc_score(labels_np, probs)
            else:
                results[metric] = sk_metrics.roc_auc_score(
                    labels_np,
                    probs,
                    multi_class="ovr",
                    average=average,
                )
        elif metric == "specificity":
            if np.unique(labels_np).size > 2:
                raise ValueError("Specificity is only defined for binary classification.")
            results[metric] = sk_metrics.recall_score(labels_np, preds, pos_label=0)
        else:
            results[metric] = _evaluate_torchmetrics(
                metric=metric,
                preds=preds,
                probs=probs,
                labels=labels_np,
                average=average,
                positive_label=positive_label,
                task="classification",
            )

    return results


def _evaluate_regression(
    logits: torch.Tensor,
    labels: Any,
    metrics: Iterable[str],
) -> Dict[str, float]:
    preds = logits.detach().cpu().numpy().reshape(-1)
    labels_np = np.asarray(labels, dtype=float).reshape(-1)

    results: Dict[str, float] = {}
    for metric in metrics:
        if metric == "mse":
            results[metric] = sk_metrics.mean_squared_error(labels_np, preds)
        elif metric == "mae":
            results[metric] = sk_metrics.mean_absolute_error(labels_np, preds)
        elif metric == "r2":
            results[metric] = sk_metrics.r2_score(labels_np, preds)
        else:
            raise ValueError(f"Unsupported regression metric: {metric}")

    return results


def _evaluate_survival_continuous(
    logits: torch.Tensor,
    labels: Any,
    metrics: Iterable[str],
) -> Dict[str, float]:
    time, event = _extract_survival_labels(labels)
    risk = logits.detach().cpu().numpy().reshape(-1)
    surv, time_grid = _exponential_survival(time, risk)
    surv_idx = _survival_index(time, time_grid)

    results: Dict[str, float] = {}
    for metric in metrics:
        if metric == "c_index":
            results[metric] = _concordance_td(time, event, surv, surv_idx)
        else:
            raise ValueError(f"Unsupported survival metric: {metric}")
    return results


def _evaluate_survival_discrete(
    logits: torch.Tensor,
    labels: Any,
    metrics: Iterable[str],
) -> Dict[str, float]:
    time, event = _extract_survival_labels(labels)
    logits_np = logits.detach().cpu().numpy()
    hazard = 1 / (1 + np.exp(-logits_np))
    surv = np.cumprod(1 - hazard, axis=1).T
    time_grid = np.arange(surv.shape[0], dtype=float)
    surv_idx = _survival_index(time, time_grid)

    results: Dict[str, float] = {}
    for metric in metrics:
        if metric == "c_index":
            results[metric] = _concordance_td(time, event, surv, surv_idx)
        else:
            raise ValueError(f"Unsupported survival metric: {metric}")
    return results


def _extract_survival_labels(labels: Any) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(labels, dict):
        time = np.asarray(labels["time"], dtype=float)
        event = np.asarray(labels["event"], dtype=float)
        return time, event
    if isinstance(labels, (list, tuple)) and labels and isinstance(labels[0], dict):
        time = np.asarray([item["time"] for item in labels], dtype=float)
        event = np.asarray([item["event"] for item in labels], dtype=float)
        return time, event
    if isinstance(labels, np.ndarray) and labels.dtype.names:
        return labels["time"].astype(float), labels["event"].astype(float)
    raise ValueError("Survival labels must be dicts or structured arrays with 'time' and 'event'.")


def _concordance_td(
    time: np.ndarray, event: np.ndarray, surv: np.ndarray, surv_idx: np.ndarray
) -> float:
    if concordance_td is None:  # pragma: no cover - optional dependency
        raise RuntimeError("pycox is required for c_index evaluation.")
    return float(concordance_td(time, event, surv, surv_idx))


def _survival_index(time: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(time_grid, time, side="right") - 1
    return np.clip(idx, 0, len(time_grid) - 1).astype(int)


def _exponential_survival(time: np.ndarray, risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build exponential survival curves using risk scores as log hazards.

    This provides a lightweight survival curve proxy for concordance metrics.
    """
    time_grid = np.sort(np.unique(time))
    hazard = np.exp(risk)
    surv = np.exp(-time_grid[:, None] * hazard[None, :])
    return surv, time_grid