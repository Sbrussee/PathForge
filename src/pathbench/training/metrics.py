from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch
from sklearn import metrics as sk_metrics

#TODO: Support any metric in torchmetrics / pycox?
def evaluate_predictions(
    logits: torch.Tensor,
    labels: np.ndarray,
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
    raise ValueError(f"Unsupported task for evaluation: {task}")


def _evaluate_classification(
    logits: torch.Tensor,
    labels: np.ndarray,
    metrics: Iterable[str],
    average: str,
    positive_label: int,
) -> Dict[str, float]:
    logits_np = logits.detach().cpu().numpy()
    labels_np = labels.astype(int)

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
            raise ValueError(f"Unsupported classification metric: {metric}")

    return results


def _evaluate_regression(
    logits: torch.Tensor,
    labels: np.ndarray,
    metrics: Iterable[str],
) -> Dict[str, float]:
    preds = logits.detach().cpu().numpy().reshape(-1)
    labels_np = labels.astype(float).reshape(-1)

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


#TODO: Discrete Survival Metrics and Continous Survival Metrics