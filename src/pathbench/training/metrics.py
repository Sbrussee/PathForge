from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

from pathbench.utils.optional.torchmil import is_torchsurv_available


@dataclass(frozen=True)
class TaskEvaluationArtifacts:
    """Persisted evaluation outputs for one trained model.

    Attributes:
        metrics_path: JSON file containing scalar task metrics.
        curve_data_path: JSON file containing inspectable curve coordinates.
        figure_paths: Mapping from figure name to the saved PNG path.
    """

    metrics_path: Path
    curve_data_path: Path
    figure_paths: dict[str, Path]


def compute_task_metrics(
    predictions: torch.Tensor,
    target: Any,
    *,
    task: str,
    selected_metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute scalar validation metrics for one PathBench task.

    Args:
        predictions: Model output tensor. Classification expects logits shaped
            ``[B]`` or ``[B, C]``. Continuous survival expects risk/log-hazard
            shaped ``[B]`` or ``[B, 1]``. Discrete survival expects logits
            shaped ``[B, T]``.
        target: Task target. Classification expects integer labels shaped
            ``[B]``. Survival tasks expect a dictionary with ``time`` and
            ``event`` tensors shaped ``[B]``.
        task: PathBench task name.

    Returns:
        dict[str, float]: Finite-or-NaN scalar metrics ready for Lightning
        logging and JSON serialization.
    """

    if task == "classification":
        metrics = _compute_classification_metrics(predictions, target)
    elif task in {"survival", "survival_discrete"}:
        metrics = _compute_survival_metrics(predictions, target, task=task)
    elif task == "regression":
        metrics = _compute_regression_metrics(predictions, target)
    else:
        metrics = {}
    if selected_metrics is None:
        return metrics
    selected_names = set(selected_metrics)
    return {name: value for name, value in metrics.items() if name in selected_names}


def save_task_evaluation_artifacts(
    predictions: torch.Tensor,
    target: Any,
    *,
    task: str,
    output_dir: Path,
    prefix: str,
    selected_metrics: list[str] | None = None,
) -> TaskEvaluationArtifacts:
    """Write scalar metrics plus task-specific visualization artifacts.

    Args:
        predictions: Task prediction tensor with the same shape contract as
            :func:`compute_task_metrics`.
        target: Task target object with the same contract as
            :func:`compute_task_metrics`.
        task: PathBench task name.
        output_dir: Directory receiving JSON and PNG artifacts.
        prefix: Output prefix such as ``"val"``.

    Returns:
        TaskEvaluationArtifacts: Paths to the persisted JSON and PNG files.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_task_metrics(
        predictions,
        target,
        task=task,
        selected_metrics=selected_metrics,
    )
    curve_payload: dict[str, Any]
    figure_paths: dict[str, Path]

    if task == "classification":
        curve_payload, figure_paths = _save_classification_artifacts(
            predictions,
            target,
            output_dir=output_dir,
            prefix=prefix,
        )
    elif task in {"survival", "survival_discrete"}:
        curve_payload, figure_paths = _save_survival_artifacts(
            predictions,
            target,
            task=task,
            output_dir=output_dir,
            prefix=prefix,
        )
    elif task == "regression":
        curve_payload, figure_paths = _save_regression_artifacts(
            predictions,
            target,
            output_dir=output_dir,
            prefix=prefix,
        )
    else:
        curve_payload = {}
        figure_paths = {}

    metrics_path = output_dir / f"{prefix}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    curve_data_path = output_dir / f"{prefix}_curves.json"
    curve_data_path.write_text(
        json.dumps(curve_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return TaskEvaluationArtifacts(
        metrics_path=metrics_path,
        curve_data_path=curve_data_path,
        figure_paths=figure_paths,
    )


def _compute_classification_metrics(
    predictions: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    y_true, probabilities, y_pred = _classification_arrays(predictions, target)
    average = "binary" if probabilities.shape[1] == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    try:
        if probabilities.shape[1] == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, probabilities[:, 1]))
            metrics["pr_auc"] = float(
                average_precision_score(y_true, probabilities[:, 1])
            )
            metrics["brier_score"] = float(
                brier_score_loss(y_true, probabilities[:, 1])
            )
        else:
            metrics["auroc"] = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )
            )
            binarized = label_binarize(y_true, classes=np.unique(y_true))
            if binarized.ndim == 1:
                binarized = binarized.reshape(-1, 1)
            metrics["pr_auc"] = float(
                average_precision_score(
                    binarized,
                    probabilities[:, : binarized.shape[1]],
                    average="macro",
                )
            )
    except ValueError:
        metrics["auroc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        if probabilities.shape[1] == 2:
            metrics["brier_score"] = float("nan")
    return metrics


def _save_classification_artifacts(
    predictions: torch.Tensor,
    target: torch.Tensor,
    *,
    output_dir: Path,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    plt = _load_matplotlib_pyplot()
    y_true, probabilities, y_pred = _classification_arrays(predictions, target)
    labels = np.unique(y_true)

    confusion_path = output_dir / f"{prefix}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)), labels=[str(label) for label in labels])
    ax.set_yticks(range(len(labels)), labels=[str(label) for label in labels])
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                str(int(matrix[row_index, col_index])),
                ha="center",
                va="center",
                color="black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(confusion_path, dpi=150)
    plt.close(fig)

    roc_path = output_dir / f"{prefix}_roc_auc_curve.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    curve_payload: dict[str, Any] = {
        "confusion_matrix": matrix.tolist(),
        "labels": [int(label) for label in labels],
        "roc_curves": {},
        "pr_curves": {},
        "calibration_curves": {},
    }
    if probabilities.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_true, probabilities[:, 1])
        roc_auc_value = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_value:.3f}")
        curve_payload["roc_curves"]["positive_class"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc_value),
        }
    else:
        binarized = label_binarize(y_true, classes=labels)
        for class_index, class_label in enumerate(labels):
            try:
                fpr, tpr, _ = roc_curve(
                    binarized[:, class_index], probabilities[:, class_index]
                )
                roc_auc_value = auc(fpr, tpr)
            except ValueError:
                continue
            ax.plot(fpr, tpr, label=f"Class {class_label} AUC = {roc_auc_value:.3f}")
            curve_payload["roc_curves"][str(int(class_label))] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(roc_auc_value),
            }
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="grey", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC-AUC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    pr_path = output_dir / f"{prefix}_pr_auc_curve.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    if probabilities.shape[1] == 2:
        precision, recall, _ = precision_recall_curve(y_true, probabilities[:, 1])
        pr_auc_value = auc(recall, precision)
        ax.plot(recall, precision, label=f"AP = {pr_auc_value:.3f}")
        curve_payload["pr_curves"]["positive_class"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "auc": float(pr_auc_value),
        }
    else:
        binarized = label_binarize(y_true, classes=labels)
        if binarized.ndim == 1:
            binarized = binarized.reshape(-1, 1)
        for class_index, class_label in enumerate(labels[: binarized.shape[1]]):
            try:
                precision, recall, _ = precision_recall_curve(
                    binarized[:, class_index],
                    probabilities[:, class_index],
                )
                pr_auc_value = auc(recall, precision)
            except ValueError:
                continue
            ax.plot(
                recall, precision, label=f"Class {class_label} AP = {pr_auc_value:.3f}"
            )
            curve_payload["pr_curves"][str(int(class_label))] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "auc": float(pr_auc_value),
            }
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR-AUC Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)

    calibration_path = output_dir / f"{prefix}_calibration_curve.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    if probabilities.shape[1] == 2:
        prob_true, prob_pred = calibration_curve(y_true, probabilities[:, 1], n_bins=5)
        ax.plot(prob_pred, prob_true, marker="o", label="Positive class")
        curve_payload["calibration_curves"]["positive_class"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
    else:
        binarized = label_binarize(y_true, classes=labels)
        if binarized.ndim == 1:
            binarized = binarized.reshape(-1, 1)
        for class_index, class_label in enumerate(labels[: binarized.shape[1]]):
            try:
                prob_true, prob_pred = calibration_curve(
                    binarized[:, class_index],
                    probabilities[:, class_index],
                    n_bins=5,
                )
            except ValueError:
                continue
            ax.plot(prob_pred, prob_true, marker="o", label=f"Class {class_label}")
            curve_payload["calibration_curves"][str(int(class_label))] = {
                "prob_true": prob_true.tolist(),
                "prob_pred": prob_pred.tolist(),
            }
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="grey", linewidth=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration Curve")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(calibration_path, dpi=150)
    plt.close(fig)

    return curve_payload, {
        "confusion_matrix": confusion_path,
        "roc_auc_curve": roc_path,
        "pr_auc_curve": pr_path,
        "calibration_curve": calibration_path,
    }


def _classification_arrays(
    predictions: torch.Tensor,
    target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = predictions.detach().cpu().float()
    y_true = target.detach().cpu().long().reshape(-1).numpy()
    if logits.ndim == 1:
        positive_probability = torch.sigmoid(logits.reshape(-1)).numpy()
        probabilities = np.stack(
            [1.0 - positive_probability, positive_probability], axis=1
        )
        y_pred = (positive_probability >= 0.5).astype(np.int64)
    else:
        assert logits.ndim == 2, "Classification logits must have shape [B] or [B, C]."
        if logits.shape[1] == 1:
            positive_probability = torch.sigmoid(logits.reshape(-1)).numpy()
            probabilities = np.stack(
                [1.0 - positive_probability, positive_probability],
                axis=1,
            )
            y_pred = (positive_probability >= 0.5).astype(np.int64)
        else:
            probabilities = torch.softmax(logits, dim=1).numpy()
            y_pred = probabilities.argmax(axis=1).astype(np.int64)
    return y_true, probabilities, y_pred


def _compute_survival_metrics(
    predictions: torch.Tensor,
    target: Any,
    *,
    task: str,
) -> dict[str, float]:
    time, event = _survival_target_arrays(target)
    risk = _survival_risk_scores(predictions, task=task, time=time, target=target)
    eval_times, td_auc = _time_dependent_auc_curve(
        predictions,
        time=time,
        event=event,
        task=task,
        target=target,
    )
    brier_times, brier_scores = _survival_brier_score_curve(
        predictions,
        time=time,
        event=event,
        task=task,
    )
    finite_auc = td_auc[np.isfinite(td_auc)]
    finite_brier = brier_scores[np.isfinite(brier_scores)]
    return {
        "brier_score": float(np.nanmean(finite_brier))
        if finite_brier.size
        else float("nan"),
        "c_index": float(_concordance_index(risk, time, event)),
        "td_auc": float(np.nanmean(finite_auc)) if finite_auc.size else float("nan"),
        "td_auc_mean": float(np.nanmean(finite_auc))
        if finite_auc.size
        else float("nan"),
        "td_auc_max": float(np.nanmax(finite_auc)) if finite_auc.size else float("nan"),
        "td_auc_min": float(np.nanmin(finite_auc)) if finite_auc.size else float("nan"),
        "num_eval_times": float(max(len(eval_times), len(brier_times))),
    }


def _compute_regression_metrics(
    predictions: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    y_pred, y_true = _regression_arrays(predictions, target)
    mse_value = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse_value,
        "rmse": float(np.sqrt(mse_value)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _save_regression_artifacts(
    predictions: torch.Tensor,
    target: torch.Tensor,
    *,
    output_dir: Path,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    plt = _load_matplotlib_pyplot()
    y_pred, y_true = _regression_arrays(predictions, target)
    residuals = y_pred - y_true

    scatter_path = output_dir / f"{prefix}_regression_scatter.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.8)
    diag_min = float(min(np.min(y_true), np.min(y_pred)))
    diag_max = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot(
        [diag_min, diag_max],
        [diag_min, diag_max],
        linestyle="--",
        color="grey",
        linewidth=1,
    )
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title("Regression Scatter")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    residuals_path = output_dir / f"{prefix}_residuals.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(
        residuals, bins=min(10, max(3, residuals.shape[0])), color="#4477AA", alpha=0.9
    )
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(residuals_path, dpi=150)
    plt.close(fig)

    payload = {
        "true": y_true.tolist(),
        "pred": y_pred.tolist(),
        "residuals": residuals.tolist(),
    }
    return payload, {
        "regression_scatter": scatter_path,
        "residuals": residuals_path,
    }


def _kaplan_meier_curve(
    time: np.ndarray,
    event: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Kaplan-Meier step-function survival estimate.

    Args:
        time: Observed times shaped ``[N]``.
        event: Event indicators shaped ``[N]`` with values in ``{0, 1}``.

    Returns:
        Tuple of ``(times, survival)`` arrays starting at ``(0, 1.0)``.
    """
    times_out: list[float] = [0.0]
    survival_out: list[float] = [1.0]
    current_survival = 1.0
    for t in np.sort(np.unique(time[event > 0.5])):
        at_risk = float(np.sum(time >= t))
        if at_risk <= 0.0:
            continue
        events_at_t = float(np.sum((time == t) & (event > 0.5)))
        current_survival *= 1.0 - events_at_t / at_risk
        times_out.append(float(t))
        survival_out.append(float(current_survival))
    return np.array(times_out, dtype=np.float32), np.array(
        survival_out, dtype=np.float32
    )


def _save_kaplan_meier_plot(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    *,
    output_dir: Path,
    prefix: str,
    x_axis_label: str = "Time",
) -> tuple[Path | None, dict[str, Any]]:
    """Save a Kaplan-Meier plot with high/low risk groups split at the median.

    Args:
        time: Observed times shaped ``[N]``.
        event: Event indicators shaped ``[N]``.
        risk: Risk scores shaped ``[N]``; higher score implies earlier event.
        output_dir: Destination directory.
        prefix: Filename prefix such as ``"val"``.

    Returns:
        Tuple of the saved PNG path (or ``None`` when there are too few events
        to produce a meaningful plot) and a JSON-serializable payload.
    """
    num_events = int(np.sum(event > 0.5))
    if num_events < 2:
        return None, {}

    plt = _load_matplotlib_pyplot()
    median_risk = float(np.median(risk))
    high_mask = risk > median_risk
    low_mask = ~high_mask
    n_high = int(high_mask.sum())
    n_low = int(low_mask.sum())
    if n_high == 0 or n_low == 0:
        return None, {}

    t_high, s_high = _kaplan_meier_curve(time[high_mask], event[high_mask])
    t_low, s_low = _kaplan_meier_curve(time[low_mask], event[low_mask])

    km_path = output_dir / f"{prefix}_kaplan_meier.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(
        t_high, s_high, where="post", color="#CC4444", label=f"High risk (n={n_high})"
    )
    ax.step(t_low, s_low, where="post", color="#4477AA", label=f"Low risk (n={n_low})")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan-Meier Survival Curves (Median Split)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(km_path, dpi=150)
    plt.close(fig)

    payload: dict[str, Any] = {
        "median_risk": median_risk,
        "high_risk": {
            "n": n_high,
            "times": t_high.tolist(),
            "survival": s_high.tolist(),
        },
        "low_risk": {
            "n": n_low,
            "times": t_low.tolist(),
            "survival": s_low.tolist(),
        },
    }
    return km_path, payload


def _save_survival_artifacts(
    predictions: torch.Tensor,
    target: Any,
    *,
    task: str,
    output_dir: Path,
    prefix: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    plt = _load_matplotlib_pyplot()
    time, event = _survival_target_arrays(target)
    km_time = _survival_kaplan_meier_time(target, fallback_time=time, task=task)
    risk = _survival_risk_scores(predictions, task=task, time=time, target=target)
    x_axis_label = _survival_time_axis_label(task)
    km_x_axis_label = "Time"
    eval_times, td_auc = _time_dependent_auc_curve(
        predictions,
        time=time,
        event=event,
        task=task,
        target=target,
    )
    brier_times, brier_scores = _survival_brier_score_curve(
        predictions,
        time=time,
        event=event,
        task=task,
    )
    c_index_value = _concordance_index(risk, time, event)
    c_index_times, c_index_curve = _concordance_index_curve(risk, time, event)

    td_auc_path = output_dir / f"{prefix}_td_auc_curve.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    _plot_survival_curve(ax, eval_times, td_auc, discrete=task == "survival_discrete")
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("TD-AUC")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Time-Dependent AUC")
    fig.tight_layout()
    fig.savefig(td_auc_path, dpi=150)
    plt.close(fig)

    brier_path = output_dir / f"{prefix}_brier_score_curve.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    _plot_survival_curve(
        ax,
        brier_times,
        brier_scores,
        discrete=task == "survival_discrete",
        color="#228833",
    )
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Brier score")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Time-Dependent Brier Score")
    fig.tight_layout()
    fig.savefig(brier_path, dpi=150)
    plt.close(fig)

    c_index_path = output_dir / f"{prefix}_concordance_index.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    c_index_plot_x = _plot_survival_curve(
        ax,
        c_index_times,
        c_index_curve,
        discrete=task == "survival_discrete",
        color="#4477AA",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Concordance index")
    ax.set_title("Concordance Over Time")
    if c_index_times.size:
        ax.text(
            float(c_index_plot_x[-1]),
            float(c_index_curve[-1]),
            f"overall={c_index_value:.3f}",
            ha="right",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(c_index_path, dpi=150)
    plt.close(fig)

    km_path, km_payload = _save_kaplan_meier_plot(
        km_time,
        event,
        risk,
        output_dir=output_dir,
        prefix=prefix,
        x_axis_label=km_x_axis_label,
    )

    payload: dict[str, Any] = {
        "evaluation_times": eval_times.tolist(),
        "td_auc": td_auc.tolist(),
        "brier_times": brier_times.tolist(),
        "brier_score_curve": brier_scores.tolist(),
        "brier_score": float(np.nanmean(brier_scores[np.isfinite(brier_scores)]))
        if np.isfinite(brier_scores).any()
        else float("nan"),
        "c_index": float(c_index_value),
        "c_index_times": c_index_times.tolist(),
        "c_index_curve": c_index_curve.tolist(),
        "kaplan_meier": km_payload,
    }
    figure_paths: dict[str, Path] = {
        "brier_score_curve": brier_path,
        "td_auc_curve": td_auc_path,
        "concordance_index": c_index_path,
    }
    if km_path is not None:
        figure_paths["kaplan_meier"] = km_path
    return payload, figure_paths


def _survival_time_axis_label(task: str) -> str:
    return "Time bins" if task == "survival_discrete" else "Evaluation time"


def _plot_survival_curve(
    ax: Any,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    discrete: bool,
    color: str | None = None,
) -> np.ndarray:
    if not discrete:
        ax.plot(x_values, y_values, marker="o", color=color)
        return x_values
    positions = np.arange(len(x_values), dtype=np.float32)
    ax.plot(positions, y_values, marker="o", color=color)
    ax.set_xticks(
        positions,
        labels=[_format_discrete_time_bin_label(value) for value in x_values],
    )
    return positions


def _format_discrete_time_bin_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


def _survival_target_arrays(target: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(target, dict) or "time" not in target or "event" not in target:
        raise ValueError(
            "Survival metrics require target dicts with 'time' and 'event'."
        )
    time = target["time"].detach().cpu().float().reshape(-1).numpy()
    event = target["event"].detach().cpu().float().reshape(-1).numpy()
    return time, event


def _optional_survival_target_array(target: Any, *keys: str) -> np.ndarray | None:
    if not isinstance(target, dict):
        return None
    for key in keys:
        value = target.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().reshape(-1).numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)
    return None


def _survival_kaplan_meier_time(
    target: Any,
    *,
    fallback_time: np.ndarray,
    task: str,
) -> np.ndarray:
    if task != "survival_discrete":
        return fallback_time
    continuous_time = _optional_survival_target_array(
        target,
        "continuous_time",
        "time_continuous",
        "raw_time",
    )
    if continuous_time is None or continuous_time.shape != fallback_time.shape:
        return fallback_time
    return continuous_time


def _regression_arrays(
    predictions: torch.Tensor,
    target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    pred = predictions.detach().cpu().float()
    truth = target.detach().cpu().float()
    if pred.ndim == 2:
        if pred.shape[1] != 1:
            raise ValueError(
                f"Regression predictions must have shape [B] or [B, 1]. Got {pred.shape}."
            )
        pred = pred.reshape(-1)
    if truth.ndim == 2:
        if truth.shape[1] != 1:
            raise ValueError(
                f"Regression targets must have shape [B] or [B, 1]. Got {truth.shape}."
            )
        truth = truth.reshape(-1)
    return pred.numpy(), truth.reshape(-1).numpy()


def _survival_risk_scores(
    predictions: torch.Tensor,
    *,
    task: str,
    time: np.ndarray,
    target: Any | None = None,
) -> np.ndarray:
    pred = predictions.detach().cpu().float()
    if task == "survival":
        return pred.reshape(-1).numpy()
    assert pred.ndim == 2, "Discrete survival predictions must have shape [B, T]."
    return _discrete_continuous_hazard_scores(
        pred,
        time=time,
        target=target,
    ).sum(axis=1)


def _discrete_continuous_hazard_scores(
    predictions: torch.Tensor,
    *,
    time: np.ndarray,
    target: Any | None = None,
) -> np.ndarray:
    """Map discrete survival bin probabilities to piecewise-constant hazards.

    For bin ``j = [t_j, t_{j+1})``, PathBench discrete survival heads model the
    conditional event probability ``p_j = P(T in bin j | T >= t_j, x)``. The
    continuous-time hazard that is constant over that bin is therefore
    ``lambda_j(x) = -log(1 - p_j) / (t_{j+1} - t_j)``.

    The implementation uses ``np.log1p(-p_j)`` for numerical stability and does
    not use ``p_j / width_j`` except as the small-probability approximation that
    this exact mapping avoids.
    """

    probability = torch.sigmoid(predictions).numpy()
    probability = np.clip(probability, 1.0e-7, 1.0 - 1.0e-7)
    widths = _discrete_survival_bin_widths(
        target,
        time=time,
        num_bins=probability.shape[1],
    )
    return -np.log1p(-probability) / widths.reshape(1, -1)


def _discrete_survival_bin_widths(
    target: Any | None,
    *,
    time: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    explicit_widths = _optional_survival_target_array(
        target,
        "bin_widths",
        "time_bin_widths",
    )
    if explicit_widths is not None:
        return _coerce_discrete_bin_widths(explicit_widths, num_bins=num_bins)

    explicit_edges = _optional_survival_target_array(
        target,
        "bin_edges",
        "time_bin_edges",
    )
    if explicit_edges is not None:
        if explicit_edges.size == num_bins + 1:
            return _coerce_discrete_bin_widths(np.diff(explicit_edges), num_bins=num_bins)
        if explicit_edges.size % (num_bins + 1) == 0:
            repeated_edges = explicit_edges.reshape(-1, num_bins + 1)
            return _coerce_discrete_bin_widths(
                np.diff(repeated_edges[0]),
                num_bins=num_bins,
            )

    continuous_time = _optional_survival_target_array(
        target,
        "continuous_time",
        "time_continuous",
        "raw_time",
    )
    if continuous_time is not None and continuous_time.shape == time.shape:
        inferred = _infer_discrete_bin_widths_from_continuous_time(
            discrete_time=time,
            continuous_time=continuous_time,
            num_bins=num_bins,
        )
        if inferred is not None:
            return inferred

    return np.ones(num_bins, dtype=np.float32)


def _coerce_discrete_bin_widths(widths: np.ndarray, *, num_bins: int) -> np.ndarray:
    values = np.asarray(widths, dtype=np.float32).reshape(-1)
    if values.size == num_bins:
        result = values
    elif values.size % num_bins == 0:
        result = values.reshape(-1, num_bins)[0]
    else:
        raise ValueError(
            f"Discrete survival bin widths must have {num_bins} values. Got {values.size}."
        )
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("Discrete survival bin widths must be finite and positive.")
    return result.astype(np.float32)


def _infer_discrete_bin_widths_from_continuous_time(
    *,
    discrete_time: np.ndarray,
    continuous_time: np.ndarray,
    num_bins: int,
) -> np.ndarray | None:
    discrete_indices = np.rint(discrete_time).astype(int)
    if discrete_indices.size == 0:
        return None

    bin_mins = np.full(num_bins, np.nan, dtype=np.float64)
    bin_maxs = np.full(num_bins, np.nan, dtype=np.float64)
    for bin_index in range(num_bins):
        values = continuous_time[discrete_indices == bin_index]
        values = values[np.isfinite(values)]
        if values.size:
            bin_mins[bin_index] = float(np.min(values))
            bin_maxs[bin_index] = float(np.max(values))
    if np.all(np.isnan(bin_mins)):
        return None

    fallback_width = _positive_fallback_width(continuous_time)
    edges = np.zeros(num_bins + 1, dtype=np.float64)
    first_min = bin_mins[np.isfinite(bin_mins)][0]
    edges[0] = min(0.0, float(first_min))
    for bin_index in range(1, num_bins):
        previous_max = bin_maxs[bin_index - 1]
        current_min = bin_mins[bin_index]
        if np.isfinite(previous_max) and np.isfinite(current_min):
            edges[bin_index] = (previous_max + current_min) / 2.0
        else:
            edges[bin_index] = edges[bin_index - 1] + fallback_width
    last_max = bin_maxs[np.isfinite(bin_maxs)][-1]
    edges[-1] = max(float(last_max), edges[-2] + fallback_width)

    widths = np.diff(edges)
    widths[~np.isfinite(widths) | (widths <= 0.0)] = fallback_width
    return widths.astype(np.float32)


def _positive_fallback_width(values: np.ndarray) -> float:
    finite_values = np.sort(np.unique(values[np.isfinite(values)]))
    diffs = np.diff(finite_values)
    positive_diffs = diffs[diffs > 0.0]
    if positive_diffs.size:
        return float(np.median(positive_diffs))
    return 1.0


def _time_dependent_auc_curve(
    predictions: torch.Tensor,
    *,
    time: np.ndarray,
    event: np.ndarray,
    task: str,
    target: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    eval_times = _select_eval_times(time, event)
    if eval_times.size == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    auc_values: list[float] = []
    if task == "survival":
        risk = predictions.detach().cpu().float().reshape(-1).numpy()
        for tau in eval_times:
            auc_values.append(_td_auc_at_time(risk, time, event, float(tau)))
    else:
        pred = predictions.detach().cpu().float()
        assert pred.ndim == 2, "Discrete survival predictions must have shape [B, T]."
        hazard = _discrete_continuous_hazard_scores(
            pred,
            time=time,
            target=target,
        )
        for tau in eval_times:
            tau_index = int(min(max(round(float(tau)), 0), hazard.shape[1] - 1))
            risk = hazard[:, : tau_index + 1].sum(axis=1)
            auc_values.append(_td_auc_at_time(risk, time, event, float(tau)))
    return eval_times.astype(np.float32), np.asarray(auc_values, dtype=np.float32)


def _survival_brier_score_curve(
    predictions: torch.Tensor,
    *,
    time: np.ndarray,
    event: np.ndarray,
    task: str,
) -> tuple[np.ndarray, np.ndarray]:
    eval_times = _select_eval_times(time, event)
    if eval_times.size == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    if not is_torchsurv_available():
        return eval_times.astype(np.float32), np.full(
            eval_times.shape,
            np.nan,
            dtype=np.float32,
        )

    BrierScore = getattr(
        importlib.import_module("torchsurv.metrics.brier_score"),
        "BrierScore",
    )

    event_tensor = torch.as_tensor(event > 0.5, dtype=torch.bool)
    time_tensor = torch.as_tensor(time, dtype=torch.float32)
    eval_time_tensor = torch.as_tensor(eval_times, dtype=torch.float32)
    estimate = _survival_probability_estimates_for_times(
        predictions,
        task=task,
        time=time_tensor,
        event=event_tensor,
        eval_times=eval_time_tensor,
    )
    scores = BrierScore()(
        estimate, event_tensor, time_tensor, new_time=eval_time_tensor
    )
    return eval_times.astype(np.float32), scores.detach().cpu().float().numpy()


def _survival_probability_estimates_for_times(
    predictions: torch.Tensor,
    *,
    task: str,
    time: torch.Tensor,
    event: torch.Tensor,
    eval_times: torch.Tensor,
) -> torch.Tensor:
    pred = predictions.detach().cpu().float()
    if task == "survival":
        cox_module = importlib.import_module("torchsurv.loss.cox")
        baseline_survival_function = getattr(
            cox_module, "baseline_survival_function"
        )
        survival_function_cox = getattr(cox_module, "survival_function_cox")

        risk = pred.reshape(-1)
        baseline = baseline_survival_function(
            risk,
            event,
            time,
            checks=False,
        )
        return survival_function_cox(
            baseline,
            risk,
            eval_times,
        ).float()

    if pred.ndim != 2:
        raise ValueError(
            "Discrete survival predictions must have shape [B, T] when computing Brier score."
        )
    hazard = torch.sigmoid(pred).clamp_(1.0e-6, 1.0 - 1.0e-6)
    survival = torch.cumprod(1.0 - hazard, dim=1)
    indices = eval_times.round().long().clamp_(0, pred.shape[1] - 1)
    return survival[:, indices].float()


def _select_eval_times(
    time: np.ndarray,
    event: np.ndarray,
    *,
    max_points: int = 12,
) -> np.ndarray:
    observed_event_times = np.unique(time[event > 0.5])
    if observed_event_times.size <= 1:
        return observed_event_times.astype(np.float32)
    if observed_event_times.size <= max_points:
        return observed_event_times.astype(np.float32)
    indices = np.linspace(0, observed_event_times.size - 1, num=max_points)
    return observed_event_times[np.round(indices).astype(int)].astype(np.float32)


def _td_auc_at_time(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    tau: float,
    *,
    min_survival: float = 1.0e-6,
) -> float:
    case_mask = (event > 0.5) & (time <= tau)
    control_mask = time > tau
    if case_mask.sum() == 0 or control_mask.sum() == 0:
        return float("nan")

    censor_survival = _kaplan_meier_censoring_survival(time, event)
    case_times = time[case_mask]
    case_weights = np.asarray(
        [
            1.0 / max(censor_survival(case_time), min_survival)
            for case_time in case_times
        ],
        dtype=np.float64,
    )
    case_scores = risk[case_mask]
    control_scores = risk[control_mask]
    comparisons = (case_scores[:, None] > control_scores[None, :]).astype(np.float64)
    ties = (case_scores[:, None] == control_scores[None, :]).astype(np.float64) * 0.5
    weighted = (comparisons + ties) * case_weights[:, None]
    denominator = case_weights.sum() * float(control_scores.size)
    if denominator <= 0.0:
        return float("nan")
    return float(weighted.sum() / denominator)


def _kaplan_meier_censoring_survival(
    time: np.ndarray,
    event: np.ndarray,
):
    censor_indicator = 1.0 - event
    unique_times = np.unique(time)
    survival_values: list[tuple[float, float]] = []
    survival = 1.0
    for current_time in np.sort(unique_times):
        at_risk = float(np.sum(time >= current_time))
        if at_risk <= 0.0:
            continue
        num_censored = float(np.sum((time == current_time) & (censor_indicator > 0.5)))
        if num_censored > 0.0:
            survival *= 1.0 - (num_censored / at_risk)
        survival_values.append((float(current_time), float(survival)))

    def lookup(query_time: float) -> float:
        result = 1.0
        for current_time, value in survival_values:
            if current_time <= query_time:
                result = value
            else:
                break
        return result

    return lookup


def _concordance_index(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    concordant = 0.0
    comparable = 0.0
    n_samples = len(risk)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if time[i] == time[j]:
                continue
            if time[i] < time[j] and event[i] > 0.5:
                comparable += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
            elif time[j] < time[i] and event[j] > 0.5:
                comparable += 1.0
                if risk[j] > risk[i]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    if comparable == 0.0:
        return float("nan")
    return float(concordant / comparable)


def _concordance_index_curve(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative concordance values at observed event times.

    Args:
        risk: Risk scores shaped ``[N]`` where higher implies earlier event.
        time: Observed time values shaped ``[N]``.
        event: Event indicators shaped ``[N]`` with values in ``{0, 1}``.

    Returns:
        tuple[np.ndarray, np.ndarray]: Evaluation times and concordance values
        shaped ``[T]``. Each point truncates follow-up at the corresponding
        time so the curve remains interpretable over time.
    """

    eval_times = _select_eval_times(time, event)
    if eval_times.size == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    curve_values: list[float] = []
    for tau in eval_times:
        truncated_time = np.minimum(time, float(tau))
        truncated_event = ((event > 0.5) & (time <= float(tau))).astype(np.float32)
        if truncated_event.sum() == 0:
            curve_values.append(float("nan"))
            continue
        curve_values.append(_concordance_index(risk, truncated_time, truncated_event))
    return eval_times.astype(np.float32), np.asarray(curve_values, dtype=np.float32)


def _load_matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt
