from __future__ import annotations

import json
from pathlib import Path

import torch

from pathbench.training.metrics import (
    compute_task_metrics,
    save_task_evaluation_artifacts,
)


def test_classification_metrics_and_artifacts_are_saved(tmp_path: Path) -> None:
    logits = torch.tensor(
        [
            [4.0, -2.0],
            [-3.0, 3.5],
            [1.2, -0.1],
            [-0.2, 1.1],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    metrics = compute_task_metrics(logits, target, task="classification")
    artifacts = save_task_evaluation_artifacts(
        logits,
        target,
        task="classification",
        output_dir=tmp_path,
        prefix="val",
    )

    assert metrics["accuracy"] >= 0.75
    assert metrics["balanced_accuracy"] >= 0.75
    assert metrics["auroc"] >= 0.75
    assert artifacts.metrics_path.exists()
    assert artifacts.curve_data_path.exists()
    assert artifacts.figure_paths["confusion_matrix"].exists()
    assert artifacts.figure_paths["roc_auc_curve"].exists()
    assert artifacts.figure_paths["pr_auc_curve"].exists()
    assert artifacts.figure_paths["calibration_curve"].exists()

    curve_payload = json.loads(artifacts.curve_data_path.read_text(encoding="utf-8"))
    assert "roc_curves" in curve_payload
    assert "pr_curves" in curve_payload
    assert "calibration_curves" in curve_payload


def test_continuous_survival_metrics_and_artifacts_are_saved(tmp_path: Path) -> None:
    risk = torch.tensor([2.5, 1.0, 0.5, -0.2], dtype=torch.float32)
    target = {
        "time": torch.tensor([1.0, 2.0, 4.0, 5.0], dtype=torch.float32),
        "event": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
    }

    metrics = compute_task_metrics(risk, target, task="survival")
    artifacts = save_task_evaluation_artifacts(
        risk,
        target,
        task="survival",
        output_dir=tmp_path,
        prefix="val",
    )

    assert metrics["c_index"] > 0.5
    assert metrics["num_eval_times"] >= 1.0
    assert artifacts.figure_paths["td_auc_curve"].exists()
    assert artifacts.figure_paths["concordance_index"].exists()
    curve_payload = json.loads(artifacts.curve_data_path.read_text(encoding="utf-8"))
    assert len(curve_payload["c_index_times"]) >= 1
    assert len(curve_payload["c_index_times"]) == len(curve_payload["c_index_curve"])


def test_discrete_survival_metrics_and_artifacts_are_saved(tmp_path: Path) -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.5],
            [2.0, 1.0, 0.2],
            [0.2, 0.1, -0.2],
            [-0.5, -0.8, -1.0],
        ],
        dtype=torch.float32,
    )
    target = {
        "time": torch.tensor([0, 1, 2, 2], dtype=torch.long),
        "event": torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
    }

    metrics = compute_task_metrics(logits, target, task="survival_discrete")
    artifacts = save_task_evaluation_artifacts(
        logits,
        target,
        task="survival_discrete",
        output_dir=tmp_path,
        prefix="val",
    )

    assert metrics["c_index"] >= 0.5
    assert artifacts.metrics_path.exists()
    assert artifacts.figure_paths["td_auc_curve"].exists()
    assert artifacts.figure_paths["concordance_index"].exists()
    curve_payload = json.loads(artifacts.curve_data_path.read_text(encoding="utf-8"))
    assert len(curve_payload["c_index_times"]) >= 1
    assert len(curve_payload["c_index_times"]) == len(curve_payload["c_index_curve"])


def test_compute_task_metrics_filters_to_selected_metrics() -> None:
    logits = torch.tensor(
        [
            [4.0, -2.0],
            [-3.0, 3.5],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([0, 1], dtype=torch.long)

    metrics = compute_task_metrics(
        logits,
        target,
        task="classification",
        selected_metrics=["balanced_accuracy", "f1"],
    )

    assert set(metrics) == {"balanced_accuracy", "f1"}


def test_regression_metrics_and_artifacts_are_saved(tmp_path: Path) -> None:
    predictions = torch.tensor([[1.0], [2.5], [3.75], [5.25]], dtype=torch.float32)
    target = torch.tensor([1.25, 2.0, 4.0, 5.0], dtype=torch.float32)

    metrics = compute_task_metrics(predictions, target, task="regression")
    artifacts = save_task_evaluation_artifacts(
        predictions,
        target,
        task="regression",
        output_dir=tmp_path,
        prefix="val",
    )

    assert set(metrics) == {"mae", "mse", "rmse", "r2"}
    assert metrics["mae"] >= 0.0
    assert artifacts.metrics_path.exists()
    assert artifacts.curve_data_path.exists()
    assert artifacts.figure_paths["regression_scatter"].exists()
    assert artifacts.figure_paths["residuals"].exists()

    curve_payload = json.loads(artifacts.curve_data_path.read_text(encoding="utf-8"))
    assert len(curve_payload["true"]) == 4
    assert len(curve_payload["pred"]) == 4
    assert len(curve_payload["residuals"]) == 4
