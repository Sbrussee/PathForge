"""Smoke coverage for scikit-learn slide-level estimators.

Each test:
1. Aggregates tile-level bag features into one slide-level vector per slide
   (mean + max pooling, reusing the session-scoped ``extracted_bag_workspace``
   fixture so no extra feature extraction runs).
2. Trains an sklearn estimator via ``SklearnSlideTrainer``.
3. Asserts that the model pickle and evaluation artefacts exist.

Survival tests require ``scikit-survival`` (``sksurv``) and are skipped
automatically when the package is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ._smoke_dataset import PreparedBagWorkspace, attach_smoke_outputs, capture_smoke_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_slide_features(
    feature_dir: Path,
    slide_ids: list[str],
) -> np.ndarray:
    """Mean-pool each bag tensor to a slide-level feature vector."""
    rows: list[np.ndarray] = []
    for sid in slide_ids:
        bag = torch.load(feature_dir / f"{sid}.pt", weights_only=True).float()
        rows.append(bag.mean(dim=0).numpy())
    return np.stack(rows, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_sklearn_logistic_regression_classification_smoke(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Logistic regression on slide-level mean-pooled features."""
    from pathbench.core.models.sklearn_slide import SklearnLogisticRegressionClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    metadata_df = pd.read_csv(extracted_bag_workspace.metadata_csv)
    slide_ids = metadata_df["slide_id"].tolist()
    X = _load_slide_features(extracted_bag_workspace.feature_dir, slide_ids)
    y = metadata_df["binary_label"].to_numpy(dtype=np.int64)

    model = SklearnLogisticRegressionClassifier(max_iter=200)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "lr_classification",
        task="classification",
    )

    with capture_smoke_metrics(
        tmp_path / "metrics",
        step_name="sklearn_lr_classification",
        metadata={"n_samples": len(slide_ids), "input_dim": X.shape[1]},
    ) as meta:
        model_path, score = trainer.fit(model, X, y, X, y)
        artifacts_dir = trainer.metrics_artifacts_dir
        attach_smoke_outputs(
            meta,
            step_name="sklearn_lr_classification",
            final={
                "model_pickle": Path(model_path),
                "val_confusion_matrix": artifacts_dir / "val_confusion_matrix.png",
                "val_metrics_json": artifacts_dir / "val_metrics.json",
            },
        )

    assert Path(model_path).exists()
    assert (artifacts_dir / "val_confusion_matrix.png").exists()
    assert (artifacts_dir / "val_roc_auc_curve.png").exists()
    assert (artifacts_dir / "val_metrics.json").exists()


@pytest.mark.smoke
def test_sklearn_random_forest_classification_smoke(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Random forest classifier on slide-level mean-pooled features."""
    from pathbench.core.models.sklearn_slide import SklearnRandomForestClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    metadata_df = pd.read_csv(extracted_bag_workspace.metadata_csv)
    slide_ids = metadata_df["slide_id"].tolist()
    X = _load_slide_features(extracted_bag_workspace.feature_dir, slide_ids)
    y = metadata_df["binary_label"].to_numpy(dtype=np.int64)

    model = SklearnRandomForestClassifier(n_estimators=10)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "rf_classification",
        task="classification",
    )
    model_path, score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert (trainer.metrics_artifacts_dir / "val_confusion_matrix.png").exists()
    assert (trainer.metrics_artifacts_dir / "val_metrics.json").exists()


@pytest.mark.smoke
def test_sklearn_multiclass_classification_heatmap_smoke(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Multiclass (3+) logistic regression; verifies confusion-matrix heatmap."""
    from pathbench.core.models.sklearn_slide import SklearnLogisticRegressionClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    metadata_df = pd.read_csv(extracted_bag_workspace.metadata_csv)
    num_classes = int(metadata_df["multiclass_label"].nunique())
    if num_classes < 3:
        pytest.skip("Fewer than 3 multiclass labels.")

    slide_ids = metadata_df["slide_id"].tolist()
    X = _load_slide_features(extracted_bag_workspace.feature_dir, slide_ids)
    y = metadata_df["multiclass_label"].to_numpy(dtype=np.int64)

    model = SklearnLogisticRegressionClassifier(max_iter=200)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "lr_multiclass",
        task="classification",
    )
    model_path, _score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert (trainer.metrics_artifacts_dir / "val_confusion_matrix.png").exists(), (
        "Confusion-matrix heatmap not generated for multiclass sklearn model."
    )
    assert (trainer.metrics_artifacts_dir / "val_roc_auc_curve.png").exists()


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_sklearn_ridge_regression_smoke(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Ridge regression on a synthetic continuous target derived from labels."""
    from pathbench.core.models.sklearn_slide import SklearnRidgeRegressor
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    metadata_df = pd.read_csv(extracted_bag_workspace.metadata_csv)
    slide_ids = metadata_df["slide_id"].tolist()
    X = _load_slide_features(extracted_bag_workspace.feature_dir, slide_ids)
    y = metadata_df["multiclass_label"].to_numpy(dtype=np.float32)

    model = SklearnRidgeRegressor(alpha=1.0)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "ridge_regression",
        task="regression",
    )
    model_path, _score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert (trainer.metrics_artifacts_dir / "val_regression_scatter.png").exists()
    assert (trainer.metrics_artifacts_dir / "val_metrics.json").exists()


@pytest.mark.smoke
def test_sklearn_gradient_boosting_regression_smoke(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """Gradient boosting regressor on slide-level mean-pooled features."""
    from pathbench.core.models.sklearn_slide import SklearnGradientBoostingRegressor
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    metadata_df = pd.read_csv(extracted_bag_workspace.metadata_csv)
    slide_ids = metadata_df["slide_id"].tolist()
    X = _load_slide_features(extracted_bag_workspace.feature_dir, slide_ids)
    y = metadata_df["multiclass_label"].to_numpy(dtype=np.float32)

    model = SklearnGradientBoostingRegressor(n_estimators=10)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "gb_regression",
        task="regression",
    )
    model_path, _score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert (trainer.metrics_artifacts_dir / "val_regression_scatter.png").exists()


@pytest.mark.smoke
def test_sklearn_factory_gradient_boosting_classification_smoke(
    extracted_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """make_sklearn_slide_model factory with GradientBoostingClassifier."""
    from pathbench.core.models.sklearn_slide import make_sklearn_slide_model
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    metadata_df = pd.read_csv(extracted_bag_workspace.metadata_csv)
    slide_ids = metadata_df["slide_id"].tolist()
    X = _load_slide_features(extracted_bag_workspace.feature_dir, slide_ids)
    y = metadata_df["binary_label"].to_numpy(dtype=np.int64)

    model = make_sklearn_slide_model("SklearnGradientBoostingClassifier", n_estimators=10)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "factory_gb_clf",
        task="classification",
    )
    model_path, _score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert (trainer.metrics_artifacts_dir / "val_confusion_matrix.png").exists()


# ---------------------------------------------------------------------------
# Survival (requires scikit-survival)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_sklearn_cox_ph_survival_smoke(
    survival_bag_workspace: PreparedBagWorkspace,
    tmp_path: Path,
) -> None:
    """CoxPH survival model on TCGA READ slide-level features."""
    pytest.importorskip("sksurv", reason="scikit-survival not installed")

    from pathbench.core.models.sklearn_slide import SklearnCoxPH
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    metadata_df = pd.read_csv(survival_bag_workspace.metadata_csv)
    slide_ids = metadata_df["slide_id"].tolist()
    X = _load_slide_features(survival_bag_workspace.feature_dir, slide_ids)
    time = metadata_df["os_months"].to_numpy(dtype=np.float64)
    event = metadata_df["status"].to_numpy(dtype=np.float64)
    y = {"time": time, "event": event}

    model = SklearnCoxPH(alpha=0.1)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "coxph_survival",
        task="survival",
    )
    model_path, _score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert (trainer.metrics_artifacts_dir / "val_td_auc_curve.png").exists()
    assert (trainer.metrics_artifacts_dir / "val_kaplan_meier.png").exists()
    assert (trainer.metrics_artifacts_dir / "val_metrics.json").exists()
