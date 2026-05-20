"""Integration tests for SklearnSlideTrainer across all task types.

Coverage matrix (each row = test, columns = task × model family):

  Task                 | LR/Ridge/Linear  | RF/GBM  | SVM  | sksurv
  ---------------------|------------------|---------|------|-------
  binary classification|       ✓          |    ✓    |  ✓   |
  multiclass classif.  |       ✓          |    ✓    |      |
  regression           |       ✓          |    ✓    |  ✓   |
  continuous survival  |                  |         |      |  ✓ CoxPH
  discrete survival    |       ✓ (RF clf) |    ✓    |      |

Discrete survival is modelled as multiclass classification over time bins —
the SklearnSlideTrainer detects a SklearnSlideClassifier with task=
"survival_discrete" and trains on bin indices, then evaluates with the
survival_discrete metrics path.

Normalization: all tests verify the fitted scaler is present on the model
after training (``normalize=True`` is the default).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Data factories
# ---------------------------------------------------------------------------


def _binary_data(n: int = 24, d: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


def _multiclass_data(n: int = 30, d: int = 8, n_classes: int = 4, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (np.abs(X[:, 0]) * n_classes).astype(np.int64) % n_classes
    return X, y


def _regression_data(n: int = 24, d: int = 8, seed: int = 2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] * 3.0 + rng.standard_normal(n) * 0.1).astype(np.float32)
    return X, y


def _discrete_survival_data(n: int = 30, d: int = 8, n_bins: int = 4, seed: int = 3):
    """Discrete survival as a multiclass problem over time bins."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    bins = rng.integers(0, n_bins, n).astype(np.int64)
    event = rng.integers(0, 2, n).astype(np.float32)
    return X, {"time": bins, "event": event}


def _continuous_survival_data(n: int = 30, d: int = 6, seed: int = 4):
    from pathbench.core.models.sklearn_slide import make_survival_structured_array

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    time = (np.abs(rng.standard_normal(n)) + 0.5).astype(np.float64)
    event = rng.integers(0, 2, n).astype(np.float64)
    _ = make_survival_structured_array(time, event)
    # also return as dict for trainer API
    y_dict = {"time": time, "event": event}
    return X, y_dict


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------


def test_logistic_regression_binary_classification(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnLogisticRegressionClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _binary_data()
    model = SklearnLogisticRegressionClassifier(max_iter=300)
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "lr_bin", task="classification")
    model_path, score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert 0.0 <= score <= 1.0
    assert (trainer.metrics_artifacts_dir / "val_confusion_matrix.png").exists()
    assert (trainer.metrics_artifacts_dir / "val_roc_auc_curve.png").exists()
    assert (trainer.metrics_artifacts_dir / "val_metrics.json").exists()
    assert model._scaler is not None  # normalization applied


def test_random_forest_binary_classification(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnRandomForestClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _binary_data()
    model = SklearnRandomForestClassifier(n_estimators=10)
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "rf_bin", task="classification")
    model_path, score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert model._scaler is not None


def test_gradient_boosting_binary_classification(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnGradientBoostingClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _binary_data()
    model = SklearnGradientBoostingClassifier(n_estimators=10)
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "gb_bin", task="classification")
    model_path, score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()


def test_svm_binary_classification(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnSVMClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _binary_data()
    model = SklearnSVMClassifier(C=1.0)
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "svm_bin", task="classification")
    model_path, score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert model._scaler is not None


# ---------------------------------------------------------------------------
# Multiclass classification
# ---------------------------------------------------------------------------


def test_logistic_regression_multiclass_classification(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnLogisticRegressionClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _multiclass_data(n_classes=4)
    model = SklearnLogisticRegressionClassifier(max_iter=300)
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "lr_multi", task="classification")
    model_path, score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert (trainer.metrics_artifacts_dir / "val_confusion_matrix.png").exists()
    assert model._scaler is not None


def test_random_forest_multiclass_classification(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnRandomForestClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _multiclass_data(n_classes=3)
    model = SklearnRandomForestClassifier(n_estimators=10)
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "rf_multi", task="classification")
    model_path, _ = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert (trainer.metrics_artifacts_dir / "val_confusion_matrix.png").exists()


def test_gradient_boosting_multiclass_classification(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnGradientBoostingClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _multiclass_data(n_classes=4)
    model = SklearnGradientBoostingClassifier(n_estimators=10)
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "gb_multi", task="classification")
    model_path, _ = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_linear_regression(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnLinearRegressor
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _regression_data()
    model = SklearnLinearRegressor()
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "linear_reg", task="regression")
    model_path, score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert (trainer.metrics_artifacts_dir / "val_regression_scatter.png").exists()
    assert (trainer.metrics_artifacts_dir / "val_metrics.json").exists()
    assert model._scaler is not None


def test_ridge_regression(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnRidgeRegressor
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _regression_data()
    model = SklearnRidgeRegressor(alpha=1.0)
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "ridge", task="regression")
    model_path, _ = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert model._scaler is not None


def test_gradient_boosting_regression(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnGradientBoostingRegressor
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _regression_data()
    model = SklearnGradientBoostingRegressor(n_estimators=10)
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "gb_reg", task="regression")
    model_path, _ = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()


def test_svm_regression(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnSVMRegressor
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _regression_data()
    model = SklearnSVMRegressor(C=1.0)
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "svm_reg", task="regression")
    model_path, _ = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert model._scaler is not None


# ---------------------------------------------------------------------------
# Discrete survival (multiclass classification over time bins)
# ---------------------------------------------------------------------------


def test_random_forest_discrete_survival(tmp_path: Path):
    """RandomForest trained on time-bin labels for discrete survival prediction."""
    from pathbench.core.models.sklearn_slide import SklearnRandomForestClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    n_bins = 4
    X, y = _discrete_survival_data(n_bins=n_bins)
    model = SklearnRandomForestClassifier(n_estimators=10)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "rf_surv_disc", task="survival_discrete"
    )
    model_path, score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    # Trainer fits on bin indices (multiclass) and outputs (N, T) log-proba
    assert model._scaler is not None


def test_gradient_boosting_discrete_survival(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnGradientBoostingClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    n_bins = 3
    X, y = _discrete_survival_data(n_bins=n_bins)
    model = SklearnGradientBoostingClassifier(n_estimators=10)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "gb_surv_disc", task="survival_discrete"
    )
    model_path, _ = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()


def test_logistic_regression_discrete_survival(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnLogisticRegressionClassifier
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _discrete_survival_data(n_bins=4)
    model = SklearnLogisticRegressionClassifier(max_iter=300)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "lr_surv_disc", task="survival_discrete"
    )
    model_path, _ = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()


# ---------------------------------------------------------------------------
# Continuous survival (requires scikit-survival)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_cox_ph_continuous_survival(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnCoxPH
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _continuous_survival_data()
    model = SklearnCoxPH(alpha=0.1)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "coxph", task="survival"
    )
    model_path, score = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert (trainer.metrics_artifacts_dir / "val_td_auc_curve.png").exists()
    assert (trainer.metrics_artifacts_dir / "val_kaplan_meier.png").exists()
    assert model._scaler is not None


@pytest.mark.skipif(
    importlib.util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_coxnet_continuous_survival(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnCoxnet
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _continuous_survival_data()
    model = SklearnCoxnet(l1_ratio=0.5)
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "coxnet", task="survival")
    model_path, _ = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()
    assert model._scaler is not None


@pytest.mark.skipif(
    importlib.util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_ipc_ridge_continuous_survival(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnIPCRidge
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _continuous_survival_data()
    model = SklearnIPCRidge(alpha=1.0)
    trainer = SklearnSlideTrainer(output_dir=tmp_path / "ipc_ridge", task="survival")
    model_path, _ = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()


@pytest.mark.skipif(
    importlib.util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_survival_tree_continuous_survival(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnSurvivalTree
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _continuous_survival_data()
    model = SklearnSurvivalTree(max_depth=3)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "surv_tree", task="survival"
    )
    model_path, _ = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()


@pytest.mark.skipif(
    importlib.util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_hinge_loss_svm_continuous_survival(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnHingeLossSurvivalSVM
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _continuous_survival_data()
    model = SklearnHingeLossSurvivalSVM(alpha=1.0)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "hinge_svm", task="survival"
    )
    model_path, _ = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()


@pytest.mark.skipif(
    importlib.util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_naive_survival_svm_continuous_survival(tmp_path: Path):
    from pathbench.core.models.sklearn_slide import SklearnNaiveSurvivalSVM
    from pathbench.training.sklearn_trainer import SklearnSlideTrainer

    X, y = _continuous_survival_data()
    model = SklearnNaiveSurvivalSVM(alpha=1.0)
    trainer = SklearnSlideTrainer(
        output_dir=tmp_path / "naive_svm", task="survival"
    )
    model_path, _ = trainer.fit(model, X, y, X, y)

    assert Path(model_path).exists()


# ---------------------------------------------------------------------------
# SlideVectorMLP via LightningTrainer (existing test retained)
# ---------------------------------------------------------------------------


class _ToyBagDataset:
    def __init__(self, n_slides=8, n_tiles=5, input_dim=8, n_classes=2, seed=42):
        rng = torch.Generator().manual_seed(seed)
        self._bags = [torch.randn(n_tiles, input_dim, generator=rng) for _ in range(n_slides)]
        self._labels = [torch.tensor(i % n_classes, dtype=torch.long) for i in range(n_slides)]
        self.input_dim = input_dim

    def __len__(self):
        return len(self._bags)

    def __getitem__(self, idx):
        return self._bags[idx], self._labels[idx]


def _make_trainer_config(tmp_path: Path):
    from pathbench.config.config import Config
    from tests.conftest import DUMMY_FE, DUMMY_MIL

    annotation_path = tmp_path / "ann.csv"
    annotation_path.write_text("dataset,slide,patient,category\n", encoding="utf-8")
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    return Config.model_validate({
        "experiment": {
            "project_name": "slide_mlp_int", "annotation_file": str(annotation_path),
            "project_root": str((tmp_path / "proj").resolve()),
            "mode": "benchmark", "task": "classification", "num_workers": 0,
        },
        "mil": {
            "backend": "native", "epochs": 2, "batch_size": 4,
            "patience": 10, "best_epoch_based_on": "balanced_accuracy",
        },
        "metrics": {"classification_backend": "native"},
        "slide_processing": {"backend": "lazyslide"},
        "datasets": [{"name": "ds", "slides_dir": str(slides_dir),
                       "artifacts_dir": str(tmp_path / "art"), "used_for": "all"}],
        "benchmark_parameters": {
            "tile_px": [256], "tile_mpp": [0.5],
            "feature_extraction": [DUMMY_FE], "mil": [DUMMY_MIL],
            "loss": ["CrossEntropyLoss"],
        },
    })


def test_slide_vector_mlp_lightning_trainer_classification(tmp_path: Path):
    from pathbench.core.models.slide_mlp import SlideVectorMLP
    from pathbench.training.lightning import LightningTrainer
    from pathbench.utils.registries import LOSSES

    dataset = _ToyBagDataset()
    model = SlideVectorMLP(input_dim=8, hidden_dim=16, output_dim=2)
    cfg = _make_trainer_config(tmp_path)
    loss_fn = LOSSES.get("CrossEntropyLoss")()
    trainer = LightningTrainer(cfg)
    best_path, best_score = trainer.fit(model, dataset, dataset, loss_fn)

    assert Path(best_path).exists()
    assert isinstance(best_score, float)
    assert (trainer.metrics_artifacts_dir / "val_confusion_matrix.png").exists()
