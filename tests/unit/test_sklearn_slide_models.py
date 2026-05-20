"""Unit tests for ScikitBase hierarchy and sklearn slide model predictions."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pathbench.core.models.base import ScikitBase


# ---------------------------------------------------------------------------
# ScikitBase abstract contract
# ---------------------------------------------------------------------------


class _MinimalSklearnModel(ScikitBase):
    """Minimal concrete implementation of ScikitBase for contract testing."""

    def initialize(self, config: Any = None) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MinimalSklearnModel":
        return self

    def predict_as_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.zeros(len(X))

    def get_learnable_parameters(self) -> list:
        return []


def test_scikit_base_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        ScikitBase()  # type: ignore[abstract]


def test_minimal_scikit_base_instantiation():
    model = _MinimalSklearnModel()
    assert isinstance(model, ScikitBase)


def test_scikit_base_save_load_roundtrip(tmp_path: Path):
    model = _MinimalSklearnModel()
    path = str(tmp_path / "model.pkl")
    model.save(path)
    assert Path(path).exists()

    model2 = _MinimalSklearnModel()
    model2.load(path)
    # after load, _estimator state is restored — no error


def test_scikit_base_save_creates_parent_dirs(tmp_path: Path):
    model = _MinimalSklearnModel()
    path = str(tmp_path / "nested" / "dir" / "model.pkl")
    model.save(path)
    assert Path(path).exists()


# ---------------------------------------------------------------------------
# SklearnSlideClassifier.predict_as_tensor shape and value correctness
# ---------------------------------------------------------------------------


def test_sklearn_logistic_regression_predict_as_tensor_shape():
    from pathbench.core.models.sklearn_slide import SklearnLogisticRegressionClassifier

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((20, 4)).astype(np.float32)
    y_train = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    model = SklearnLogisticRegressionClassifier(max_iter=200)
    model.fit(X_train, y_train)

    X_val = rng.standard_normal((5, 4)).astype(np.float32)
    tensor = model.predict_as_tensor(X_val)

    assert tensor.shape == (5, 2)
    assert tensor.dtype == torch.float32


def test_sklearn_logistic_regression_predict_as_tensor_is_log_probabilities():
    """softmax(log_proba) must recover probabilities summing to 1."""
    from pathbench.core.models.sklearn_slide import SklearnLogisticRegressionClassifier

    rng = np.random.default_rng(1)
    X_train = rng.standard_normal((20, 4)).astype(np.float32)
    y_train = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    model = SklearnLogisticRegressionClassifier(max_iter=200)
    model.fit(X_train, y_train)

    X_val = rng.standard_normal((6, 4)).astype(np.float32)
    log_proba = model.predict_as_tensor(X_val)
    recovered = torch.softmax(log_proba, dim=-1)

    assert torch.allclose(recovered.sum(dim=-1), torch.ones(6), atol=1e-5)
    assert (recovered >= 0.0).all()
    assert (recovered <= 1.0).all()


# ---------------------------------------------------------------------------
# SklearnSlideRegressor.predict_as_tensor shape
# ---------------------------------------------------------------------------


def test_sklearn_ridge_predict_as_tensor_shape():
    from pathbench.core.models.sklearn_slide import SklearnRidgeRegressor

    rng = np.random.default_rng(2)
    X_train = rng.standard_normal((15, 8)).astype(np.float32)
    y_train = rng.standard_normal(15).astype(np.float32)

    model = SklearnRidgeRegressor(alpha=1.0)
    model.fit(X_train, y_train)

    X_val = rng.standard_normal((4, 8)).astype(np.float32)
    tensor = model.predict_as_tensor(X_val)

    assert tensor.shape == (4,)
    assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# New named models — GradientBoosting, Linear, SVR
# ---------------------------------------------------------------------------


def test_gradient_boosting_classifier_predict_as_tensor_shape():
    from pathbench.core.models.sklearn_slide import SklearnGradientBoostingClassifier

    rng = np.random.default_rng(3)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    model = SklearnGradientBoostingClassifier(n_estimators=5)
    model.fit(X, y)
    tensor = model.predict_as_tensor(rng.standard_normal((5, 4)).astype(np.float32))

    assert tensor.shape == (5, 2)
    assert tensor.dtype == torch.float32


def test_linear_regressor_predict_as_tensor_shape():
    from pathbench.core.models.sklearn_slide import SklearnLinearRegressor

    rng = np.random.default_rng(4)
    X = rng.standard_normal((20, 6)).astype(np.float32)
    y = rng.standard_normal(20).astype(np.float32)

    model = SklearnLinearRegressor()
    model.fit(X, y)
    tensor = model.predict_as_tensor(rng.standard_normal((7, 6)).astype(np.float32))

    assert tensor.shape == (7,)
    assert tensor.dtype == torch.float32


def test_gradient_boosting_regressor_predict_as_tensor_shape():
    from pathbench.core.models.sklearn_slide import SklearnGradientBoostingRegressor

    rng = np.random.default_rng(5)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = rng.standard_normal(20).astype(np.float32)

    model = SklearnGradientBoostingRegressor(n_estimators=5)
    model.fit(X, y)
    tensor = model.predict_as_tensor(rng.standard_normal((4, 4)).astype(np.float32))

    assert tensor.shape == (4,)


def test_svm_regressor_predict_as_tensor_shape():
    from pathbench.core.models.sklearn_slide import SklearnSVMRegressor

    rng = np.random.default_rng(6)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = rng.standard_normal(20).astype(np.float32)

    model = SklearnSVMRegressor(C=1.0)
    model.fit(X, y)
    tensor = model.predict_as_tensor(rng.standard_normal((3, 4)).astype(np.float32))

    assert tensor.shape == (3,)


# ---------------------------------------------------------------------------
# Dynamic catalog and factory
# ---------------------------------------------------------------------------


def test_sklearn_estimator_catalog_has_required_models():
    from pathbench.core.models.sklearn_slide import SKLEARN_ESTIMATOR_CATALOG

    required = {
        "SklearnLogisticRegressionClassifier",
        "SklearnSVMClassifier",
        "SklearnRandomForestClassifier",
        "SklearnGradientBoostingClassifier",
        "SklearnLinearRegressor",
        "SklearnRidgeRegressor",
        "SklearnRandomForestRegressor",
        "SklearnGradientBoostingRegressor",
        "SklearnCoxPH",
        "SklearnRandomSurvivalForest",
    }
    assert required <= set(SKLEARN_ESTIMATOR_CATALOG)


def test_list_sklearn_slide_models_returns_sklearn_entries():
    from pathbench.core.models.sklearn_slide import list_sklearn_slide_models

    all_models = list_sklearn_slide_models()
    # sklearn is always present — at minimum classification + regression models
    assert len(all_models) >= 5
    assert "SklearnLogisticRegressionClassifier" in all_models
    assert "SklearnRidgeRegressor" in all_models


def test_list_sklearn_slide_models_task_filter():
    from pathbench.core.models.sklearn_slide import list_sklearn_slide_models

    clf_models = list_sklearn_slide_models(task="classification")
    reg_models = list_sklearn_slide_models(task="regression")
    surv_models = list_sklearn_slide_models(task="survival")

    assert all("Classifier" in m or m.endswith("SVM") or "SVM" in m or "Logistic" in m or "Forest" in m or "Boosting" in m for m in clf_models), clf_models
    # regression models should not appear in classification list
    assert "SklearnRidgeRegressor" not in clf_models
    # no survival in regression list
    assert "SklearnCoxPH" not in reg_models


def test_make_sklearn_slide_model_classification():
    from pathbench.core.models.sklearn_slide import (
        SklearnSlideClassifier,
        make_sklearn_slide_model,
    )

    rng = np.random.default_rng(7)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    model = make_sklearn_slide_model(
        "SklearnLogisticRegressionClassifier", max_iter=200
    )
    assert isinstance(model, SklearnSlideClassifier)
    model.fit(X, y)
    tensor = model.predict_as_tensor(X)
    assert tensor.shape == (20, 2)


def test_make_sklearn_slide_model_regression():
    from pathbench.core.models.sklearn_slide import (
        SklearnSlideRegressor,
        make_sklearn_slide_model,
    )

    rng = np.random.default_rng(8)
    X = rng.standard_normal((15, 4)).astype(np.float32)
    y = rng.standard_normal(15).astype(np.float32)

    model = make_sklearn_slide_model("SklearnRidgeRegressor", alpha=0.5)
    assert isinstance(model, SklearnSlideRegressor)
    model.fit(X, y)
    tensor = model.predict_as_tensor(X)
    assert tensor.shape == (15,)


def test_make_sklearn_slide_model_unknown_name_raises():
    from pathbench.core.models.sklearn_slide import make_sklearn_slide_model

    with pytest.raises(ValueError, match="Unknown sklearn slide model"):
        make_sklearn_slide_model("NotARealModel")


def test_make_sklearn_slide_model_svm_classifier_uses_probability():
    """SVC in catalog must have probability=True so predict_proba works."""
    from pathbench.core.models.sklearn_slide import make_sklearn_slide_model

    rng = np.random.default_rng(9)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)

    model = make_sklearn_slide_model("SklearnSVMClassifier")
    model.fit(X, y)
    # should not raise — predict_proba requires probability=True
    tensor = model.predict_as_tensor(X)
    assert tensor.shape == (20, 2)


# ---------------------------------------------------------------------------
# SKLEARN_SLIDE_MODEL_NAMES and SLIDE_LEVEL_MODEL_NAMES completeness
# ---------------------------------------------------------------------------


def test_sklearn_slide_model_names_derived_from_catalog():
    from pathbench.core.models.sklearn_slide import (
        SKLEARN_ESTIMATOR_CATALOG,
        SKLEARN_SLIDE_MODEL_NAMES,
    )

    assert set(SKLEARN_SLIDE_MODEL_NAMES) == set(SKLEARN_ESTIMATOR_CATALOG)


def test_slide_level_model_names_includes_mlp():
    from pathbench.core.models.sklearn_slide import SLIDE_LEVEL_MODEL_NAMES

    assert "SlideVectorMLP" in SLIDE_LEVEL_MODEL_NAMES


# ---------------------------------------------------------------------------
# Survival structured array helper
# ---------------------------------------------------------------------------


def test_make_survival_structured_array_dtype():
    from pathbench.core.models.sklearn_slide import make_survival_structured_array

    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1, 0, 1])
    y = make_survival_structured_array(time, event)

    assert y.dtype.names == ("event", "time")
    assert y["event"].dtype == bool
    assert y["time"].dtype == np.float64
    assert y["event"][0] is np.bool_(True)
    assert y["event"][1] is np.bool_(False)


# ---------------------------------------------------------------------------
# Normalization (StandardScaler)
# ---------------------------------------------------------------------------


def _make_clf_data(n: int = 20, d: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32) * 10 + 50  # non-zero mean/scale
    y = (X[:, 0] > 50).astype(np.int64)
    return X, y


def test_normalization_scaler_fitted_after_fit():
    from pathbench.core.models.sklearn_slide import SklearnLogisticRegressionClassifier

    X, y = _make_clf_data()
    model = SklearnLogisticRegressionClassifier(max_iter=200, normalize=True)
    assert model._scaler is None
    model.fit(X, y)
    assert model._scaler is not None


def test_normalization_disabled_no_scaler():
    from pathbench.core.models.sklearn_slide import SklearnLogisticRegressionClassifier

    X, y = _make_clf_data()
    model = SklearnLogisticRegressionClassifier(max_iter=200, normalize=False)
    model.fit(X, y)
    assert model._scaler is None


def test_normalization_scaler_applied_in_predict_as_tensor():
    """Prediction with normalize=True must apply the same scaler as fitted."""
    from pathbench.core.models.sklearn_slide import SklearnLogisticRegressionClassifier

    X, y = _make_clf_data()
    model = SklearnLogisticRegressionClassifier(max_iter=200, normalize=True)
    model.fit(X, y)

    # Manually transform X and compare
    X_scaled = model._scaler.transform(X)
    proba_direct = model._estimator.predict_proba(X_scaled).astype(np.float32)
    log_proba_expected = np.log(np.clip(proba_direct, 1e-7, 1.0))
    tensor = model.predict_as_tensor(X)

    assert torch.allclose(tensor, torch.from_numpy(log_proba_expected), atol=1e-5)


def test_normalization_regression_scaler_applied():
    from pathbench.core.models.sklearn_slide import SklearnRidgeRegressor

    rng = np.random.default_rng(10)
    X = (rng.standard_normal((20, 4)) * 100).astype(np.float32)
    y = rng.standard_normal(20).astype(np.float32)

    model = SklearnRidgeRegressor(alpha=1.0, normalize=True)
    model.fit(X, y)
    assert model._scaler is not None

    X_scaled = model._scaler.transform(X)
    expected = model._estimator.predict(X_scaled).astype(np.float32)
    tensor = model.predict_as_tensor(X)
    assert torch.allclose(tensor, torch.from_numpy(expected), atol=1e-5)


def test_normalization_logs_info_message(caplog):
    import logging
    from pathbench.core.models.sklearn_slide import SklearnLogisticRegressionClassifier

    X, y = _make_clf_data()
    model = SklearnLogisticRegressionClassifier(max_iter=200, normalize=True)
    with caplog.at_level(logging.INFO, logger="pathbench.core.models.sklearn_slide"):
        model.fit(X, y)

    assert any("StandardScaler" in r.message for r in caplog.records)
    assert any("normalize" in r.message.lower() or "standardized" in r.message.lower() for r in caplog.records)


def test_normalization_factory_normalize_false():
    from pathbench.core.models.sklearn_slide import make_sklearn_slide_model

    rng = np.random.default_rng(11)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    model = make_sklearn_slide_model(
        "SklearnLogisticRegressionClassifier", normalize=False, max_iter=200
    )
    model.fit(X, y)
    assert model._scaler is None


# ---------------------------------------------------------------------------
# sksurv models (require scikit-survival)
# ---------------------------------------------------------------------------


def _make_surv_data(n: int = 30, d: int = 6, seed: int = 0):
    from pathbench.core.models.sklearn_slide import make_survival_structured_array

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    time = np.abs(rng.standard_normal(n)) + 0.5
    event = rng.integers(0, 2, n).astype(np.float64)
    y = make_survival_structured_array(time, event)
    return X, y


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_sklearn_coxph_normalizes_and_predicts():
    from pathbench.core.models.sklearn_slide import SklearnCoxPH

    X, y = _make_surv_data()
    model = SklearnCoxPH(alpha=0.1, normalize=True)
    model.fit(X, y)
    assert model._scaler is not None
    tensor = model.predict_as_tensor(X)
    assert tensor.shape == (30,)
    assert tensor.dtype == torch.float32


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_sklearn_coxnet_normalizes_and_predicts():
    from pathbench.core.models.sklearn_slide import SklearnCoxnet

    X, y = _make_surv_data()
    model = SklearnCoxnet(l1_ratio=0.5, normalize=True)
    model.fit(X, y)
    tensor = model.predict_as_tensor(X)
    assert tensor.shape == (30,)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_sklearn_ipc_ridge_normalizes_and_predicts():
    from pathbench.core.models.sklearn_slide import SklearnIPCRidge

    X, y = _make_surv_data()
    model = SklearnIPCRidge(alpha=1.0, normalize=True)
    model.fit(X, y)
    tensor = model.predict_as_tensor(X)
    assert tensor.shape == (30,)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_sklearn_survival_tree_normalizes_and_predicts():
    from pathbench.core.models.sklearn_slide import SklearnSurvivalTree

    X, y = _make_surv_data()
    model = SklearnSurvivalTree(max_depth=3, normalize=True)
    model.fit(X, y)
    tensor = model.predict_as_tensor(X)
    assert tensor.shape == (30,)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_sklearn_hinge_loss_svm_normalizes_and_predicts():
    from pathbench.core.models.sklearn_slide import SklearnHingeLossSurvivalSVM

    X, y = _make_surv_data()
    model = SklearnHingeLossSurvivalSVM(alpha=1.0, normalize=True)
    model.fit(X, y)
    tensor = model.predict_as_tensor(X)
    assert tensor.shape == (30,)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_sklearn_naive_survival_svm_normalizes_and_predicts():
    from pathbench.core.models.sklearn_slide import SklearnNaiveSurvivalSVM

    X, y = _make_surv_data()
    model = SklearnNaiveSurvivalSVM(alpha=1.0, normalize=True)
    model.fit(X, y)
    tensor = model.predict_as_tensor(X)
    assert tensor.shape == (30,)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sksurv") is None,
    reason="scikit-survival not installed",
)
def test_list_sklearn_slide_models_includes_all_sksurv():
    from pathbench.core.models.sklearn_slide import list_sklearn_slide_models

    survival = list_sklearn_slide_models(task="survival")
    expected = {
        "SklearnCoxPH",
        "SklearnCoxnet",
        "SklearnIPCRidge",
        "SklearnHingeLossSurvivalSVM",
        "SklearnNaiveSurvivalSVM",
        "SklearnSurvivalTree",
        "SklearnRandomSurvivalForest",
    }
    assert expected <= set(survival)
