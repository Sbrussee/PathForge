"""Scikit-learn and scikit-survival based slide-level predictors.

All classes inherit from :class:`~pathforge.core.models.base.ScikitBase`
(which in turn inherits from :class:`~pathforge.core.models.base.ModelBase`)
so they participate in the shared PathForge model hierarchy without
requiring PyTorch or the Lightning training stack.

These models are designed exclusively for **slide-level vectors** — either
precomputed aggregated feature vectors or mean/max-pooled tile features.
They are trained via :class:`~pathforge.training.sklearn_trainer.SklearnSlideTrainer`
rather than :class:`~pathforge.training.lightning.LightningTrainer`.

**Feature normalization**

All base wrapper classes (classifiers, regressors, survival) fit a
``sklearn.preprocessing.StandardScaler`` on the training features by default
(``normalize=True``).  The scaler is stored on the instance so that the same
transformation is applied during prediction.  A structured log message at
INFO level is emitted whenever normalization is performed, making it easy to
verify that scaling occurred.  Pass ``normalize=False`` to disable.

**Survival models**

``sksurv`` models require ``scikit-survival`` (optional).  All sksurv classes
guard their import inside ``__init__`` via :func:`_require_sksurv` so the
rest of the module remains importable when the package is absent.

**Dynamic catalog**

:data:`SKLEARN_ESTIMATOR_CATALOG` maps PathForge model names to
``(sklearn_module, class_name, task, fixed_kwargs)``.
Use :func:`make_sklearn_slide_model` to instantiate any catalog entry and
:func:`list_sklearn_slide_models` to enumerate those whose package is installed.

Attributes:
    SKLEARN_ESTIMATOR_CATALOG: Full model catalog.
    SKLEARN_SLIDE_MODEL_NAMES: Frozenset derived from catalog keys.
    SLIDE_LEVEL_MODEL_NAMES: Union of sklearn names and ``SlideVectorMLP``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Optional

import numpy as np

from pathforge.core.models.base import ScikitBase

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog: PathForge name → (sklearn_module, class_name, task, fixed_kwargs)
# ---------------------------------------------------------------------------

SKLEARN_ESTIMATOR_CATALOG: dict[str, tuple[str, str, str, dict[str, Any]]] = {
    # --- Classification --------------------------------------------------
    "SklearnLogisticRegressionClassifier": (
        "sklearn.linear_model",
        "LogisticRegression",
        "classification",
        {},
    ),
    "SklearnRandomForestClassifier": (
        "sklearn.ensemble",
        "RandomForestClassifier",
        "classification",
        {},
    ),
    "SklearnGradientBoostingClassifier": (
        "sklearn.ensemble",
        "GradientBoostingClassifier",
        "classification",
        {},
    ),
    "SklearnSVMClassifier": (
        "sklearn.svm",
        "SVC",
        "classification",
        {"probability": True},  # required for predict_proba
    ),
    # --- Regression ------------------------------------------------------
    "SklearnLinearRegressor": (
        "sklearn.linear_model",
        "LinearRegression",
        "regression",
        {},
    ),
    "SklearnRidgeRegressor": (
        "sklearn.linear_model",
        "Ridge",
        "regression",
        {},
    ),
    "SklearnElasticNetRegressor": (
        "sklearn.linear_model",
        "ElasticNet",
        "regression",
        {},
    ),
    "SklearnRandomForestRegressor": (
        "sklearn.ensemble",
        "RandomForestRegressor",
        "regression",
        {},
    ),
    "SklearnGradientBoostingRegressor": (
        "sklearn.ensemble",
        "GradientBoostingRegressor",
        "regression",
        {},
    ),
    "SklearnSVMRegressor": (
        "sklearn.svm",
        "SVR",
        "regression",
        {},
    ),
    # --- Survival (requires scikit-survival) -----------------------------
    "SklearnCoxPH": (
        "sksurv.linear_model",
        "CoxPHSurvivalAnalysis",
        "survival",
        {},
    ),
    "SklearnCoxnet": (
        "sksurv.linear_model",
        "CoxnetSurvivalAnalysis",
        "survival",
        {},
    ),
    "SklearnIPCRidge": (
        "sksurv.linear_model",
        "IPCRidge",
        "survival",
        {},
    ),
    "SklearnHingeLossSurvivalSVM": (
        "sksurv.svm",
        "HingeLossSurvivalSVM",
        "survival",
        {},
    ),
    "SklearnNaiveSurvivalSVM": (
        "sksurv.svm",
        "NaiveSurvivalSVM",
        "survival",
        {},
    ),
    "SklearnSurvivalTree": (
        "sksurv.tree",
        "SurvivalTree",
        "survival",
        {},
    ),
    "SklearnRandomSurvivalForest": (
        "sksurv.ensemble",
        "RandomSurvivalForest",
        "survival",
        {},
    ),
}

# ---------------------------------------------------------------------------
# Known model name sets (used by config validation)
# ---------------------------------------------------------------------------

SKLEARN_SLIDE_MODEL_NAMES: frozenset[str] = frozenset(SKLEARN_ESTIMATOR_CATALOG)

SLIDE_LEVEL_MODEL_NAMES: frozenset[str] = frozenset(
    {"SlideVectorMLP"} | SKLEARN_SLIDE_MODEL_NAMES
)


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------


def _fit_scaler(X: np.ndarray, model_name: str) -> Any:
    """Fit a StandardScaler and log the action.

    Args:
        X: Feature matrix shaped ``[N, D]``.
        model_name: Class name used in the log message.

    Returns:
        Fitted ``sklearn.preprocessing.StandardScaler``.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(X)
    _log.info(
        "%s: StandardScaler fitted on %d samples × %d features "
        "(mean |μ| = %.4f, mean σ = %.4f) — features will be standardized "
        "before fitting and inference.",
        model_name,
        X.shape[0],
        X.shape[1],
        float(np.abs(scaler.mean_).mean()),
        float(scaler.scale_.mean()),
    )
    return scaler


# ---------------------------------------------------------------------------
# Dynamic helpers
# ---------------------------------------------------------------------------


def list_sklearn_slide_models(task: Optional[str] = None) -> list[str]:
    """Return catalog names whose backing package is installed.

    Args:
        task: Optional filter — ``"classification"``, ``"regression"``, or
            ``"survival"``.  ``None`` returns all available models.

    Returns:
        Sorted list of PathForge model names whose underlying sklearn/sksurv
        package can be imported.
    """
    available = []
    for name, (module_path, _cls, entry_task, _fixed) in SKLEARN_ESTIMATOR_CATALOG.items():
        if task is not None and entry_task != task:
            continue
        try:
            importlib.import_module(module_path)
            available.append(name)
        except ImportError:
            pass
    return sorted(available)


def make_sklearn_slide_model(
    name: str, normalize: bool = True, **kwargs: Any
) -> ScikitBase:
    """Instantiate a slide-level sklearn/sksurv model by catalog name.

    Args:
        name: One of the keys in :data:`SKLEARN_ESTIMATOR_CATALOG`.
        normalize: When ``True`` (default), a ``StandardScaler`` is fitted on
            training data in :meth:`fit` and applied during prediction.
        kwargs: Forwarded to the underlying sklearn estimator constructor,
            overriding catalog defaults.

    Returns:
        A fitted-ready :class:`~pathforge.core.models.base.ScikitBase` instance.

    Raises:
        ValueError: If ``name`` is not in the catalog.
        ImportError: If the backing package (e.g. ``sksurv``) is not installed.
    """
    if name not in SKLEARN_ESTIMATOR_CATALOG:
        raise ValueError(
            f"Unknown sklearn slide model {name!r}. "
            f"Available: {sorted(SKLEARN_ESTIMATOR_CATALOG)}"
        )
    module_path, class_name, task, fixed_kwargs = SKLEARN_ESTIMATOR_CATALOG[name]
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import {class_name} from {module_path}. "
            "Install the required package and try again."
        ) from exc

    estimator_cls = getattr(mod, class_name)
    merged_kwargs = {**fixed_kwargs, **kwargs}
    estimator = estimator_cls(**merged_kwargs)

    if task == "classification":
        return SklearnSlideClassifier(estimator, normalize=normalize)
    elif task == "regression":
        return SklearnSlideRegressor(estimator, normalize=normalize)
    else:
        return SklearnSlideSurvival(estimator, normalize=normalize)


# ---------------------------------------------------------------------------
# Classification base
# ---------------------------------------------------------------------------


class SklearnSlideClassifier(ScikitBase):
    """Wrapper around any scikit-learn classifier for slide-level prediction.

    Features are standardized with ``StandardScaler`` before fitting and
    prediction when ``normalize=True`` (default).  A log message is emitted
    whenever normalization is applied.

    Args:
        estimator: A fitted or unfitted sklearn classifier implementing
            ``fit`` and ``predict_proba``.
        normalize: Standardize features with ``StandardScaler`` (default True).
    """

    def __init__(self, estimator: Any, normalize: bool = True) -> None:
        self._estimator = estimator
        self._normalize = normalize
        self._scaler: Any = None

    def initialize(self, config: Optional[dict[str, Any]] = None) -> None:
        pass

    def __call__(self, X: Any) -> Any:
        return self.predict_as_tensor(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnSlideClassifier":
        if self._normalize:
            self._scaler = _fit_scaler(X, type(self).__name__)
            X = self._scaler.transform(X)
        self._estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._estimator.predict_proba(X).astype(np.float32)

    def predict_as_tensor(self, X: np.ndarray) -> Any:
        """Return log-probability tensor ``(N, C)`` compatible with metrics helpers."""
        import torch

        proba = self._estimator.predict_proba(
            self._scaler.transform(X) if self._scaler is not None else X
        ).astype(np.float32)
        log_proba = np.log(np.clip(proba, 1e-7, 1.0))
        return torch.from_numpy(log_proba)

    def get_learnable_parameters(self) -> list:
        return []


# ---------------------------------------------------------------------------
# Named classification models
# ---------------------------------------------------------------------------


class SklearnLogisticRegressionClassifier(SklearnSlideClassifier):
    """Logistic regression slide-level classifier.

    Args:
        C: Inverse regularisation strength.
        max_iter: Maximum solver iterations.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sklearn.linear_model.LogisticRegression``.
    """

    def __init__(
        self, C: float = 1.0, max_iter: int = 1000, normalize: bool = True, **kwargs: Any
    ) -> None:
        from sklearn.linear_model import LogisticRegression

        super().__init__(LogisticRegression(C=C, max_iter=max_iter, **kwargs), normalize=normalize)


class SklearnRandomForestClassifier(SklearnSlideClassifier):
    """Random-forest slide-level classifier.

    Args:
        n_estimators: Number of trees.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sklearn.ensemble.RandomForestClassifier``.
    """

    def __init__(
        self, n_estimators: int = 100, normalize: bool = True, **kwargs: Any
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier

        super().__init__(
            RandomForestClassifier(n_estimators=n_estimators, **kwargs),
            normalize=normalize,
        )


class SklearnGradientBoostingClassifier(SklearnSlideClassifier):
    """Gradient boosted trees slide-level classifier.

    Args:
        n_estimators: Number of boosting rounds.
        learning_rate: Shrinkage factor applied to each tree.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sklearn.ensemble.GradientBoostingClassifier``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        from sklearn.ensemble import GradientBoostingClassifier

        super().__init__(
            GradientBoostingClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate, **kwargs
            ),
            normalize=normalize,
        )


class SklearnSVMClassifier(SklearnSlideClassifier):
    """SVM slide-level classifier with probability calibration.

    Args:
        C: Regularisation parameter.
        kernel: SVM kernel (``"rbf"``, ``"linear"``, …).
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sklearn.svm.SVC``.
    """

    def __init__(
        self, C: float = 1.0, kernel: str = "rbf", normalize: bool = True, **kwargs: Any
    ) -> None:
        from sklearn.svm import SVC

        super().__init__(
            SVC(C=C, kernel=kernel, probability=True, **kwargs),
            normalize=normalize,
        )


# ---------------------------------------------------------------------------
# Regression base
# ---------------------------------------------------------------------------


class SklearnSlideRegressor(ScikitBase):
    """Wrapper around any scikit-learn regressor for slide-level prediction.

    Features are standardized with ``StandardScaler`` before fitting and
    prediction when ``normalize=True`` (default).

    Args:
        estimator: A fitted or unfitted sklearn regressor implementing
            ``fit`` and ``predict``.
        normalize: Standardize features with ``StandardScaler`` (default True).
    """

    def __init__(self, estimator: Any, normalize: bool = True) -> None:
        self._estimator = estimator
        self._normalize = normalize
        self._scaler: Any = None

    def initialize(self, config: Optional[dict[str, Any]] = None) -> None:
        pass

    def __call__(self, X: Any) -> Any:
        return self.predict_as_tensor(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnSlideRegressor":
        if self._normalize:
            self._scaler = _fit_scaler(X, type(self).__name__)
            X = self._scaler.transform(X)
        self._estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._estimator.predict(X).astype(np.float32)

    def predict_as_tensor(self, X: np.ndarray) -> Any:
        """Return predictions as a float tensor shaped ``(N,)``."""
        import torch

        return torch.from_numpy(self.predict(X).reshape(-1))

    def get_learnable_parameters(self) -> list:
        return []


# ---------------------------------------------------------------------------
# Named regression models
# ---------------------------------------------------------------------------


class SklearnLinearRegressor(SklearnSlideRegressor):
    """Ordinary least-squares linear regression.

    Args:
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sklearn.linear_model.LinearRegression``.
    """

    def __init__(self, normalize: bool = True, **kwargs: Any) -> None:
        from sklearn.linear_model import LinearRegression

        super().__init__(LinearRegression(**kwargs), normalize=normalize)


class SklearnRidgeRegressor(SklearnSlideRegressor):
    """Ridge regression slide-level regressor.

    Args:
        alpha: Regularisation strength.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sklearn.linear_model.Ridge``.
    """

    def __init__(self, alpha: float = 1.0, normalize: bool = True, **kwargs: Any) -> None:
        from sklearn.linear_model import Ridge

        super().__init__(Ridge(alpha=alpha, **kwargs), normalize=normalize)


class SklearnElasticNetRegressor(SklearnSlideRegressor):
    """ElasticNet slide-level regressor.

    Args:
        alpha: Overall regularisation strength.
        l1_ratio: Mix between L1 (1.0) and L2 (0.0) penalties.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sklearn.linear_model.ElasticNet``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        from sklearn.linear_model import ElasticNet

        super().__init__(
            ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs),
            normalize=normalize,
        )


class SklearnRandomForestRegressor(SklearnSlideRegressor):
    """Random-forest slide-level regressor.

    Args:
        n_estimators: Number of trees.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sklearn.ensemble.RandomForestRegressor``.
    """

    def __init__(
        self, n_estimators: int = 100, normalize: bool = True, **kwargs: Any
    ) -> None:
        from sklearn.ensemble import RandomForestRegressor

        super().__init__(
            RandomForestRegressor(n_estimators=n_estimators, **kwargs),
            normalize=normalize,
        )


class SklearnGradientBoostingRegressor(SklearnSlideRegressor):
    """Gradient boosted trees slide-level regressor.

    Args:
        n_estimators: Number of boosting rounds.
        learning_rate: Shrinkage factor applied to each tree.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sklearn.ensemble.GradientBoostingRegressor``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        super().__init__(
            GradientBoostingRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate, **kwargs
            ),
            normalize=normalize,
        )


class SklearnSVMRegressor(SklearnSlideRegressor):
    """Support vector regression slide-level regressor.

    Args:
        C: Regularisation parameter.
        kernel: SVR kernel (``"rbf"``, ``"linear"``, …).
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sklearn.svm.SVR``.
    """

    def __init__(
        self, C: float = 1.0, kernel: str = "rbf", normalize: bool = True, **kwargs: Any
    ) -> None:
        from sklearn.svm import SVR

        super().__init__(SVR(C=C, kernel=kernel, **kwargs), normalize=normalize)


# ---------------------------------------------------------------------------
# Survival base (optional — requires scikit-survival)
# ---------------------------------------------------------------------------


class SklearnSlideSurvival(ScikitBase):
    """Wrapper around scikit-survival estimators for slide-level survival prediction.

    Features are standardized with ``StandardScaler`` before fitting and
    prediction when ``normalize=True`` (default).

    Args:
        estimator: A fitted or unfitted scikit-survival estimator.
        normalize: Standardize features with ``StandardScaler`` (default True).
    """

    def __init__(self, estimator: Any, normalize: bool = True) -> None:
        self._estimator = estimator
        self._normalize = normalize
        self._scaler: Any = None

    def initialize(self, config: Optional[dict[str, Any]] = None) -> None:
        pass

    def __call__(self, X: Any) -> Any:
        return self.predict_as_tensor(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnSlideSurvival":
        """Fit the survival estimator.

        Args:
            X: Feature matrix shaped ``[N, D]``.
            y: Structured array with dtype ``[('event', bool), ('time', float)]``
               as expected by scikit-survival.
        """
        if self._normalize:
            self._scaler = _fit_scaler(X, type(self).__name__)
            X = self._scaler.transform(X)
        self._estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._estimator.predict(X).astype(np.float32)

    def predict_as_tensor(self, X: np.ndarray) -> Any:
        """Return risk scores as a float tensor shaped ``(N,)``."""
        import torch

        return torch.from_numpy(self.predict(X).reshape(-1))

    def get_learnable_parameters(self) -> list:
        return []


def _require_sksurv() -> None:
    try:
        import sksurv  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "scikit-survival is required for survival sklearn models. "
            "Install it with: pip install scikit-survival"
        ) from exc


# ---------------------------------------------------------------------------
# Named survival models (all require scikit-survival)
# ---------------------------------------------------------------------------


class SklearnCoxPH(SklearnSlideSurvival):
    """Cox proportional-hazards model (L2 penalty) from scikit-survival.

    Args:
        alpha: Ridge penalty strength.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sksurv.linear_model.CoxPHSurvivalAnalysis``.
    """

    def __init__(
        self, alpha: float = 0.1, normalize: bool = True, **kwargs: Any
    ) -> None:
        _require_sksurv()
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        super().__init__(CoxPHSurvivalAnalysis(alpha=alpha, **kwargs), normalize=normalize)


class SklearnCoxnet(SklearnSlideSurvival):
    """Elastic-net penalized Cox proportional-hazards model from scikit-survival.

    Combines L1 (Lasso) and L2 (Ridge) penalties — useful for high-dimensional
    feature vectors.

    Args:
        l1_ratio: Mixing parameter between Ridge (0) and Lasso (1).
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sksurv.linear_model.CoxnetSurvivalAnalysis``.
    """

    def __init__(
        self, l1_ratio: float = 0.5, normalize: bool = True, **kwargs: Any
    ) -> None:
        _require_sksurv()
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        super().__init__(
            CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, **kwargs),
            normalize=normalize,
        )


class SklearnIPCRidge(SklearnSlideSurvival):
    """Inverse probability of censoring weighted Ridge regression for survival.

    Args:
        alpha: Ridge penalty strength.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sksurv.linear_model.IPCRidge``.
    """

    def __init__(
        self, alpha: float = 1.0, normalize: bool = True, **kwargs: Any
    ) -> None:
        _require_sksurv()
        from sksurv.linear_model import IPCRidge

        super().__init__(IPCRidge(alpha=alpha, **kwargs), normalize=normalize)


class SklearnHingeLossSurvivalSVM(SklearnSlideSurvival):
    """Ranking SVM with hinge loss for survival from scikit-survival.

    Args:
        alpha: Regularisation strength.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sksurv.svm.HingeLossSurvivalSVM``.
    """

    def __init__(
        self, alpha: float = 1.0, normalize: bool = True, **kwargs: Any
    ) -> None:
        _require_sksurv()
        from sksurv.svm import HingeLossSurvivalSVM

        super().__init__(
            HingeLossSurvivalSVM(alpha=alpha, **kwargs),
            normalize=normalize,
        )


class SklearnNaiveSurvivalSVM(SklearnSlideSurvival):
    """Naive ranking SVM for survival from scikit-survival.

    Args:
        alpha: Regularisation strength.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sksurv.svm.NaiveSurvivalSVM``.
    """

    def __init__(
        self, alpha: float = 1.0, normalize: bool = True, **kwargs: Any
    ) -> None:
        _require_sksurv()
        from sksurv.svm import NaiveSurvivalSVM

        super().__init__(
            NaiveSurvivalSVM(alpha=alpha, **kwargs),
            normalize=normalize,
        )


class SklearnSurvivalTree(SklearnSlideSurvival):
    """Survival decision tree from scikit-survival.

    Args:
        max_depth: Maximum depth of the tree.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sksurv.tree.SurvivalTree``.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        normalize: bool = True,
        **kwargs: Any,
    ) -> None:
        _require_sksurv()
        from sksurv.tree import SurvivalTree

        super().__init__(
            SurvivalTree(max_depth=max_depth, **kwargs),
            normalize=normalize,
        )


class SklearnRandomSurvivalForest(SklearnSlideSurvival):
    """Random survival forest from scikit-survival.

    Args:
        n_estimators: Number of trees.
        normalize: Standardize features before fitting (default True).
        kwargs: Forwarded to ``sksurv.ensemble.RandomSurvivalForest``.
    """

    def __init__(
        self, n_estimators: int = 100, normalize: bool = True, **kwargs: Any
    ) -> None:
        _require_sksurv()
        from sksurv.ensemble import RandomSurvivalForest

        super().__init__(
            RandomSurvivalForest(n_estimators=n_estimators, **kwargs),
            normalize=normalize,
        )


# ---------------------------------------------------------------------------
# Survival structured-array helper
# ---------------------------------------------------------------------------


def make_survival_structured_array(
    time: np.ndarray,
    event: np.ndarray,
) -> np.ndarray:
    """Build the structured array expected by scikit-survival estimators.

    Args:
        time: Survival times shaped ``[N]``.
        event: Event indicators (0/1) shaped ``[N]``.

    Returns:
        numpy structured array with dtype ``[('event', bool), ('time', float64)]``.
    """
    dtype = np.dtype([("event", bool), ("time", np.float64)])
    y = np.empty(len(time), dtype=dtype)
    y["event"] = event.astype(bool)
    y["time"] = time.astype(np.float64)
    return y
