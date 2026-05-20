# tests/unit/test_config_validation.py

from __future__ import annotations

import pytest
from pydantic import ValidationError

# IMPORTANT: import registries FIRST and register dummy plugins before importing Config
from pathbench.utils.registries import FEATURE_EXTRACTORS, LAZYSLIDE_MODEL_NAMES, MODELS
from pathbench.core.models.mil_base import MILModelBase


# --- Names used in tests ---
LAZY_NAME = "lazy_specific_model"
GENERIC_NAME = "generic_model"
MIL_NAME = "MockMIL"
TORCHMIL_BACKEND_NAME = "torchmil"


# --- Ensure lazyslide-only name is tracked (for backend constraint check) ---
LAZYSLIDE_MODEL_NAMES.add(LAZY_NAME)


# --- Register dummy feature extractors (idempotent) ---
if not FEATURE_EXTRACTORS.is_available(GENERIC_NAME):

    @FEATURE_EXTRACTORS.register(GENERIC_NAME)
    def _generic_model():  # pragma: no cover
        return "generic"


if not FEATURE_EXTRACTORS.is_available(LAZY_NAME):

    @FEATURE_EXTRACTORS.register(LAZY_NAME)
    def _lazy_model():  # pragma: no cover
        return "lazy"


# --- Register dummy MIL model (idempotent) ---
if not MODELS.is_available(MIL_NAME):

    @MODELS.register(MIL_NAME)
    class MockMIL(MILModelBase):
        @property
        def bag_size(self):  # pragma: no cover
            return None

        def forward_bag(self, x, **kwargs):  # pragma: no cover
            return x


if not MODELS.is_available(TORCHMIL_BACKEND_NAME):

    @MODELS.register(TORCHMIL_BACKEND_NAME)
    class MockTorchMILBackend(MILModelBase):
        @property
        def bag_size(self):  # pragma: no cover
            return None

        def forward_bag(self, x, **kwargs):  # pragma: no cover
            return x


# Sanity: ensure registration actually happened (this will make failures obvious)
assert FEATURE_EXTRACTORS.is_available(GENERIC_NAME)
assert FEATURE_EXTRACTORS.is_available(LAZY_NAME)
assert MODELS.is_available(MIL_NAME)
assert MODELS.is_available(TORCHMIL_BACKEND_NAME)


# Now import config models (after registration)
from pathbench.config.config import Config, BenchmarkParameters, SearchSpaceParameter  # noqa: E402


# --- Tests ---


def test_valid_benchmark_params():
    bp = BenchmarkParameters(
        tile_px=[256, 512],
        tile_mpp=[0.5],
        feature_extraction=[GENERIC_NAME],
        mil=[MIL_NAME],
        activation_function=["ReLU"],
        optimizer=["Adam"],
        epochs=[10, 20],
        z_dim=[128, 256],
    )
    assert bp.tile_px == [256, 512]


def test_invalid_tile_px():
    with pytest.raises(ValidationError) as excinfo:
        BenchmarkParameters(tile_px=[255])
    assert "divisible by 2" in str(excinfo.value)


def test_invalid_tile_mpp():
    with pytest.raises(ValidationError) as excinfo:
        BenchmarkParameters(tile_mpp=[0.0])
    assert "Must be > 0" in str(excinfo.value)


def test_search_space_parameter_accepts_documented_type_alias() -> None:
    spec = SearchSpaceParameter.model_validate(
        {"type": "categorical", "choices": ["a", "b"]}
    )

    assert spec.kind == "categorical"
    assert spec.choices == ["a", "b"]


def test_invalid_epochs_in_benchmark_parameters() -> None:
    with pytest.raises(ValidationError, match="Invalid epochs"):
        BenchmarkParameters(epochs=[0])


def test_backend_constraint_failure():
    cfg_data = {
        "experiment": {
            "project_name": "test",
            "annotation_file": "x",
            "task": "classification",
            "mode": "benchmark",
        },
        "slide_processing": {"backend": "openslide"},
        "mil": {"backend": "native"},
        "benchmark_parameters": {
            "feature_extraction": [LAZY_NAME],  # requires lazyslide backend
            "mil": [MIL_NAME],
        },
    }
    with pytest.raises(ValidationError) as excinfo:
        Config.model_validate(cfg_data)
    assert "requires 'lazyslide' backend" in str(excinfo.value)


def test_backend_constraint_success():
    cfg_data = {
        "experiment": {
            "project_name": "test",
            "annotation_file": "x",
            "task": "classification",
            "mode": "benchmark",
        },
        "slide_processing": {"backend": "lazyslide"},
        "mil": {"backend": "native"},
        "benchmark_parameters": {
            "feature_extraction": [LAZY_NAME],
            "mil": [MIL_NAME],
        },
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.slide_processing.backend == "lazyslide"


def test_torchmil_backend_requires_package_when_selected(monkeypatch):
    monkeypatch.setattr("pathbench.config.config.is_torchmil_available", lambda: False)
    cfg_data = {
        "experiment": {
            "project_name": "test",
            "annotation_file": "x",
            "task": "classification",
            "mode": "benchmark",
        },
        "slide_processing": {"backend": "lazyslide"},
        "mil": {"backend": "torchmil", "torchmil_model": "ABMIL"},
        "metrics": {"classification_backend": "native"},
        "benchmark_parameters": {
            "feature_extraction": [GENERIC_NAME],
            "mil": ["torchmil"],
        },
    }

    with pytest.raises(RuntimeError, match="MIL backend 'torchmil' selected"):
        Config.model_validate(cfg_data)


def test_torchmil_backend_requires_model_name_when_available(monkeypatch):
    monkeypatch.setattr("pathbench.config.config.is_torchmil_available", lambda: True)
    cfg_data = {
        "experiment": {
            "project_name": "test",
            "annotation_file": "x",
            "task": "classification",
            "mode": "benchmark",
        },
        "slide_processing": {"backend": "lazyslide"},
        "mil": {"backend": "torchmil"},
        "metrics": {"classification_backend": "native"},
        "benchmark_parameters": {
            "feature_extraction": [GENERIC_NAME],
            "mil": ["torchmil"],
        },
    }

    with pytest.raises(ValueError, match="mil.torchmil_model is required"):
        Config.model_validate(cfg_data)


def test_classification_metrics_backend_requires_torchmetrics(monkeypatch):
    monkeypatch.setattr("pathbench.config.config.is_torchmil_available", lambda: True)
    monkeypatch.setattr(
        "pathbench.config.config.is_torchmetrics_available", lambda: False
    )
    cfg_data = {
        "experiment": {
            "project_name": "test",
            "annotation_file": "x",
            "task": "classification",
            "mode": "benchmark",
        },
        "slide_processing": {"backend": "lazyslide"},
        "mil": {"backend": "torchmil", "torchmil_model": "ABMIL"},
        "benchmark_parameters": {
            "feature_extraction": [GENERIC_NAME],
            "mil": ["torchmil"],
        },
    }

    with pytest.raises(
        RuntimeError, match="Classification metrics backend requires 'torchmetrics'"
    ):
        Config.model_validate(cfg_data)


def test_native_backend_does_not_require_optional_backend_packages(monkeypatch):
    monkeypatch.setattr("pathbench.config.config.is_torchmil_available", lambda: False)
    monkeypatch.setattr(
        "pathbench.config.config.is_torchmetrics_available", lambda: False
    )
    cfg_data = {
        "experiment": {
            "project_name": "test",
            "annotation_file": "x",
            "task": "classification",
            "mode": "benchmark",
        },
        "slide_processing": {"backend": "lazyslide"},
        "mil": {"backend": "native"},
        "metrics": {"classification_backend": "native"},
        "benchmark_parameters": {
            "feature_extraction": [GENERIC_NAME],
            "mil": [MIL_NAME],
        },
    }

    cfg = Config.model_validate(cfg_data)

    assert cfg.mil.backend == "native"


def test_invalid_classification_metric_name_is_rejected() -> None:
    cfg_data = {
        "experiment": {
            "project_name": "test",
            "annotation_file": "x",
            "task": "classification",
            "mode": "benchmark",
        },
        "slide_processing": {"backend": "lazyslide"},
        "mil": {"backend": "native"},
        "metrics": {
            "classification_backend": "native",
            "classification_metrics": ["balanced_accuracy", "c_index"],
        },
        "benchmark_parameters": {
            "feature_extraction": [GENERIC_NAME],
            "mil": [MIL_NAME],
        },
    }

    with pytest.raises(ValidationError, match="Unsupported classification metrics"):
        Config.model_validate(cfg_data)


def test_survival_brier_score_metric_is_accepted() -> None:
    cfg_data = {
        "experiment": {
            "project_name": "test",
            "annotation_file": "x",
            "task": "survival",
            "mode": "benchmark",
        },
        "slide_processing": {"backend": "lazyslide"},
        "mil": {"backend": "native"},
        "metrics": {
            "classification_backend": "native",
            "survival_metrics": ["c_index", "td_auc", "brier_score"],
        },
        "benchmark_parameters": {
            "feature_extraction": [GENERIC_NAME],
            "mil": [MIL_NAME],
        },
    }

    cfg = Config.model_validate(cfg_data)

    assert "brier_score" in cfg.metrics.survival_metrics


def _base_benchmark_cfg(
    extra_experiment: dict | None = None, extra_benchmark: dict | None = None
) -> dict:
    cfg: dict = {
        "experiment": {
            "project_name": "test",
            "annotation_file": "x",
            "task": "classification",
            "mode": "benchmark",
        },
        "slide_processing": {"backend": "lazyslide"},
        "mil": {"backend": "native"},
        "metrics": {"classification_backend": "native"},
        "benchmark_parameters": {
            "feature_extraction": [GENERIC_NAME],
        },
    }
    if extra_experiment:
        cfg["experiment"].update(extra_experiment)
    if extra_benchmark:
        cfg["benchmark_parameters"].update(extra_benchmark)
    return cfg


# ---------------------------------------------------------------------------
# prediction_level="mil" constraints
# ---------------------------------------------------------------------------


def test_prediction_level_mil_requires_mil_models():
    cfg_data = _base_benchmark_cfg(
        extra_experiment={"prediction_level": "mil"},
    )
    # no mil models → should raise
    with pytest.raises((ValueError, Exception), match="requires at least one model"):
        Config.model_validate(cfg_data)


def test_prediction_level_mil_rejects_slide_level_models():
    cfg_data = _base_benchmark_cfg(
        extra_experiment={"prediction_level": "mil"},
        extra_benchmark={"mil": [MIL_NAME], "slide_level_models": ["SlideVectorMLP"]},
    )
    with pytest.raises((ValueError, Exception), match="incompatible with"):
        Config.model_validate(cfg_data)


def test_prediction_level_mil_with_mil_models_passes():
    cfg_data = _base_benchmark_cfg(
        extra_experiment={"prediction_level": "mil"},
        extra_benchmark={"mil": [MIL_NAME]},
    )
    cfg = Config.model_validate(cfg_data)
    assert cfg.experiment.prediction_level == "mil"


# ---------------------------------------------------------------------------
# prediction_level="slide" constraints
# ---------------------------------------------------------------------------


def test_prediction_level_slide_rejects_mil_models():
    cfg_data = _base_benchmark_cfg(
        extra_experiment={"prediction_level": "slide"},
        extra_benchmark={"mil": [MIL_NAME]},
    )
    with pytest.raises((ValueError, Exception), match="incompatible with"):
        Config.model_validate(cfg_data)


def test_prediction_level_slide_defaults_to_slide_vector_mlp():
    cfg_data = _base_benchmark_cfg(
        extra_experiment={"prediction_level": "slide"},
    )
    cfg = Config.model_validate(cfg_data)
    assert "SlideVectorMLP" in cfg.benchmark_parameters.slide_level_models


def test_prediction_level_slide_with_explicit_sklearn_model():
    cfg_data = _base_benchmark_cfg(
        extra_experiment={"prediction_level": "slide"},
        extra_benchmark={"slide_level_models": ["SklearnLogisticRegressionClassifier"]},
    )
    cfg = Config.model_validate(cfg_data)
    assert (
        "SklearnLogisticRegressionClassifier"
        in cfg.benchmark_parameters.slide_level_models
    )


def test_slide_level_models_rejects_unknown_model_name():
    with pytest.raises((ValueError, Exception)):
        BenchmarkParameters(
            feature_extraction=[GENERIC_NAME],
            slide_level_models=["NonExistentSlideModel"],
        )


def test_best_epoch_monitor_must_match_task_metrics() -> None:
    cfg_data = {
        "experiment": {
            "project_name": "test",
            "annotation_file": "x",
            "task": "classification",
            "mode": "benchmark",
        },
        "slide_processing": {"backend": "lazyslide"},
        "mil": {"backend": "native", "best_epoch_based_on": "c_index"},
        "metrics": {"classification_backend": "native"},
        "benchmark_parameters": {
            "feature_extraction": [GENERIC_NAME],
            "mil": [MIL_NAME],
        },
    }

    with pytest.raises(ValueError, match="mil.best_epoch_based_on must be compatible"):
        Config.model_validate(cfg_data)
