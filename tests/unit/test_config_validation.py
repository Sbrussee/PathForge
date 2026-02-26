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


# Sanity: ensure registration actually happened (this will make failures obvious)
assert FEATURE_EXTRACTORS.is_available(GENERIC_NAME)
assert FEATURE_EXTRACTORS.is_available(LAZY_NAME)
assert MODELS.is_available(MIL_NAME)


# Now import config models (after registration)
from pathbench.config.config import Config, BenchmarkParameters  # noqa: E402


# --- Tests ---

def test_valid_benchmark_params():
    bp = BenchmarkParameters(
        tile_px=[256, 512],
        tile_mpp=[0.5],
        feature_extraction=[GENERIC_NAME],
        mil=[MIL_NAME],
        activation_function=["ReLU"],
        optimizer=["Adam"],
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


def test_backend_constraint_failure():
    cfg_data = {
        "experiment": {
            "project_name": "test",
            "annotation_file": "x",
            "task": "classification",
            "mode": "benchmark",
        },
        "slide_processing": {"backend": "openslide"},
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
        "benchmark_parameters": {
            "feature_extraction": [LAZY_NAME],
            "mil": [MIL_NAME],
        },
    }
    cfg = Config.model_validate(cfg_data)
    assert cfg.slide_processing.backend == "lazyslide"
