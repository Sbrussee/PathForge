"""Shared fixtures for PathForge tests.

Registers lightweight dummy plugins once so every test module can rely on
a known feature-extractor name (_DUMMY_FE) and a known MIL model name
(_DUMMY_MIL) without downloading or instantiating real models.
"""
from __future__ import annotations

import pytest

from pathforge.utils.registries import FEATURE_EXTRACTORS, MODELS
from pathforge.core.models.mil_base import MILModelBase

# ---------------------------------------------------------------------------
# Dummy plugin names exposed to tests
# ---------------------------------------------------------------------------
DUMMY_FE = "dummy_fe"
DUMMY_MIL = "DummyMIL"

if not FEATURE_EXTRACTORS.is_available(DUMMY_FE):
    @FEATURE_EXTRACTORS.register(DUMMY_FE)
    def _dummy_fe():  # pragma: no cover
        return DUMMY_FE

if not MODELS.is_available(DUMMY_MIL):
    @MODELS.register(DUMMY_MIL)
    class _DummyMIL(MILModelBase):
        @property
        def bag_size(self):  # pragma: no cover
            return None

        def forward_bag(self, x, **kwargs):  # pragma: no cover
            return x


# ---------------------------------------------------------------------------
# Config dict fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_fe_config(tmp_path):
    """Minimal feature-extraction Config dict with a writable project root."""
    ann = tmp_path / "annotations.csv"
    ann.write_text("dataset,slide,patient,category\n", encoding="utf-8")
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    return {
        "experiment": {
            "project_name": "test_fe",
            "annotation_file": str(ann),
            "project_root": str(tmp_path / "project"),
            "mode": "feature_extraction",
        },
        "slide_processing": {"backend": "lazyslide"},
        "datasets": [
            {
                "name": "test_ds",
                "slides_dir": str(slides_dir),
                "artifacts_dir": str(tmp_path / "artifacts"),
                "used_for": "all",
            }
        ],
        "benchmark_parameters": {
            "tile_px": [256],
            "tile_mpp": [0.5],
            "feature_extraction": [DUMMY_FE],
            "mil": [],
        },
    }


@pytest.fixture()
def minimal_benchmark_config(tmp_path):
    """Minimal benchmark Config dict using native backend and dummy plugins."""
    ann = tmp_path / "annotations.csv"
    ann.write_text("dataset,slide,patient,category\n", encoding="utf-8")
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    return {
        "experiment": {
            "project_name": "test_benchmark",
            "annotation_file": str(ann),
            "project_root": str(tmp_path / "project"),
            "mode": "benchmark",
            "task": "classification",
        },
        "slide_processing": {"backend": "lazyslide"},
        "mil": {"backend": "native"},
        "metrics": {"classification_backend": "native"},
        "datasets": [
            {
                "name": "test_ds",
                "slides_dir": str(slides_dir),
                "artifacts_dir": str(tmp_path / "artifacts"),
                "used_for": "all",
            }
        ],
        "benchmark_parameters": {
            "tile_px": [256],
            "tile_mpp": [0.5],
            "feature_extraction": [DUMMY_FE],
            "mil": [DUMMY_MIL],
            "loss": ["CrossEntropyLoss"],
        },
    }
