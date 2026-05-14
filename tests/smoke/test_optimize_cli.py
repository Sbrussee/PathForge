"""Smoke tests for the pathbench-optimize CLI entry point."""

from __future__ import annotations

import pytest

from pathbench.config.config import Config
from tests.conftest import DUMMY_FE


@pytest.mark.smoke
def test_optimize_cli_importable() -> None:
    """The optimization CLI module must be importable without side-effects."""
    from pathbench.cli import optimize  # noqa: F401


@pytest.mark.smoke
def test_optimize_cli_missing_config_exits(tmp_path) -> None:
    """main() with a nonexistent config path must raise FileNotFoundError."""
    from pathbench.cli.optimize import main

    with pytest.raises(FileNotFoundError):
        main(["--config", str(tmp_path / "missing.yaml")])


@pytest.mark.smoke
def test_optimization_config_validates(tmp_path) -> None:
    """A minimal optimization config dict round-trips through Config validation."""
    ann = tmp_path / "annotations.csv"
    ann.write_text("dataset,slide,patient,category\n", encoding="utf-8")
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()

    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "test_optimization",
                "annotation_file": str(ann),
                "project_root": str(tmp_path / "project"),
                "mode": "optimization",
                "task": "classification",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {"backend": "native"},
            "metrics": {"classification_backend": "native"},
            "optimization": {
                "study_name": "smoke_opt",
                "sampler": "TPESampler",
                "pruner": "MedianPruner",
                "objective_mode": "max",
                "objective_metric": "balanced_accuracy",
                "trials": 1,
            },
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
                "mil": ["DummyMIL"],
                "loss": ["CrossEntropyLoss"],
            },
        }
    )

    assert cfg.experiment.mode == "optimization"
    assert cfg.optimization.study_name == "smoke_opt"
