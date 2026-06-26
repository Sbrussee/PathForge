"""Smoke tests for the pathbench-optimize CLI entry point."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pathbench.config.config import Config
from tests.conftest import DUMMY_FE


@pytest.mark.smoke
def test_optimize_cli_importable() -> None:
    """The optimization CLI module must be importable without side-effects."""
    from pathbench.cli import optimize_run  # noqa: F401


@pytest.mark.smoke
def test_optimize_cli_missing_config_exits(tmp_path) -> None:
    """main() with a nonexistent config path must raise FileNotFoundError."""
    from pathbench.cli.optimize_run import main

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


@pytest.mark.smoke
def test_optimize_cli_writes_summary_and_visualizations(
    monkeypatch, tmp_path: Path
) -> None:
    """CLI smoke run should emit optimization CSV summaries and visual reports."""
    from pathbench.cli.optimize_run import main
    import pathbench.policy.optimization as opt_mod

    cfg_path = tmp_path / "optimize.yaml"
    project_root = (tmp_path / "project").resolve()
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    annotation_file = tmp_path / "annotations.csv"
    annotation_file.write_text("dataset,slide,patient,category\n", encoding="utf-8")
    cfg_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  project_name: smoke_optimize_cli",
                f"  annotation_file: {annotation_file}",
                f"  project_root: {project_root}",
                "  mode: optimization",
                "  task: classification",
                "slide_processing:",
                "  backend: lazyslide",
                "mil:",
                "  backend: native",
                "metrics:",
                "  classification_backend: native",
                "optimization:",
                "  study_name: smoke_opt",
                "  sampler: TPESampler",
                "  pruner: MedianPruner",
                "  objective_mode: max",
                "  objective_metric: balanced_accuracy",
                "  trials: 1",
                "datasets:",
                "  - name: smoke_ds",
                f"    slides_dir: {slides_dir}",
                f"    artifacts_dir: {tmp_path / 'artifacts'}",
                "    used_for: all",
                "benchmark_parameters:",
                "  tile_px: [256]",
                "  tile_mpp: [0.5]",
                f"  feature_extraction: [{DUMMY_FE}]",
                "  mil: [DummyMIL]",
                "  loss: [CrossEntropyLoss]",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(opt_mod.OptimizationPolicy, "objective", lambda self, trial: 0.5)

    exit_code = main(["--config", str(cfg_path)])

    assert exit_code == 0
    raw_results = project_root / "smoke_opt_results.csv"
    summary_results = project_root / "optimization_results.csv"
    vis_dir = project_root / "optimization_visualizations"
    assert raw_results.exists()
    assert summary_results.exists()
    summary_df = pd.read_csv(summary_results)
    assert len(summary_df) == 1
    assert summary_df["objective_value"].iloc[0] == pytest.approx(0.5)
    assert (vis_dir / "plot_optimization_history.html").exists()
    assert (vis_dir / "plot_optimization_history.png").exists()
