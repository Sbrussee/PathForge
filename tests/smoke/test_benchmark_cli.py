# tests/smoke/test_benchmark_cli.py
"""Smoke tests for the pathbench-benchmark CLI entry point."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from pathbench.config.config import Config


@pytest.mark.smoke
def test_benchmark_cli_importable():
    """The benchmark CLI module must be importable without side-effects."""
    from pathbench.cli import benchmark  # noqa: F401


@pytest.mark.smoke
def test_benchmark_cli_missing_config_exits(monkeypatch, tmp_path):
    """main() with a nonexistent config path must raise FileNotFoundError."""
    from pathbench.cli.benchmark import main

    monkeypatch.setattr(
        sys, "argv", ["pathbench-benchmark", "--config", str(tmp_path / "missing.yaml")]
    )
    with pytest.raises(FileNotFoundError):
        main()


@pytest.mark.smoke
def test_benchmark_config_validates(minimal_benchmark_config):
    """A minimal benchmark config dict round-trips through Config validation."""
    cfg = Config.model_validate(minimal_benchmark_config)
    assert cfg.experiment.mode == "benchmark"
    assert cfg.experiment.task == "classification"
    assert cfg.mil.backend == "native"


@pytest.mark.smoke
def test_benchmark_cli_writes_summary_and_visualizations(
    monkeypatch, tmp_path: Path
) -> None:
    """CLI smoke run should emit the benchmark summary CSV and HTML reports."""
    from pathbench.cli.benchmark import main
    import pathbench.policy.benchmarking as bench_mod
    import pathbench.policy.utils as policy_utils

    class _FakeDataset:
        feature_dim = 8

        def output_dim(self) -> int:
            return 2

    class _FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def fit(self, model, ds_train, ds_val, loss_fn):
            _ = (model, ds_train, ds_val, loss_fn)
            score = 0.9 if self.cfg.mil.batch_size == 1 else 0.6
            return f"batch_{self.cfg.mil.batch_size}.ckpt", score

    class _FakeFigure:
        def update_layout(self, **kwargs):
            _ = kwargs

        def write_html(self, path: str) -> None:
            Path(path).write_text("<html></html>", encoding="utf-8")

    class _FakePX:
        def bar(self, *args, **kwargs):
            _ = (args, kwargs)
            return _FakeFigure()

        def scatter(self, *args, **kwargs):
            _ = (args, kwargs)
            return _FakeFigure()

    cfg_path = tmp_path / "benchmark.yaml"
    project_root = (tmp_path / "project").resolve()
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    artifacts_dir = tmp_path / "artifacts"
    cfg_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  project_name: smoke_benchmark_cli",
                "  annotation_file: x",
                f"  project_root: {project_root}",
                "  mode: benchmark",
                "  task: classification",
                "slide_processing:",
                "  backend: lazyslide",
                "mil:",
                "  backend: native",
                "  best_epoch_based_on: balanced_accuracy",
                "metrics:",
                "  classification_backend: native",
                "datasets:",
                "  - name: smoke_ds",
                f"    slides_dir: {slides_dir}",
                f"    artifacts_dir: {artifacts_dir}",
                "    used_for: all",
                "benchmark_parameters:",
                "  tile_px: [256]",
                "  tile_mpp: [0.5]",
                "  feature_extraction: [resnet18]",
                "  mil: [DummyMIL]",
                "  loss: [CrossEntropyLoss]",
                "  batch_size: [1, 4]",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(bench_mod, "build_bag_dataset_for_task", lambda *args, **kwargs: _FakeDataset())
    monkeypatch.setattr(bench_mod, "resolve_dataset_feature_dir", lambda dataset_entry: tmp_path)
    monkeypatch.setattr(bench_mod, "infer_model_dimensions", lambda dataset: (dataset.feature_dim, dataset.output_dim()))
    monkeypatch.setattr(bench_mod.LOSSES, "get", lambda name: (lambda: object()))
    monkeypatch.setattr(bench_mod.TRAINERS, "get", lambda name: _FakeTrainer)
    monkeypatch.setattr(policy_utils, "_load_plotly_modules", lambda: (_FakePX(), object()))

    exit_code = main(["--config", str(cfg_path)])

    assert exit_code == 0
    summary_path = project_root / "benchmark_results.csv"
    vis_dir = project_root / "benchmark_visualizations"
    assert summary_path.exists()
    summary_df = pd.read_csv(summary_path)
    assert set(summary_df["objective_value"].dropna().tolist()) == {0.6, 0.9}
    assert (vis_dir / "benchmark_performance_ranked.html").exists()
    assert (vis_dir / "benchmark_rank_scatter.html").exists()
