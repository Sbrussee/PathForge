# tests/smoke/test_benchmark_cli.py
"""Smoke tests for the pathbench-benchmark CLI entry point."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from pathbench.config.config import Config
from ._smoke_dataset import PreparedBagWorkspace


@pytest.mark.smoke
def test_benchmark_cli_importable():
    """The benchmark CLI module must be importable without side-effects."""
    from pathbench.cli import benchmark_run  # noqa: F401


@pytest.mark.smoke
def test_benchmark_cli_missing_config_exits(monkeypatch, tmp_path):
    """main() with a nonexistent config path must raise FileNotFoundError."""
    from pathbench.cli.benchmark_run import main

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
    monkeypatch,
    tmp_path: Path,
    extracted_bag_workspace: PreparedBagWorkspace,
) -> None:
    """CLI smoke run should emit the benchmark summary CSV and HTML reports.

    Uses real VarMIL training on extracted GTEx bags and real plotly for visualizations.
    """
    from pathbench.cli.benchmark_run import main
    from pathbench.cli.base import load_config
    from pathbench.core.datasets.bag_dataset import BagDataset
    from types import SimpleNamespace
    import pathbench.policy.benchmarking as bench_mod

    real_dataset = BagDataset(
        "smoke_cli_ds",
        str(extracted_bag_workspace.feature_dir),
        str(extracted_bag_workspace.metadata_csv),
        "binary_label",
    )

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
                "  num_workers: 0",
                "slide_processing:",
                "  backend: lazyslide",
                "mil:",
                "  backend: native",
                "  best_epoch_based_on: balanced_accuracy",
                "  epochs: 3",
                "  patience: 2",
                "metrics:",
                "  classification_backend: native",
                "datasets:",
                "  - name: smoke_cli_ds",
                f"    slides_dir: {slides_dir}",
                f"    artifacts_dir: {artifacts_dir}",
                "    used_for: all",
                "benchmark_parameters:",
                "  tile_px: [224]",
                "  tile_mpp: [1.0]",
                "  feature_extraction: [resnet18]",
                "  mil: [VarMIL]",
                "  loss: [CrossEntropyLoss]",
                "  batch_size: [1, 2]",
                "  seeds: [1]",
            ]
        ),
        encoding="utf-8",
    )

    # Bypass real experiment loading (no real WSIs / H5 artifacts needed for
    # benchmarking policy test — we provide the dataset via monkeypatch below).
    monkeypatch.setattr(
        "pathbench.cli.benchmark_run.Experiment",
        lambda cfg: SimpleNamespace(cfg=cfg),
    )
    monkeypatch.setattr(
        bench_mod,
        "build_bag_dataset_for_task",
        lambda *args, **kwargs: real_dataset,
    )
    monkeypatch.setattr(
        bench_mod,
        "infer_model_dimensions",
        lambda dataset: (extracted_bag_workspace.input_dim, 2),
    )

    exit_code = main(["--config", str(cfg_path)])

    assert exit_code == 0
    summary_path = project_root / "benchmark_results.csv"
    vis_dir = project_root / "benchmark_visualizations"
    assert summary_path.exists(), "benchmark_results.csv not written by CLI"
    summary_df = pd.read_csv(summary_path)
    assert {"run_index", "status", "objective_value", "rank"}.issubset(summary_df.columns)
    assert len(summary_df) == 2, "Expected one row per batch_size config"
    assert (vis_dir / "benchmark_performance_ranked.html").exists()
    assert (vis_dir / "benchmark_rank_scatter.html").exists()
