# tests/smoke/test_benchmark_cli.py
"""Smoke tests for the pathbench-benchmark CLI entry point."""

from __future__ import annotations

import sys
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
