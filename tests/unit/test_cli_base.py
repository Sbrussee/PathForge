from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

import pathbench.cli.base as cli_base


def test_add_shared_arguments_parses_config_and_log_level() -> None:
    parser = argparse.ArgumentParser()
    cli_base.add_config_argument(parser)
    cli_base.add_log_level_argument(parser)

    args = parser.parse_args(["--config", "config.yaml", "--log-level", "DEBUG"])

    assert args.config == Path("config.yaml")
    assert args.log_level == "DEBUG"


def test_load_config_delegates_to_config_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    fake_cfg = SimpleNamespace(name="cfg")

    def fake_from_yaml(path: Path) -> object:
        calls.append(path)
        return fake_cfg

    monkeypatch.setattr(cli_base.Config, "from_yaml", fake_from_yaml)

    result = cli_base.load_config("configs/example.yaml")

    assert result is fake_cfg
    assert calls == [Path("configs/example.yaml")]


def test_load_experiment_builds_experiment_from_loaded_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cfg = SimpleNamespace(name="cfg")
    fake_experiment = SimpleNamespace(name="experiment")
    load_calls: list[Path] = []

    monkeypatch.setattr(
        cli_base,
        "load_config",
        lambda path: (load_calls.append(Path(path)) or fake_cfg),
    )
    monkeypatch.setattr(cli_base, "Experiment", lambda cfg: fake_experiment)

    result = cli_base.load_experiment("configs/example.yaml")

    assert result is fake_experiment
    assert load_calls == [Path("configs/example.yaml")]
