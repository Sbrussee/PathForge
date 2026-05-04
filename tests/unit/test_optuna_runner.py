"""Unit tests for the lightweight Optuna helper module."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from pathbench.optimization.optuna_runner import build_study, suggest_params


def test_build_study_returns_maximize_study() -> None:
    """The default helper should construct a maximize study."""
    cfg = SimpleNamespace()

    study = build_study(cfg)

    assert isinstance(study, optuna.study.Study)
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE


def test_suggest_params_supports_float_int_and_categorical(tmp_path: Path) -> None:
    """Search-space suggestions should mirror the JSON spec."""
    search_space = {
        "lr": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
        "depth": {"type": "int", "low": 1, "high": 3, "step": 1},
        "model": {"type": "categorical", "choices": ["a", "b"]},
    }
    search_space_path = tmp_path / "search_space.json"
    search_space_path.write_text(json.dumps(search_space), encoding="utf-8")

    study = optuna.create_study(direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, str(search_space_path))
        assert 1e-4 <= params["lr"] <= 1e-3
        assert params["depth"] in {1, 2, 3}
        assert params["model"] in {"a", "b"}
        return 1.0

    study.optimize(objective, n_trials=1)

    assert len(study.trials) == 1


def test_suggest_params_rejects_unknown_parameter_types(tmp_path: Path) -> None:
    """Unknown search-space parameter types should fail clearly."""
    search_space_path = tmp_path / "bad_search_space.json"
    search_space_path.write_text(
        json.dumps({"bad": {"type": "matrix", "low": 0, "high": 1}}),
        encoding="utf-8",
    )
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    with pytest.raises(ValueError, match="Unknown type"):
        suggest_params(trial, str(search_space_path))
