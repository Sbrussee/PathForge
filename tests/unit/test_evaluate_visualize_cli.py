from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pathbench.cli.evaluate as evaluate_cli
import pathbench.cli.visualize as visualize_cli


def test_evaluate_main_executes_evaluation_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate_calls: list[object] = []

    fake_experiment = SimpleNamespace(name="experiment")

    class _FakeOrchestrator:
        def __init__(self, experiment: object) -> None:
            evaluate_calls.append(experiment)

        def evaluate(self) -> None:
            return None

    def fake_load_experiment(path: Path) -> object:
        assert path == Path("configs/evaluate.yaml")
        return fake_experiment

    monkeypatch.setattr(evaluate_cli, "load_experiment", fake_load_experiment)
    monkeypatch.setattr(evaluate_cli, "EvaluationOrchestrator", _FakeOrchestrator)

    exit_code = evaluate_cli.main(["--config", "configs/evaluate.yaml"])

    assert exit_code == 0
    assert evaluate_calls == [fake_experiment]


def test_visualize_main_executes_visualization_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visualize_calls: list[object] = []

    fake_experiment = SimpleNamespace(name="experiment")

    class _FakeOrchestrator:
        def __init__(self, experiment: object) -> None:
            visualize_calls.append(experiment)

        def visualize(self) -> None:
            return None

    def fake_load_experiment(path: Path) -> object:
        assert path == Path("configs/visualize.yaml")
        return fake_experiment

    monkeypatch.setattr(visualize_cli, "load_experiment", fake_load_experiment)
    monkeypatch.setattr(visualize_cli, "VisualizationOrchestrator", _FakeOrchestrator)

    exit_code = visualize_cli.main(["--config", "configs/visualize.yaml"])

    assert exit_code == 0
    assert visualize_calls == [fake_experiment]


def test_visualize_main_rejects_invalid_log_level() -> None:
    with pytest.raises(SystemExit) as error:
        visualize_cli.main(["--config", "configs/visualize.yaml", "--log-level", "TRACE"])

    assert error.value.code == 2
