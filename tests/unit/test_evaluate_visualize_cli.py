from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pathforge.cli.evaluate_run as evaluate_cli
import pathforge.cli.visualize_run as visualize_cli


def test_evaluate_main_executes_evaluation_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate_calls: list[object] = []

    fake_cfg = SimpleNamespace(name="config")
    fake_experiment = SimpleNamespace(name="experiment")

    class _FakeOrchestrator:
        def __init__(self, experiment: object) -> None:
            evaluate_calls.append(experiment)

        def evaluate(self) -> None:
            return None

    def fake_from_yaml(path: Path) -> object:
        assert Path(path) == Path("configs/evaluate.yaml")
        return fake_cfg

    monkeypatch.setattr(evaluate_cli.Config, "from_yaml", staticmethod(fake_from_yaml))
    monkeypatch.setattr(evaluate_cli, "Experiment", lambda cfg: fake_experiment)
    monkeypatch.setattr(evaluate_cli, "EvaluationOrchestrator", _FakeOrchestrator)

    exit_code = evaluate_cli.main(["--config", "configs/evaluate.yaml"])

    assert exit_code == 0
    assert evaluate_calls == [fake_experiment]


def test_visualize_main_executes_visualization_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visualize_calls: list[object] = []

    fake_cfg = SimpleNamespace(name="config")
    fake_experiment = SimpleNamespace(name="experiment")

    class _FakeOrchestrator:
        def __init__(self, experiment: object) -> None:
            visualize_calls.append(experiment)

        def visualize(self) -> dict[str, object]:
            return {"status": "ok", "num_runs": 0, "created_files": []}

    def fake_from_yaml(path: Path) -> object:
        assert Path(path) == Path("configs/visualize.yaml")
        return fake_cfg

    monkeypatch.setattr(visualize_cli.Config, "from_yaml", staticmethod(fake_from_yaml))
    monkeypatch.setattr(visualize_cli, "Experiment", lambda cfg: fake_experiment)
    monkeypatch.setattr(visualize_cli, "VisualizationOrchestrator", _FakeOrchestrator)

    exit_code = visualize_cli.main(["--config", "configs/visualize.yaml"])

    assert exit_code == 0
    assert visualize_calls == [fake_experiment]


def test_visualize_main_rejects_invalid_log_level() -> None:
    with pytest.raises(SystemExit) as error:
        visualize_cli.main(["--config", "configs/visualize.yaml", "--log-level", "TRACE"])

    assert error.value.code == 2


def test_visualize_summary_command_renders_saved_global_csv(tmp_path: Path) -> None:
    """Saved benchmark/optimization summaries remain visualizable afterwards."""

    from typer.testing import CliRunner

    from pathforge.cli.app import app

    summary = tmp_path / "benchmark_results.csv"
    summary.write_text(
        "status,objective_metric,objective_value,rank,model,loss\n"
        "success,balanced_accuracy,0.8,1,ABMIL,CrossEntropyLoss\n"
        "success,balanced_accuracy,0.7,2,DSMIL,CrossEntropyLoss\n",
        encoding="utf-8",
    )
    output = tmp_path / "charts"

    result = CliRunner().invoke(
        app,
        [
            "visualize",
            "summary",
            "--input",
            str(summary),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "global_results_ranked.html").is_file()
    assert (output / "global_results_rank_scatter.html").is_file()


def test_visualize_summary_rejects_csv_without_global_contract(tmp_path: Path) -> None:
    """A non-PathForge CSV fails with a precise missing-column error."""

    from pathforge.policy.utils import save_global_summary_visualizations

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("metric,value\naccuracy,0.8\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        save_global_summary_visualizations(invalid, output_dir=tmp_path / "charts")
