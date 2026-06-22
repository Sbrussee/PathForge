from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pathbench.cli.benchmark as benchmark_cli


def test_main_executes_benchmarking_policy_with_config_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_calls: list[object] = []
    executed_outputs: list[dict[str, object]] = []

    fake_experiment = SimpleNamespace(name="experiment")

    class _FakePolicy:
        def __init__(self, experiment: object) -> None:
            experiment_calls.append(experiment)

        def execute(self) -> dict[str, object]:
            output = {"status": "benchmark_done", "num_runs": 3}
            executed_outputs.append(output)
            return output

    def fake_load_experiment(path: Path) -> object:
        assert path == Path("configs/benchmark.yaml")
        return fake_experiment

    monkeypatch.setattr(benchmark_cli, "load_experiment", fake_load_experiment)
    monkeypatch.setattr(benchmark_cli, "BenchmarkingPolicy", _FakePolicy)

    exit_code = benchmark_cli.main(["--config", "configs/benchmark.yaml"])

    assert exit_code == 0
    assert experiment_calls == [fake_experiment]
    assert executed_outputs == [{"status": "benchmark_done", "num_runs": 3}]


def test_main_rejects_invalid_log_level() -> None:
    with pytest.raises(SystemExit) as error:
        benchmark_cli.main(["--config", "configs/benchmark.yaml", "--log-level", "TRACE"])

    assert error.value.code == 2
