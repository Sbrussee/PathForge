from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pathbench.cli.benchmark as benchmark_cli


def test_main_executes_benchmarking_policy_with_config_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_calls: list[Path] = []
    experiment_calls: list[object] = []
    executed_outputs: list[dict[str, object]] = []

    fake_cfg = SimpleNamespace(name="cfg")
    fake_experiment = SimpleNamespace(name="experiment")

    class _FakePolicy:
        def __init__(self, experiment: object) -> None:
            experiment_calls.append(experiment)

        def execute(self) -> dict[str, object]:
            output = {"status": "benchmark_done", "num_runs": 3}
            executed_outputs.append(output)
            return output

    def fake_from_yaml(path: Path) -> object:
        config_calls.append(path)
        return fake_cfg

    def fake_experiment_ctor(cfg: object) -> object:
        assert cfg is fake_cfg
        return fake_experiment

    monkeypatch.setattr(benchmark_cli.Config, "from_yaml", fake_from_yaml)
    monkeypatch.setattr(benchmark_cli, "Experiment", fake_experiment_ctor)
    monkeypatch.setattr(benchmark_cli, "BenchmarkingPolicy", _FakePolicy)

    exit_code = benchmark_cli.main(["--config", "configs/benchmark.yaml"])

    assert exit_code == 0
    assert config_calls == [Path("configs/benchmark.yaml")]
    assert experiment_calls == [fake_experiment]
    assert executed_outputs == [{"status": "benchmark_done", "num_runs": 3}]


def test_main_rejects_invalid_log_level() -> None:
    with pytest.raises(SystemExit) as error:
        benchmark_cli.main(["--config", "configs/benchmark.yaml", "--log-level", "TRACE"])

    assert error.value.code == 2
