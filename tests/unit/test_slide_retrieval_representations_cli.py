from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pathbench.cli.slide_retrieval_representations as retrieval_repr_cli


def test_main_executes_representation_precompute_runner(
    monkeypatch,
) -> None:
    config_calls: list[Path] = []
    experiment_calls: list[object] = []
    runner_calls: list[object] = []
    executed_outputs: list[dict[str, object]] = []

    fake_cfg = SimpleNamespace(
        experiment=SimpleNamespace(mode="benchmark", task="slide_retrieval"),
    )
    fake_experiment = SimpleNamespace(name="experiment")

    class _FakePolicy:
        def __init__(self, experiment: object) -> None:
            experiment_calls.append(experiment)

    def fake_from_yaml(path: Path) -> object:
        config_calls.append(path)
        return fake_cfg

    def fake_experiment_ctor(cfg: object) -> object:
        assert cfg is fake_cfg
        return fake_experiment

    monkeypatch.setattr(retrieval_repr_cli.Config, "from_yaml", fake_from_yaml)
    monkeypatch.setattr(retrieval_repr_cli, "Experiment", fake_experiment_ctor)
    monkeypatch.setattr(retrieval_repr_cli, "BenchmarkingPolicy", _FakePolicy)
    monkeypatch.setattr(
        retrieval_repr_cli,
        "_run_representation_precompute",
        lambda policy: (
            runner_calls.append(policy)
            or executed_outputs.append({"status": "representations_done", "num_runs": 2})
            or {"status": "representations_done", "num_runs": 2}
        ),
    )

    exit_code = retrieval_repr_cli.main(["--config", "configs/benchmark.yaml"])

    assert exit_code == 0
    assert config_calls == [Path("configs/benchmark.yaml")]
    assert experiment_calls == [fake_experiment]
    assert len(runner_calls) == 1
    assert executed_outputs == [{"status": "representations_done", "num_runs": 2}]
