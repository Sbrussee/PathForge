from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pathbench.cli.slide_retrieval_representations as retrieval_repr_cli


def test_main_executes_representation_precompute_runner(
    monkeypatch,
) -> None:
    experiment_calls: list[object] = []
    runner_calls: list[tuple[object, bool]] = []
    executed_outputs: list[dict[str, object]] = []

    fake_cfg = SimpleNamespace(
        experiment=SimpleNamespace(mode="benchmark", task="slide_retrieval"),
    )
    fake_experiment = SimpleNamespace(name="experiment")

    class _FakePolicy:
        def __init__(self, experiment: object) -> None:
            experiment_calls.append(experiment)

    def fake_load_experiment(path: Path) -> object:
        assert path == Path("configs/benchmark.yaml")
        return fake_experiment

    fake_experiment.cfg = fake_cfg
    monkeypatch.setattr(retrieval_repr_cli, "load_experiment", fake_load_experiment)
    monkeypatch.setattr(retrieval_repr_cli, "BenchmarkingPolicy", _FakePolicy)
    monkeypatch.setattr(
        retrieval_repr_cli,
        "_run_representation_precompute",
        lambda policy, skip_missing_features=False: (
            runner_calls.append((policy, bool(skip_missing_features)))
            or executed_outputs.append({"status": "representations_done", "num_runs": 2})
            or {"status": "representations_done", "num_runs": 2}
        ),
    )

    exit_code = retrieval_repr_cli.main(["--config", "configs/benchmark.yaml"])

    assert exit_code == 0
    assert experiment_calls == [fake_experiment]
    assert len(runner_calls) == 1
    assert runner_calls[0][1] is False
    assert executed_outputs == [{"status": "representations_done", "num_runs": 2}]


def test_main_forwards_skip_missing_features_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cfg = SimpleNamespace(
        experiment=SimpleNamespace(mode="benchmark", task="slide_retrieval"),
    )
    fake_experiment = SimpleNamespace(name="experiment")
    runner_calls: list[tuple[object, bool]] = []

    class _FakePolicy:
        def __init__(self, experiment: object) -> None:
            self.experiment = experiment

    fake_experiment.cfg = fake_cfg
    monkeypatch.setattr(retrieval_repr_cli, "load_experiment", lambda path: fake_experiment)
    monkeypatch.setattr(retrieval_repr_cli, "BenchmarkingPolicy", _FakePolicy)
    monkeypatch.setattr(
        retrieval_repr_cli,
        "_run_representation_precompute",
        lambda policy, skip_missing_features=False: (
            runner_calls.append((policy, bool(skip_missing_features)))
            or {"status": "representations_done", "num_runs": 1}
        ),
    )

    exit_code = retrieval_repr_cli.main(
        ["--config", "configs/benchmark.yaml", "--skip-missing-features"]
    )

    assert exit_code == 0
    assert len(runner_calls) == 1
    assert runner_calls[0][1] is True


def test_main_rejects_non_benchmark_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cfg = SimpleNamespace(
        experiment=SimpleNamespace(mode="feature_extraction", task="slide_retrieval"),
    )
    fake_experiment = SimpleNamespace(cfg=fake_cfg)
    monkeypatch.setattr(retrieval_repr_cli, "load_experiment", lambda path: fake_experiment)

    with pytest.raises(ValueError, match="experiment.mode='benchmark'"):
        retrieval_repr_cli.main(["--config", "configs/benchmark.yaml"])


def test_main_rejects_non_retrieval_task(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cfg = SimpleNamespace(
        experiment=SimpleNamespace(mode="benchmark", task="classification"),
    )
    fake_experiment = SimpleNamespace(cfg=fake_cfg)
    monkeypatch.setattr(retrieval_repr_cli, "load_experiment", lambda path: fake_experiment)

    with pytest.raises(ValueError, match="experiment.task='slide_retrieval'"):
        retrieval_repr_cli.main(["--config", "configs/benchmark.yaml"])
