from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pathbench.cli.infer_run as infer_cli


def test_main_executes_inference_policy_with_config_and_input_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_calls: list[Path] = []
    experiment_calls: list[object] = []
    execute_calls: list[Path] = []

    fake_cfg = SimpleNamespace(experiment=SimpleNamespace(mode="inference"))
    fake_experiment = SimpleNamespace(name="experiment")

    class _FakePolicy:
        def __init__(self, experiment: object) -> None:
            experiment_calls.append(experiment)

        def execute(self, *, input_csv: Path) -> dict[str, object]:
            execute_calls.append(input_csv)
            return {"status": "inference_done", "num_runs": 1}

    def fake_from_yaml(path: Path) -> object:
        config_calls.append(path)
        return fake_cfg

    def fake_experiment_ctor(cfg: object) -> object:
        assert cfg is fake_cfg
        return fake_experiment

    monkeypatch.setattr(infer_cli.Config, "from_yaml", fake_from_yaml)
    monkeypatch.setattr(infer_cli, "Experiment", fake_experiment_ctor)
    monkeypatch.setattr(infer_cli, "InferencePolicy", _FakePolicy)

    exit_code = infer_cli.main(
        ["--config", "configs/infer.yaml", "--input-csv", "batch.csv"]
    )

    assert exit_code == 0
    assert config_calls == [Path("configs/infer.yaml")]
    assert experiment_calls == [fake_experiment]
    assert execute_calls == [Path("batch.csv")]


def test_main_rejects_invalid_log_level() -> None:
    with pytest.raises(SystemExit) as error:
        infer_cli.main(
            [
                "--config",
                "configs/infer.yaml",
                "--input-csv",
                "batch.csv",
                "--log-level",
                "TRACE",
            ]
        )

    assert error.value.code == 2
