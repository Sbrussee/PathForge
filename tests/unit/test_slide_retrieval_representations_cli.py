from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import pathbench.cli.retrieval_representations as retrieval_repr_cli


def test_main_executes_representation_precompute_runner(
    monkeypatch,
) -> None:
    config_calls: list[Path] = []
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
        lambda policy, skip_missing_features=False: (
            runner_calls.append((policy, bool(skip_missing_features)))
            or executed_outputs.append({"status": "representations_done", "num_runs": 2})
            or {"status": "representations_done", "num_runs": 2}
        ),
    )

    exit_code = retrieval_repr_cli.main(["--config", "configs/benchmark.yaml"])

    assert exit_code == 0
    assert config_calls == [Path("configs/benchmark.yaml")]
    assert experiment_calls == [fake_experiment]
    assert len(runner_calls) == 1
    assert runner_calls[0][1] is False
    assert executed_outputs == [{"status": "representations_done", "num_runs": 2}]


def test_run_representation_precompute_never_triggers_feature_extraction(
    monkeypatch,
) -> None:
    combo_cfg = SimpleNamespace(name="combo")
    task = SimpleNamespace(get_grid_keys=lambda: ["feature_extraction", "tile_px", "tile_mpp"])
    annotations_df = pd.DataFrame(
        {
            "dataset": ["train_ds"],
            "slide_id": ["S1"],
        }
    )
    filtered_annotations_df = annotations_df.iloc[0:0]
    bag_dataset = SimpleNamespace(name="bag_ds")
    build_calls: list[tuple[object, object]] = []
    group_calls: list[list[object]] = []
    validation_calls: list[dict[str, list[object]]] = []
    materialize_calls: list[tuple[object, dict[str, list[object]]]] = []

    class _FakePolicy:
        def __init__(self) -> None:
            self.task = task
            self.experiment = SimpleNamespace(
                load_annotations=lambda: annotations_df,
                cfg=SimpleNamespace(),
            )

        def _group_combos_by_bag_source(self, combinations: list[object]) -> dict[str, list[object]]:
            assert combinations == [combo_cfg]
            return {"bag_source": [combo_cfg]}

        def ensure_bag_features_exist(self, **kwargs) -> None:
            raise AssertionError("representation precompute must not trigger feature extraction")

        def build_bag_datasets_for_combo(self, *, combo_cfg: object, annotations_df: object) -> list[object]:
            build_calls.append((combo_cfg, annotations_df))
            return [bag_dataset]

        def group_bag_datasets_by_use(self, bag_datasets: list[object]) -> dict[str, list[object]]:
            group_calls.append(bag_datasets)
            return {"reference": bag_datasets}

        def _validate_dataset_uses(self, *, datasets_by_use: dict[str, list[object]]) -> None:
            validation_calls.append(datasets_by_use)

    policy = _FakePolicy()

    monkeypatch.setattr(retrieval_repr_cli, "SlideRetrievalTask", type(task))
    monkeypatch.setattr(retrieval_repr_cli, "build_combinations", lambda **kwargs: [combo_cfg])
    monkeypatch.setattr(
        retrieval_repr_cli,
        "_filter_annotations_with_existing_features",
        lambda **kwargs: filtered_annotations_df,
    )
    monkeypatch.setattr(
        retrieval_repr_cli,
        "_materialize_representations_for_combo",
        lambda *, task, combo_cfg, datasets_by_use: materialize_calls.append(
            (combo_cfg, datasets_by_use)
        ),
    )

    output = retrieval_repr_cli._run_representation_precompute(policy)

    assert output == {"status": "representations_done", "num_runs": 1}
    assert build_calls == [(combo_cfg, filtered_annotations_df)]
    assert group_calls == [[bag_dataset]]
    assert validation_calls == [{"reference": [bag_dataset]}]
    assert materialize_calls == [(combo_cfg, {"reference": [bag_dataset]})]
