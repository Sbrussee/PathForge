from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import pathbench.policy.benchmarking as benchmark_mod
from pathbench.core.experiments.combinations import ComboConfig
from pathbench.policy.benchmarking import BenchmarkingPolicy


class _FakeTask:
    def __init__(self) -> None:
        self.calls: list[tuple[ComboConfig, dict[str, list[object]]]] = []

    @classmethod
    def get_grid_keys(cls) -> list[str]:
        return ["feature_extraction", "tile_px", "tile_mpp", "mil"]

    def execute(self, combo_cfg: ComboConfig, datasets_by_use: dict[str, list[object]]) -> dict[str, object]:
        self.calls.append((combo_cfg, datasets_by_use))
        return {"combo": combo_cfg, "datasets_by_use": datasets_by_use}


class _FakeFeaturePolicy:
    def __init__(self, experiment: object) -> None:
        self.experiment = experiment
        self.calls: list[tuple[object, ComboConfig]] = []

    def execute_dataset(self, dataset: object, combo_cfg: ComboConfig) -> None:
        self.calls.append((dataset, combo_cfg))


def _make_experiment() -> SimpleNamespace:
    datasets = [
        SimpleNamespace(name="train_ds", used_for="training"),
        SimpleNamespace(name="ignored_ds", used_for="ignore"),
        SimpleNamespace(name="test_ds", used_for="testing"),
    ]
    cfg = SimpleNamespace(
        experiment=SimpleNamespace(task="slide_retrieval"),
        datasets=datasets,
    )
    annotations_df = pd.DataFrame({"dataset": ["train_ds", "test_ds"], "slide_id": ["S1", "S2"]})
    return SimpleNamespace(
        cfg=cfg,
        load_annotations=lambda: annotations_df,
    )


@pytest.fixture
def benchmark_policy(monkeypatch: pytest.MonkeyPatch) -> BenchmarkingPolicy:
    fake_task = _FakeTask()
    monkeypatch.setattr(benchmark_mod, "import_task_modules", lambda: None)
    monkeypatch.setattr(benchmark_mod, "build_task", lambda task_name, experiment: fake_task)
    monkeypatch.setattr(benchmark_mod, "FeatureExtractionPolicy", _FakeFeaturePolicy)
    policy = BenchmarkingPolicy(_make_experiment())
    policy.task = fake_task
    return policy


def test_benchmark_policy_does_not_build_feature_policy_eagerly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_task = _FakeTask()
    feature_policy_init_calls: list[object] = []

    monkeypatch.setattr(benchmark_mod, "import_task_modules", lambda: None)
    monkeypatch.setattr(benchmark_mod, "build_task", lambda task_name, experiment: fake_task)

    class _TrackingFeaturePolicy:
        def __init__(self, experiment: object) -> None:
            feature_policy_init_calls.append(experiment)

    monkeypatch.setattr(benchmark_mod, "FeatureExtractionPolicy", _TrackingFeaturePolicy)

    policy = BenchmarkingPolicy(_make_experiment())

    assert feature_policy_init_calls == []
    _ = policy.feature_policy
    assert len(feature_policy_init_calls) == 1


def test_group_combos_by_bag_source_groups_matching_feature_sources(
    benchmark_policy: BenchmarkingPolicy,
) -> None:
    combo_a = ComboConfig(feature_extraction="uni", tile_px=256, tile_mpp=0.5, mil="a")
    combo_b = ComboConfig(feature_extraction="uni", tile_px=256, tile_mpp=0.5, mil="b")
    combo_c = ComboConfig(feature_extraction="gigapath", tile_px=256, tile_mpp=0.5, mil="a")

    grouped = benchmark_policy._group_combos_by_bag_source([combo_a, combo_b, combo_c])

    assert set(grouped) == {"256px_0.5mpp__uni", "256px_0.5mpp__gigapath"}
    assert grouped["256px_0.5mpp__uni"] == [combo_a, combo_b]
    assert grouped["256px_0.5mpp__gigapath"] == [combo_c]


def test_ensure_bag_features_exist_extracts_only_datasets_with_missing_features(
    benchmark_policy: BenchmarkingPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    combo_cfg = ComboConfig(feature_extraction="uni", tile_px=256, tile_mpp=0.5)
    dataset_by_name = {
        ds_cfg.name: ds_cfg for ds_cfg in benchmark_policy.cfg.datasets
    }

    def fake_find_slides_with_missing_features(
        ds_cfg: object,
        annotations_df: pd.DataFrame,
        combo_cfg: ComboConfig,
    ) -> list[str]:
        assert not annotations_df.empty
        return {
            "train_ds": ["S1", "S3"],
            "ignored_ds": ["SHOULD_NOT_BE_USED"],
            "test_ds": [],
        }[ds_cfg.name]

    subset_datasets = {
        "train_ds": SimpleNamespace(name="train_subset"),
        "test_ds": SimpleNamespace(name="test_subset"),
    }

    def fake_build_wsi_dataset(
        ds_cfg: object,
        annotations_df: pd.DataFrame,
        slide_ids: list[str] | None = None,
    ) -> object:
        assert slide_ids is not None
        assert ds_cfg is dataset_by_name["train_ds"]
        assert slide_ids == ["S1", "S3"]
        return subset_datasets[ds_cfg.name]

    monkeypatch.setattr(
        benchmark_mod,
        "find_slides_with_missing_features",
        fake_find_slides_with_missing_features,
    )
    monkeypatch.setattr(benchmark_mod, "build_wsi_dataset", fake_build_wsi_dataset)

    benchmark_policy.ensure_bag_features_exist(combo_cfg=combo_cfg)

    assert benchmark_policy.feature_policy.calls == [
        (subset_datasets["train_ds"], combo_cfg),
    ]


def test_execute_combination_resolves_features_builds_datasets_and_runs_one_task(
    benchmark_policy: BenchmarkingPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_combo = ComboConfig(
        feature_extraction="uni",
        tile_px=256,
        tile_mpp=0.5,
        mil="attention_mil",
    )
    datasets_by_use = {"training": [SimpleNamespace(name="train_bag")]}
    feature_resolution_calls: list[ComboConfig] = []
    build_calls: list[ComboConfig] = []
    group_calls: list[list[object]] = []
    bag_datasets = [SimpleNamespace(name="train_bag")]

    def fake_ensure_bag_features_exist(
        *,
        combo_cfg: ComboConfig,
        annotations_df: pd.DataFrame | None = None,
    ) -> None:
        feature_resolution_calls.append(combo_cfg)
        assert annotations_df is not None
    
    def fake_build_bag_datasets_for_combo(
        *,
        combo_cfg: ComboConfig,
        annotations_df: pd.DataFrame | None = None,
    ) -> list[object]:
        build_calls.append(combo_cfg)
        assert annotations_df is not None
        return bag_datasets

    def fake_group_bag_datasets_by_use(
        bag_datasets_input: list[object],
    ) -> dict[str, list[object]]:
        group_calls.append(bag_datasets_input)
        return datasets_by_use

    monkeypatch.setattr(
        benchmark_policy,
        "ensure_bag_features_exist",
        fake_ensure_bag_features_exist,
    )
    monkeypatch.setattr(
        benchmark_policy,
        "build_bag_datasets_for_combo",
        fake_build_bag_datasets_for_combo,
    )
    monkeypatch.setattr(
        benchmark_policy,
        "group_bag_datasets_by_use",
        fake_group_bag_datasets_by_use,
    )

    output = benchmark_policy.execute_combination(full_combo)

    assert feature_resolution_calls == [full_combo]
    assert build_calls == [full_combo]
    assert group_calls == [bag_datasets]
    assert output["status"] == "benchmark_done"
    assert output["num_runs"] == 1
    assert output["task_output"]["combo"] is full_combo
    assert benchmark_policy.task.calls == [(full_combo, datasets_by_use)]


def test_build_feature_extraction_dataset_raises_runtime_error_for_missing_slides(
    benchmark_policy: BenchmarkingPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    combo_cfg = ComboConfig(feature_extraction="uni", tile_px=256, tile_mpp=0.5)
    ds_cfg = benchmark_policy.cfg.datasets[0]
    annotations_df = pd.DataFrame({"dataset": ["train_ds"], "slide_id": ["S1"]})

    def fake_build_wsi_dataset(
        ds_cfg: object,
        annotations_df: pd.DataFrame,
        slide_ids: list[str] | None = None,
    ) -> object:
        raise FileNotFoundError("slides missing on disk")

    monkeypatch.setattr(benchmark_mod, "build_wsi_dataset", fake_build_wsi_dataset)

    with pytest.raises(RuntimeError, match="Cannot continue benchmark for dataset 'train_ds'"):
        benchmark_policy._build_feature_extraction_dataset(
            ds_cfg=ds_cfg,
            annotations_df=annotations_df,
            missing_slide_ids=["S1"],
            combo_cfg=combo_cfg,
        )
