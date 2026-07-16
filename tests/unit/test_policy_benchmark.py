from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import pathforge.policy.benchmarking as benchmark_mod
import pathforge.policy.utils as policy_utils
from pathforge.config.config import Config
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.models.mil_base import MILModelBase
from pathforge.policy.benchmarking import BenchmarkingPolicy
from pathforge.policy.utils import (
    apply_search_params,
    build_mil_model_for_config,
    write_experiment_summary_csv,
)
from pathforge.utils.registries import MODELS
from tests.conftest import DUMMY_FE, DUMMY_MIL

SECOND_DUMMY_MIL = "DummyMIL2"

if not MODELS.is_available(SECOND_DUMMY_MIL):

    @MODELS.register(SECOND_DUMMY_MIL)
    class _DummyMIL2(MILModelBase):
        @property
        def bag_size(self):
            return None

        def forward_bag(self, x, **kwargs):
            return x


# ---------------------------------------------------------------------------
# Legacy SimpleNamespace / Experiment-based tests (HEAD)
# ---------------------------------------------------------------------------


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


def test_experiment_benchmark_writes_tutorial_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The production Experiment path writes the summary used by the tutorial."""

    config = Config.model_validate(
        {
            "experiment": {
                "project_name": "tutorial",
                "annotation_file": str(tmp_path / "annotations.csv"),
                "project_root": str(tmp_path),
                "mode": "benchmark",
                "task": "classification",
            },
            "metrics": {"classification_backend": "native"},
            "datasets": [
                {
                    "name": "train",
                    "slides_dir": str(tmp_path / "slides"),
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "used_for": "all",
                }
            ],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": [DUMMY_FE],
                "mil": [DUMMY_MIL],
                "loss": ["CrossEntropyLoss"],
            },
        }
    )
    combo = ComboConfig(
        feature_extraction=DUMMY_FE,
        tile_px=256,
        tile_mpp=0.5,
        mil=DUMMY_MIL,
        loss="CrossEntropyLoss",
    )

    class SuccessfulTask:
        @classmethod
        def get_grid_keys(cls) -> list[str]:
            return ["feature_extraction", "tile_px", "tile_mpp", "mil", "loss"]

        def execute(self, **_: object) -> dict[str, object]:
            return {
                "status": "success",
                "checkpoint_path": str(tmp_path / "best.ckpt"),
                "objective_value": 0.25,
            }

    experiment = SimpleNamespace(
        cfg=config,
        project_root=str(tmp_path / "tutorial"),
        load_annotations=lambda: pd.DataFrame({"slide": ["S1"]}),
    )
    monkeypatch.setattr(benchmark_mod, "import_task_modules", lambda: None)
    monkeypatch.setattr(
        benchmark_mod,
        "build_task",
        lambda task_name, experiment: SuccessfulTask(),
    )
    monkeypatch.setattr(benchmark_mod, "build_combinations", lambda **kwargs: [combo])
    policy = BenchmarkingPolicy(experiment)
    monkeypatch.setattr(policy, "ensure_bag_features_exist", lambda **kwargs: None)
    monkeypatch.setattr(policy, "build_bag_datasets_for_combo", lambda **kwargs: [])
    monkeypatch.setattr(policy, "group_bag_datasets_by_use", lambda datasets: {"all": []})
    monkeypatch.setattr(policy, "_validate_dataset_uses", lambda **kwargs: None)

    result = policy.execute()

    summary = pd.read_csv(tmp_path / "tutorial" / "benchmark_results.csv")
    assert result == {"status": "benchmark_done", "num_runs": 1}
    assert summary.loc[0, "status"] == "success"
    assert summary.loc[0, "checkpoint_path"].endswith("best.ckpt")
    assert summary.loc[0, "objective_value"] == pytest.approx(0.25)


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


# ---------------------------------------------------------------------------
# Config-based tests (mil)
# ---------------------------------------------------------------------------


def test_benchmark_grid_generation():
    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "test",
                "annotation_file": "x",
                "task": "classification",
                "mode": "benchmark",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {"backend": "native"},
            "metrics": {"classification_backend": "native"},
            "datasets": [],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": [DUMMY_FE],
                "mil": [DUMMY_MIL, SECOND_DUMMY_MIL],
                "loss": ["CrossEntropyLoss", "MSELoss"],
            },
        }
    )

    policy = BenchmarkingPolicy(cfg)
    configs = policy._generate_configs()

    assert len(configs) == 12

    model_names = set(c._active_model_name for c in configs)
    assert DUMMY_MIL in model_names
    assert SECOND_DUMMY_MIL in model_names


def test_benchmark_grid_applies_configurable_batch_size():
    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "test",
                "annotation_file": "x",
                "task": "classification",
                "mode": "benchmark",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {"backend": "native"},
            "metrics": {"classification_backend": "native"},
            "datasets": [],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": [DUMMY_FE],
                "mil": [DUMMY_MIL],
                "loss": ["CrossEntropyLoss"],
                "batch_size": [1, 4],
                "seeds": [7],
            },
        }
    )

    configs = BenchmarkingPolicy(cfg)._generate_configs()

    assert len(configs) == 2
    assert sorted(config.mil.batch_size for config in configs) == [1, 4]


def test_apply_search_params_updates_pipeline_and_training_fields():
    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "test",
                "annotation_file": "x",
                "task": "classification",
                "mode": "benchmark",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {"backend": "native"},
            "metrics": {"classification_backend": "native"},
            "datasets": [],
            "benchmark_parameters": {
                "tile_px": [256, 512],
                "tile_mpp": [0.5, 1.0],
                "feature_extraction": [DUMMY_FE],
                "mil": [DUMMY_MIL],
                "loss": ["CrossEntropyLoss"],
                "epochs": [10, 20],
                "z_dim": [128, 256],
                "bag_size": [64, 128],
            },
        }
    )

    updated = apply_search_params(
        cfg,
        {
            "tile_px": 512,
            "tile_mpp": 1.0,
            "feature_extraction": DUMMY_FE,
            "mil": DUMMY_MIL,
            "loss": "CrossEntropyLoss",
            "epochs": 20,
            "z_dim": 128,
            "bag_size": 64,
        },
    )

    assert updated.benchmark_parameters.tile_px == [512]
    assert updated.benchmark_parameters.tile_mpp == [1.0]
    assert updated.benchmark_parameters.feature_extraction == [DUMMY_FE]
    assert updated.mil.epochs == 20
    assert updated.mil.z_dim == 128
    assert updated.mil.bag_size == 64
    assert updated._active_model_name == DUMMY_MIL
    assert updated._active_loss_name == "CrossEntropyLoss"


def test_benchmark_execute_uses_inferred_bag_and_annotation_dimensions(monkeypatch, tmp_path):
    captured: dict[str, int] = {}

    class _CaptureModel(MILModelBase):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            captured["input_dim"] = int(input_dim)
            captured["output_dim"] = int(output_dim)

        @property
        def bag_size(self):
            return None

        def forward_bag(self, x, **kwargs):
            return x

    class _FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def fit(self, model, ds_train, ds_val, loss_fn):
            return "best.ckpt", 0.75

    class _FakeDataset:
        feature_dim = 11

        def output_dim(self):
            return 4

    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "test",
                "annotation_file": "x",
                "task": "survival_discrete",
                "mode": "benchmark",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {"backend": "native"},
            "metrics": {"classification_backend": "native"},
            "datasets": [
                {
                    "name": "ds_train",
                    "slides_dir": str(tmp_path / "slides"),
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "used_for": "all",
                }
            ],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": [DUMMY_FE],
                "mil": [DUMMY_MIL],
                "loss": ["CrossEntropyLoss"],
                "seeds": [7],
            },
        }
    )

    monkeypatch.setattr(benchmark_mod, "build_bag_dataset_for_task", lambda *args, **kwargs: _FakeDataset())
    monkeypatch.setattr(benchmark_mod, "resolve_dataset_feature_dir", lambda dataset_entry: tmp_path)
    monkeypatch.setattr(benchmark_mod, "infer_model_dimensions", lambda dataset: (dataset.feature_dim, dataset.output_dim()))
    monkeypatch.setattr(policy_utils.MODELS, "get", lambda name: _CaptureModel)
    monkeypatch.setattr(benchmark_mod.LOSSES, "get", lambda name: (lambda: object()))
    monkeypatch.setattr(benchmark_mod.TRAINERS, "get", lambda name: _FakeTrainer)
    monkeypatch.setattr(BenchmarkingPolicy, "_save_report", lambda self: None)

    BenchmarkingPolicy(cfg).execute()

    assert captured == {"input_dim": 11, "output_dim": 4}


def test_benchmark_execute_writes_sorted_summary_and_visualizations(
    monkeypatch, tmp_path
):
    class _FakeTrainer:
        def __init__(self, cfg):
            self.cfg = cfg

        def fit(self, model, ds_train, ds_val, loss_fn):
            _ = (model, ds_train, ds_val, loss_fn)
            score = 0.9 if self.cfg.mil.batch_size == 1 else 0.6
            return f"batch_{self.cfg.mil.batch_size}.ckpt", score

    class _FakeDataset:
        feature_dim = 8

        def output_dim(self):
            return 2

    class _FakeFigure:
        def __init__(self):
            self.paths: list[str] = []

        def update_layout(self, **kwargs):
            _ = kwargs

        def write_html(self, path: str):
            Path(path).write_text("<html></html>", encoding="utf-8")

    class _FakePX:
        def bar(self, *args, **kwargs):
            _ = (args, kwargs)
            return _FakeFigure()

        def scatter(self, *args, **kwargs):
            _ = (args, kwargs)
            return _FakeFigure()

    monkeypatch.setattr(benchmark_mod, "build_bag_dataset_for_task", lambda *args, **kwargs: _FakeDataset())
    monkeypatch.setattr(benchmark_mod, "resolve_dataset_feature_dir", lambda dataset_entry: tmp_path)
    monkeypatch.setattr(benchmark_mod, "infer_model_dimensions", lambda dataset: (dataset.feature_dim, dataset.output_dim()))
    monkeypatch.setattr(benchmark_mod.LOSSES, "get", lambda name: (lambda: object()))
    monkeypatch.setattr(benchmark_mod.TRAINERS, "get", lambda name: _FakeTrainer)
    monkeypatch.setattr(policy_utils, "_load_plotly_modules", lambda: (_FakePX(), object()))

    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "bench_summary",
                "annotation_file": "x",
                "project_root": str((tmp_path / "project").resolve()),
                "task": "classification",
                "mode": "benchmark",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {"backend": "native", "best_epoch_based_on": "balanced_accuracy"},
            "metrics": {"classification_backend": "native"},
            "datasets": [
                {
                    "name": "ds_train",
                    "slides_dir": str(tmp_path / "slides"),
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "used_for": "all",
                }
            ],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": [DUMMY_FE],
                "mil": [DUMMY_MIL],
                "loss": ["CrossEntropyLoss"],
                "batch_size": [1, 4],
            },
        }
    )

    BenchmarkingPolicy(cfg).execute()

    output_root = tmp_path / "project" / "bench_summary"
    summary_path = output_root / "benchmark_results.csv"
    vis_dir = output_root / "benchmark_visualizations"
    assert summary_path.exists()
    df = pd.read_csv(summary_path)
    assert set(df["objective_value"].dropna().tolist()) == {0.6, 0.9}
    assert df["objective_value"].dropna().is_monotonic_decreasing
    successful = df[df["status"] == "success"]
    assert successful["rank"].dropna().tolist() == list(
        range(1, len(successful) + 1)
    )
    assert (vis_dir / "benchmark_performance_ranked.html").exists()
    assert (vis_dir / "benchmark_rank_scatter.html").exists()


def test_build_mil_model_for_config_preserves_torchmil_user_kwargs(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _CaptureTorchMIL(MILModelBase):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

        @property
        def bag_size(self):
            return None

        def forward_bag(self, x, **kwargs):
            return x

    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "torchmil_cfg",
                "annotation_file": "x",
                "task": "classification",
                "mode": "benchmark",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {
                "torchmil_model_kwargs": {
                    "in_shape": (7,),
                    "out_shape": 9,
                    "heads": 4,
                },
            },
            "metrics": {"classification_backend": "native"},
            "datasets": [
                {
                    "name": "ds_train",
                    "slides_dir": str(tmp_path / "slides"),
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "used_for": "all",
                }
            ],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": [DUMMY_FE],
                "mil": ["ABMIL"],
                "loss": ["CrossEntropyLoss"],
                "seeds": [7],
            },
        }
    )

    monkeypatch.setattr(policy_utils.MODELS, "get", lambda name: _CaptureTorchMIL)

    build_mil_model_for_config(
        cfg,
        model_name="ABMIL",
        input_dim=11,
        output_dim=4,
    )

    assert captured["torchmil_model"] == "ABMIL"
    assert captured["task"] == "classification"
    assert captured["torchmil_model_kwargs"] == {
        "in_shape": (7,),
        "out_shape": 9,
        "heads": 4,
    }


def test_build_mil_model_for_config_passes_native_architecture_kwargs(
    monkeypatch, tmp_path
):
    captured: dict[str, object] = {}

    class _CaptureNative(MILModelBase):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

        @property
        def bag_size(self):
            return None

        def forward_bag(self, x, **kwargs):
            return x

    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "native_cfg",
                "annotation_file": "x",
                "task": "classification",
                "mode": "benchmark",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {
                "backend": "native",
                "z_dim": 123,
                "dropout_p": 0.25,
                "encoder_layers": 3,
                "k": 7,
            },
            "metrics": {"classification_backend": "native"},
            "datasets": [
                {
                    "name": "ds_train",
                    "slides_dir": str(tmp_path / "slides"),
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "used_for": "all",
                }
            ],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": [DUMMY_FE],
                "mil": [DUMMY_MIL],
                "loss": ["CrossEntropyLoss"],
            },
        }
    )

    monkeypatch.setattr(policy_utils.MODELS, "get", lambda name: _CaptureNative)

    build_mil_model_for_config(
        cfg,
        model_name=DUMMY_MIL,
        input_dim=11,
        output_dim=4,
    )

    assert captured["input_dim"] == 11
    assert captured["output_dim"] == 4
    assert captured["hidden_dim"] == 123
    assert captured["latent_dim"] == 123
    assert captured["dropout"] == 0.25
    assert captured["encoder_layers"] == 3
    assert captured["k"] == 7


def test_build_mil_model_for_config_merges_mil_lab_defaults(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class _CaptureMILLab(MILModelBase):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

        @property
        def bag_size(self):
            return None

        def forward_bag(self, x, **kwargs):
            return x

    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "mil_lab_cfg",
                "annotation_file": "x",
                "task": "classification",
                "mode": "benchmark",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {"backend": "native"},
            "metrics": {"classification_backend": "native"},
            "datasets": [
                {
                    "name": "ds_train",
                    "slides_dir": str(tmp_path / "slides"),
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "used_for": "all",
                }
            ],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": [DUMMY_FE],
                "mil": [DUMMY_MIL],
                "loss": ["CrossEntropyLoss"],
                "seeds": [7],
            },
        }
    )
    cfg.experiment.task = "regression"
    cfg.mil.backend = "mil-lab"
    cfg.mil.mil_lab_model = "abmil"
    cfg.mil.mil_lab_model_kwargs = {"embed_dim": 128}
    cfg.mil.mil_lab_from_pretrained = True

    monkeypatch.setattr(policy_utils.MODELS, "get", lambda name: _CaptureMILLab)

    build_mil_model_for_config(
        cfg,
        model_name="mil-lab",
        input_dim=13,
        output_dim=1,
    )

    assert captured["mil_lab_model"] == "abmil"
    assert captured["task"] == "regression"
    assert captured["mil_lab_from_pretrained"] is True
    assert captured["mil_lab_model_kwargs"] == {
        "embed_dim": 128,
        "input_dim": 13,
        "output_dim": 1,
    }


def test_write_experiment_summary_csv_ranks_successes_and_appends_failures(tmp_path: Path):
    rows = [
        {
            "run_index": 0,
            "status": "success",
            "objective_metric": "balanced_accuracy",
            "objective_value": 0.81,
        },
        {
            "run_index": 1,
            "status": "failed",
            "objective_metric": "balanced_accuracy",
            "objective_value": None,
            "error": "boom",
        },
        {
            "run_index": 2,
            "status": "success",
            "objective_metric": "balanced_accuracy",
            "objective_value": 0.93,
        },
    ]

    output_path = tmp_path / "benchmark_results.csv"
    df = write_experiment_summary_csv(
        rows,
        output_path=output_path,
        objective_metric="balanced_accuracy",
        minimize=False,
    )

    assert output_path.exists()
    assert df["objective_value"].tolist()[:2] == [0.93, 0.81]
    assert df["rank"].tolist()[:2] == [1, 2]
    assert pd.isna(df.iloc[2]["rank"])
    assert df.iloc[2]["status"] == "failed"
