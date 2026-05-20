from pathbench.config.config import Config
from pathbench.core.models.mil_base import MILModelBase
import pathbench.policy.benchmarking as bench_mod
import pathbench.policy.utils as policy_utils
from pathbench.policy.benchmarking import BenchmarkingPolicy
from pathbench.policy.utils import (
    apply_search_params,
    build_mil_model_for_config,
    write_experiment_summary_csv,
)
from pathbench.utils.registries import MODELS
from tests.conftest import DUMMY_FE, DUMMY_MIL
import pandas as pd
from pathlib import Path

SECOND_DUMMY_MIL = "DummyMIL2"

if not MODELS.is_available(SECOND_DUMMY_MIL):

    @MODELS.register(SECOND_DUMMY_MIL)
    class _DummyMIL2(MILModelBase):
        @property
        def bag_size(self):
            return None

        def forward_bag(self, x, **kwargs):
            return x


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

    monkeypatch.setattr(bench_mod, "build_bag_dataset_for_task", lambda *args, **kwargs: _FakeDataset())
    monkeypatch.setattr(bench_mod, "resolve_dataset_feature_dir", lambda dataset_entry: tmp_path)
    monkeypatch.setattr(bench_mod, "infer_model_dimensions", lambda dataset: (dataset.feature_dim, dataset.output_dim()))
    monkeypatch.setattr(policy_utils.MODELS, "get", lambda name: _CaptureModel)
    monkeypatch.setattr(bench_mod.LOSSES, "get", lambda name: (lambda: object()))
    monkeypatch.setattr(bench_mod.TRAINERS, "get", lambda name: _FakeTrainer)
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

    monkeypatch.setattr(bench_mod, "build_bag_dataset_for_task", lambda *args, **kwargs: _FakeDataset())
    monkeypatch.setattr(bench_mod, "resolve_dataset_feature_dir", lambda dataset_entry: tmp_path)
    monkeypatch.setattr(bench_mod, "infer_model_dimensions", lambda dataset: (dataset.feature_dim, dataset.output_dim()))
    monkeypatch.setattr(bench_mod.LOSSES, "get", lambda name: (lambda: object()))
    monkeypatch.setattr(bench_mod.TRAINERS, "get", lambda name: _FakeTrainer)
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

    summary_path = tmp_path / "project" / "benchmark_results.csv"
    vis_dir = tmp_path / "project" / "benchmark_visualizations"
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
                "backend": "torchmil",
                "torchmil_model": "ABMIL",
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
                "mil": ["torchmil"],
                "loss": ["CrossEntropyLoss"],
                "seeds": [7],
            },
        }
    )

    monkeypatch.setattr(policy_utils.MODELS, "get", lambda name: _CaptureTorchMIL)

    build_mil_model_for_config(
        cfg,
        model_name="torchmil",
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
