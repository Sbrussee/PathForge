from pathbench.config.config import Config
from pathbench.core.models.mil_base import MILModelBase
import pathbench.policy.benchmarking as bench_mod
import pathbench.policy.utils as policy_utils
from pathbench.policy.benchmarking import BenchmarkingPolicy
from pathbench.policy.utils import build_mil_model_for_config
from pathbench.utils.registries import MODELS
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
            return None

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
