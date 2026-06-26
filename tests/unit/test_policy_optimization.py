from __future__ import annotations

from pathlib import Path
import sys
import types

import optuna
import pandas as pd
import pytest

import pathbench.policy.optimization as opt_mod
import pathbench.policy.utils as policy_utils
from pathbench.config.config import Config
from pathbench.policy.optimization import OptimizationPolicy
from pathbench.policy.utils import build_bag_dataset_for_task, optimization_search_space
from tests.conftest import DUMMY_FE


def _make_cfg(
    tmp_path: Path,
    *,
    sampler: str = "TPESampler",
    pruner: str = "MedianPruner",
    objective_mode: str = "max",
) -> Config:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("dataset,slide,patient,category\n", encoding="utf-8")
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()

    return Config.model_validate(
        {
            "experiment": {
                "project_name": "opt_policy",
                "annotation_file": str(annotation_path),
                "project_root": str((tmp_path / "project").resolve()),
                "mode": "optimization",
                "task": "classification",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {"backend": "native"},
            "metrics": {"classification_backend": "native"},
            "optimization": {
                "study_name": "study",
                "sampler": sampler,
                "pruner": pruner,
                "objective_mode": objective_mode,
                "objective_metric": "balanced_accuracy",
                "trials": 1,
            },
            "datasets": [
                {
                    "name": "ds",
                    "slides_dir": str(slides_dir),
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "used_for": "all",
                }
            ],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": [DUMMY_FE],
                "mil": ["DummyMIL"],
                "loss": ["CrossEntropyLoss"],
            },
        }
    )


def test_optimization_policy_selects_sampler(tmp_path: Path) -> None:
    policy = OptimizationPolicy(_make_cfg(tmp_path, sampler="RandomSampler"))
    sampler = policy._get_sampler()
    assert isinstance(sampler, optuna.samplers.RandomSampler)


def test_optimization_policy_selects_pruner(tmp_path: Path) -> None:
    policy = OptimizationPolicy(_make_cfg(tmp_path, pruner="HyperbandPruner"))
    pruner = policy._get_pruner()
    assert isinstance(pruner, optuna.pruners.HyperbandPruner)


def test_optimization_policy_maps_max_mode_to_maximize(tmp_path: Path) -> None:
    policy = OptimizationPolicy(_make_cfg(tmp_path, objective_mode="max"))
    assert policy._get_direction() == "maximize"


def test_optimization_policy_maps_loss_metric_to_minimize_when_mode_unknown(
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(tmp_path, objective_mode="max")
    cfg.optimization.objective_mode = "unexpected"
    cfg.optimization.objective_metric = "val_loss"
    policy = OptimizationPolicy(cfg)
    assert policy._get_direction() == "minimize"


def test_optimization_objective_uses_mil_lab_user_config_and_inferred_dims(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _CaptureMILLabModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _FakeDataset:
        feature_dim = 13

        def output_dim(self) -> int:
            return 5

    class _FakeTrainer:
        def __init__(self, cfg, extra_callbacks=None):
            self.cfg = cfg
            self.extra_callbacks = extra_callbacks or []

        def fit(self, model, ds_train, ds_val, loss_fn):
            _ = (model, ds_train, ds_val, loss_fn)
            return "checkpoint.ckpt", 0.75

    cfg = _make_cfg(tmp_path)
    cfg.experiment.task = "survival_discrete"
    cfg.mil.backend = "mil-lab"
    cfg.mil.mil_lab_model = "abmil"
    cfg.mil.mil_lab_model_kwargs = {"embed_dim": 128}
    cfg.mil.mil_lab_from_pretrained = True
    cfg.benchmark_parameters.mil = ["mil-lab"]
    cfg.optimization.search_space = {}

    fake_integration = types.ModuleType("optuna.integration")
    fake_integration.PyTorchLightningPruningCallback = (
        lambda trial, monitor: object()
    )
    monkeypatch.setitem(sys.modules, "optuna.integration", fake_integration)
    monkeypatch.setattr(opt_mod, "build_bag_dataset_for_task", lambda *args, **kwargs: _FakeDataset())
    monkeypatch.setattr(opt_mod, "resolve_dataset_feature_dir", lambda dataset_entry: tmp_path)
    monkeypatch.setattr(
        opt_mod,
        "infer_model_dimensions",
        lambda dataset: (dataset.feature_dim, dataset.output_dim()),
    )
    monkeypatch.setattr(opt_mod, "build_mil_model_for_config", lambda config, **kwargs: _CaptureMILLabModel(**kwargs))
    monkeypatch.setattr(opt_mod.LOSSES, "get", lambda name: (lambda: object()))
    monkeypatch.setattr(opt_mod.TRAINERS, "get", lambda name: _FakeTrainer)

    score = OptimizationPolicy(cfg).objective(optuna.trial.FixedTrial({}))

    assert score == 0.75
    assert captured == {
        "model_name": "mil-lab",
        "input_dim": 13,
        "output_dim": 5,
        "extra_kwargs": {"dropout": cfg.mil.dropout_p},
    }


def test_build_bag_dataset_for_task_filters_to_dataset_entry_rows(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text(
        "dataset,slide_id,category\ntrain_ds,S1,0\nval_ds,S2,1\n",
        encoding="utf-8",
    )
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    (tmp_path / "slides").mkdir()

    import torch

    torch.save(torch.zeros(2, 3), feature_dir / "S1.pt")
    torch.save(torch.ones(2, 3), feature_dir / "S2.pt")

    cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "opt_filter",
                "annotation_file": str(annotation_path),
                "project_root": str((tmp_path / "project").resolve()),
                "mode": "optimization",
                "task": "classification",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {"backend": "native"},
            "metrics": {"classification_backend": "native"},
            "optimization": {
                "study_name": "study",
                "sampler": "TPESampler",
                "pruner": "MedianPruner",
                "objective_mode": "max",
                "objective_metric": "balanced_accuracy",
                "trials": 1,
            },
            "datasets": [
                {
                    "name": "train_ds",
                    "slides_dir": str(tmp_path / "slides"),
                    "artifacts_dir": str(feature_dir),
                    "used_for": "training",
                },
                {
                    "name": "val_ds",
                    "slides_dir": str(tmp_path / "slides"),
                    "artifacts_dir": str(feature_dir),
                    "used_for": "validation",
                },
            ],
            "benchmark_parameters": {
                "tile_px": [256],
                "tile_mpp": [0.5],
                "feature_extraction": [DUMMY_FE],
                "mil": ["DummyMIL"],
                "loss": ["CrossEntropyLoss"],
            },
        }
    )

    dataset = build_bag_dataset_for_task(
        cfg,
        feature_dir=feature_dir,
        name="train",
        dataset_entry=cfg.datasets[0],
    )

    assert dataset.num_bags == 1
    assert dataset.annotations["dataset"].tolist() == ["train_ds"]
    assert dataset.annotations["slide_id"].tolist() == ["S1"]


def test_optimization_objective_uses_dataset_use_semantics_and_does_not_mutate_base_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_calls: list[tuple[str, str]] = []

    class _FakeDataset:
        feature_dim = 11

        def output_dim(self) -> int:
            return 2

    class _FakeTrainer:
        def __init__(self, cfg, extra_callbacks=None):
            self.cfg = cfg
            self.extra_callbacks = extra_callbacks or []

        def fit(self, model, ds_train, ds_val, loss_fn):
            _ = (model, ds_train, ds_val, loss_fn)
            return "checkpoint.ckpt", 0.5

    fake_integration = types.ModuleType("optuna.integration")
    fake_integration.PyTorchLightningPruningCallback = (
        lambda trial, monitor: object()
    )
    monkeypatch.setitem(sys.modules, "optuna.integration", fake_integration)

    cfg = _make_cfg(tmp_path)
    cfg.datasets = [
        cfg.datasets[0].model_copy(update={"name": "val_ds", "used_for": "validation"}),
        cfg.datasets[0].model_copy(update={"name": "train_ds", "used_for": "training"}),
    ]
    cfg.benchmark_parameters.mil = ["DummyMIL", "DummyMIL2"]
    cfg.optimization.search_space = {
        "mil": {"type": "categorical", "choices": ["DummyMIL", "DummyMIL2"]},
        "batch_size": {"type": "categorical", "choices": [1, 4]},
    }

    def _fake_build_bag_dataset_for_task(config, *, feature_dir, name, dataset_entry=None):
        _ = config, feature_dir
        captured_calls.append((name, dataset_entry.name))
        return _FakeDataset()

    monkeypatch.setattr(opt_mod, "build_bag_dataset_for_task", _fake_build_bag_dataset_for_task)
    monkeypatch.setattr(opt_mod, "resolve_dataset_feature_dir", lambda dataset_entry: tmp_path)
    monkeypatch.setattr(
        opt_mod,
        "infer_model_dimensions",
        lambda dataset: (dataset.feature_dim, dataset.output_dim()),
    )
    monkeypatch.setattr(opt_mod, "build_mil_model_for_config", lambda config, **kwargs: object())
    monkeypatch.setattr(opt_mod.LOSSES, "get", lambda name: (lambda: object()))
    monkeypatch.setattr(opt_mod.TRAINERS, "get", lambda name: _FakeTrainer)

    trial = optuna.trial.FixedTrial({"mil": "DummyMIL2", "batch_size": 4})
    score = OptimizationPolicy(cfg).objective(trial)

    assert score == 0.5
    assert captured_calls == [("train", "train_ds"), ("val", "val_ds")]
    assert cfg.benchmark_parameters.mil == ["DummyMIL", "DummyMIL2"]
    assert cfg.mil.batch_size == 1


def test_optimization_search_space_merges_benchmark_component_choices(
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(tmp_path)
    cfg.benchmark_parameters.tile_px = [256, 512]
    cfg.benchmark_parameters.mil = ["DummyMIL", "DummyMIL2"]
    cfg.benchmark_parameters.loss = ["CrossEntropyLoss", "MSELoss"]
    cfg.optimization.search_space = {
        "epochs": {"type": "int", "low": 5, "high": 15, "step": 5},
        "lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
    }

    merged = optimization_search_space(cfg)

    assert merged["epochs"].kind == "int"
    assert merged["tile_px"].kind == "categorical"
    assert merged["tile_px"].choices == [256, 512]
    assert merged["mil"].choices == ["DummyMIL", "DummyMIL2"]
    assert merged["loss"].choices == ["CrossEntropyLoss", "MSELoss"]


def test_optimization_execute_writes_summary_csv_and_visualizations(
    monkeypatch, tmp_path: Path
) -> None:
    class _FakeStudy:
        best_params = {"lr": 1e-4}
        best_value = 0.82

        def __init__(self) -> None:
            self.optimize_calls: list[int] = []

        def optimize(self, objective, n_trials: int) -> None:
            _ = objective
            self.optimize_calls.append(n_trials)

        def trials_dataframe(self) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {
                        "number": 0,
                        "value": 0.82,
                        "state": "COMPLETE",
                        "params_lr": 1e-4,
                    },
                    {
                        "number": 1,
                        "value": 0.61,
                        "state": "COMPLETE",
                        "params_lr": 5e-4,
                    },
                ]
            )

    exported_dirs: list[Path] = []
    fake_study = _FakeStudy()
    monkeypatch.setattr(opt_mod.optuna, "create_study", lambda **kwargs: fake_study)

    def _fake_save_optuna_visualizations(study, output_dir):
        _ = study
        exported_dirs.append(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plot_optimization_history.html").write_text(
            "<html></html>", encoding="utf-8"
        )
        return [output_dir / "plot_optimization_history.html"]

    monkeypatch.setattr(
        opt_mod,
        "save_optuna_visualizations",
        _fake_save_optuna_visualizations,
    )

    cfg = _make_cfg(tmp_path)
    policy = OptimizationPolicy(cfg)
    monkeypatch.setattr(policy, "objective", lambda trial: 0.5)

    policy.execute()

    raw_results = tmp_path / "project" / "study_results.csv"
    summary_results = tmp_path / "project" / "optimization_results.csv"
    vis_dir = tmp_path / "project" / "optimization_visualizations"
    assert raw_results.exists()
    assert summary_results.exists()
    summary_df = pd.read_csv(summary_results)
    assert summary_df["objective_value"].tolist()[:2] == [0.82, 0.61]
    assert summary_df["rank"].tolist()[:2] == [1, 2]
    assert exported_dirs == [vis_dir]
    assert (vis_dir / "plot_optimization_history.html").exists()


def test_save_optuna_visualizations_exports_requested_figures(
    monkeypatch, tmp_path: Path
) -> None:
    written_html_paths: list[str] = []
    written_image_paths: list[str] = []

    class _FakeFigure:
        def write_html(self, path: str) -> None:
            written_html_paths.append(path)
            Path(path).write_text("<html></html>", encoding="utf-8")

        def write_image(self, path: str, format: str = "png") -> None:
            assert format == "png"
            written_image_paths.append(path)
            Path(path).write_bytes(b"png")

    class _FakeDirection:
        def __init__(self, label: str) -> None:
            self.label = label

        def __str__(self) -> str:
            return self.label

    class _FakeTrial:
        def __init__(self, values: list[float]) -> None:
            self.values = values

    class _FakeVisualization:
        def plot_optimization_history(self, study, **kwargs):
            _ = (study, kwargs)
            return _FakeFigure()

        def plot_param_importances(self, study, **kwargs):
            _ = (study, kwargs)
            return _FakeFigure()

        def plot_rank(self, study, **kwargs):
            _ = (study, kwargs)
            return _FakeFigure()

        def plot_timeline(self, study, **kwargs):
            _ = (study, kwargs)
            return _FakeFigure()

        def plot_hypervolume_history(self, study, reference_point):
            _ = study
            assert len(reference_point) == 2
            return _FakeFigure()

    def _fake_import_module(name: str):
        if name == "optuna.visualization":
            return _FakeVisualization()
        raise ImportError(name)

    monkeypatch.setattr(policy_utils.importlib, "import_module", _fake_import_module)
    fake_study = types.SimpleNamespace(
        directions=[_FakeDirection("minimize"), _FakeDirection("maximize")],
        trials=[_FakeTrial([0.5, 0.7]), _FakeTrial([0.4, 0.8])],
    )

    exported = policy_utils.save_optuna_visualizations(
        fake_study,
        output_dir=tmp_path / "optuna_visualizations",
    )

    assert sorted(path.name for path in exported) == [
        "plot_hypervolume_history.html",
        "plot_hypervolume_history.png",
        "plot_optimization_history.html",
        "plot_optimization_history.png",
        "plot_param_importances.html",
        "plot_param_importances.png",
        "plot_rank.html",
        "plot_rank.png",
        "plot_timeline.html",
        "plot_timeline.png",
    ]
    assert len(written_html_paths) == 5
    assert len(written_image_paths) == 5
