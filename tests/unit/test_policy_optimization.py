from __future__ import annotations

from pathlib import Path
import sys
import types

import optuna

import pathbench.policy.optimization as opt_mod
from pathbench.config.config import Config
from pathbench.policy.optimization import OptimizationPolicy
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
