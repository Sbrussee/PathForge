from __future__ import annotations

import copy
from importlib import import_module

import pytest

import pathforge.config.config as config_module
import pathforge.utils.registries as registries_module
from pathforge.adapters.mil_lab.backend import register_mil_lab_backend
from pathforge.adapters.torchmil.backend import register_torchmil_backend
from pathforge.config.config import Config
from pathforge.core.slide_processing.base import SlideProcessorBase
from pathforge.utils.registries import list_feature_extractors, list_mil_models
from tests.conftest import DUMMY_FE, DUMMY_MIL


def test_lazyslide_backend_registers_slide_processor() -> None:
    """PathForge should expose the LazySlide backend via the slide processor registry."""
    pytest.importorskip("lazyslide")
    import_module("pathforge.core.slide_processing.lazyslide")

    processor_cls = registries_module.SLIDE_PROCESSORS.get("lazyslide")

    assert issubclass(processor_cls, SlideProcessorBase)


def test_config_accepts_lazyslide_backend_for_lazyslide_extractors(
    minimal_fe_config: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LazySlide-only feature extractors should validate when the backend matches."""
    cfg_dict = copy.deepcopy(minimal_fe_config)
    cfg_dict["benchmark_parameters"]["feature_extraction"] = ["lazy_backbone"]

    monkeypatch.setattr(config_module, "populate_dynamic_registries", lambda: None)
    monkeypatch.setattr(
        config_module,
        "is_feature_extractor_available",
        lambda name: name == "lazy_backbone",
    )
    monkeypatch.setattr(
        config_module,
        "all_feature_extractor_names",
        lambda: {"lazy_backbone"},
    )
    monkeypatch.setattr(config_module, "LAZYSLIDE_MODEL_NAMES", {"lazy_backbone"})

    cfg = Config.model_validate(cfg_dict)

    assert cfg.slide_processing.backend == "lazyslide"
    assert cfg.benchmark_parameters.feature_extraction == ["lazy_backbone"]


def test_config_accepts_torchmil_backend_and_heatmap_backend_when_available(
    minimal_benchmark_config: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TorchMIL should be selectable as a PathForge MIL and heatmap backend."""
    cfg_dict = copy.deepcopy(minimal_benchmark_config)
    cfg_dict["mil"] = {
        "backend": "torchmil",
        "torchmil_model": "ABMIL",
        "torchmil_model_kwargs": {"in_shape": (128,), "out_shape": 2},
    }
    cfg_dict["explainability"] = {"heatmap_backend": "torchmil"}
    cfg_dict["benchmark_parameters"]["mil"] = ["torchmil"]

    register_torchmil_backend()
    monkeypatch.setattr(config_module, "is_torchmil_available", lambda: True)
    monkeypatch.setattr(config_module, "populate_dynamic_registries", register_torchmil_backend)

    cfg = Config.model_validate(cfg_dict)

    assert cfg.mil.backend == "torchmil"
    assert cfg.mil.torchmil_model == "ABMIL"
    assert cfg.explainability.heatmap_backend == "torchmil"


def test_config_accepts_mil_lab_backend_when_available(
    minimal_benchmark_config: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MIL-Lab should be selectable as a PathForge MIL backend."""
    cfg_dict = copy.deepcopy(minimal_benchmark_config)
    cfg_dict["mil"] = {
        "backend": "mil-lab",
        "mil_lab_model": "abmil",
        "mil_lab_model_kwargs": {"num_classes": 2},
    }
    cfg_dict["benchmark_parameters"]["mil"] = ["mil-lab"]

    register_mil_lab_backend()
    monkeypatch.setattr(config_module, "is_mil_lab_available", lambda: True)
    monkeypatch.setattr(config_module, "populate_dynamic_registries", register_mil_lab_backend)

    cfg = Config.model_validate(cfg_dict)

    assert cfg.mil.backend == "mil-lab"
    assert cfg.mil.mil_lab_model == "abmil"


def test_config_accepts_torchmetrics_and_torchsurv_metric_backends_when_available(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PathForge metric backend selectors should validate optional backends."""
    ann = tmp_path / "annotations.csv"
    ann.write_text("dataset,slide,patient,category\n", encoding="utf-8")
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()

    monkeypatch.setattr(config_module, "is_torchmetrics_available", lambda: True)
    monkeypatch.setattr(config_module, "is_torchsurv_available", lambda: True)

    classification_cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "classification_backend_cfg",
                "annotation_file": str(ann),
                "project_root": str(tmp_path / "project_cls"),
                "mode": "benchmark",
                "task": "classification",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {"backend": "native"},
            "metrics": {"classification_backend": "torchmetrics"},
            "datasets": [
                {
                    "name": "ds",
                    "slides_dir": str(slides_dir),
                    "artifacts_dir": str(tmp_path / "artifacts_cls"),
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

    survival_cfg = Config.model_validate(
        {
            "experiment": {
                "project_name": "survival_backend_cfg",
                "annotation_file": str(ann),
                "project_root": str(tmp_path / "project_surv"),
                "mode": "benchmark",
                "task": "survival",
            },
            "slide_processing": {"backend": "lazyslide"},
            "mil": {"backend": "native"},
            "metrics": {
                "classification_backend": "native",
                "survival_continuous_backend": "torchsurv",
            },
            "datasets": [
                {
                    "name": "ds",
                    "slides_dir": str(slides_dir),
                    "artifacts_dir": str(tmp_path / "artifacts_surv"),
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

    assert classification_cfg.metrics.classification_backend == "torchmetrics"
    assert survival_cfg.metrics.survival_continuous_backend == "torchsurv"


def test_populate_dynamic_registries_registers_optional_metric_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamic registry population should expose optional metric/loss backends."""
    monkeypatch.setattr(registries_module, "_populated", False)
    monkeypatch.setattr(registries_module, "is_torchmil_available", lambda: False)
    monkeypatch.setattr(registries_module, "is_mil_lab_available", lambda: False)
    monkeypatch.setattr(registries_module, "is_torchmetrics_available", lambda: True)
    monkeypatch.setattr(registries_module, "is_torchsurv_available", lambda: True)

    registries_module.populate_dynamic_registries()

    assert registries_module.CLASSIFICATION_METRICS.is_available("torchmetrics")
    assert registries_module.SURVIVAL_METRICS.is_available("torchsurv")
    assert registries_module.SURVIVAL_LOSSES.is_available("torchsurv")


def test_list_feature_extractors_reports_backend_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Feature extractor listing should expose the backend required per entry."""
    monkeypatch.setattr(
        registries_module,
        "registered_feature_extractor_names",
        lambda: {"native_fe"},
    )
    monkeypatch.setattr(registries_module, "timm_model_names", lambda: {"resnet18"})
    monkeypatch.setattr(
        registries_module,
        "lazyslide_model_names",
        lambda: {"lazy_uni"},
    )

    entries = list_feature_extractors()

    observed = {
        (entry.name, entry.backend, entry.config_field, entry.available)
        for entry in entries
    }
    assert observed == {
        ("native_fe", "native", "benchmark_parameters.feature_extraction", True),
        ("resnet18", "timm", "benchmark_parameters.feature_extraction", True),
        ("lazy_uni", "lazyslide", "benchmark_parameters.feature_extraction", True),
    }


def test_list_feature_extractors_handles_missing_optional_catalogs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extractor listing should still work when optional backends are absent."""
    monkeypatch.setattr(
        registries_module,
        "registered_feature_extractor_names",
        lambda: {"native_fe"},
    )
    monkeypatch.setattr(registries_module, "timm_model_names", lambda: set())
    monkeypatch.setattr(registries_module, "lazyslide_model_names", lambda: set())

    entries = list_feature_extractors()

    assert len(entries) == 1
    assert entries[0].name == "native_fe"
    assert entries[0].backend == "native"


def test_list_mil_models_reports_backend_requirements_and_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MIL listing should expose backend requirements even for unavailable adapters."""
    monkeypatch.setattr(registries_module, "_import_native_model_modules", lambda: None)
    monkeypatch.setattr(
        registries_module.MODELS,
        "is_available",
        lambda name: name in {"PerceiverMIL", "PrototypeMIL", "VarMIL"},
    )
    monkeypatch.setattr(registries_module, "is_torchmil_available", lambda: False)
    monkeypatch.setattr(registries_module, "is_mil_lab_available", lambda: True)

    entries = list_mil_models()
    lookup = {(entry.name, entry.backend): entry for entry in entries}

    assert ("GCNConvMIL", "native") not in lookup
    assert lookup[("MambaMIL", "native")].available is False
    assert lookup[("ABMIL", "torchmil")].config_field == "benchmark_parameters.mil"
    assert lookup[("ABMIL", "torchmil")].available is False
    assert lookup[("abmil", "mil-lab")].config_field == "benchmark_parameters.mil"
    assert lookup[("abmil", "mil-lab")].available is True
