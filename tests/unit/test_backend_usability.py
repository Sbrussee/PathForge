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
from pathforge.core.slide_processing.factory import build_slide_processor
from pathforge.core.feature_extractors.factory import (
    available_feature_extractor_names,
    resolve_feature_extractor_source,
)
from pathforge.utils.registries import (
    FEATURE_EXTRACTORS,
    list_feature_extractors,
    list_mil_models,
)
from pathforge.utils.registry import Registry
from tests.conftest import DUMMY_FE, DUMMY_MIL


def test_lazyslide_backend_registers_slide_processor() -> None:
    """PathForge should expose the LazySlide backend via the slide processor registry."""
    pytest.importorskip("lazyslide")
    import_module("pathforge.core.slide_processing.lazyslide")

    processor_cls = registries_module.SLIDE_PROCESSORS.get("lazyslide")

    assert issubclass(processor_cls, SlideProcessorBase)


def test_build_slide_processor_loads_registered_backend() -> None:
    """The shared factory constructs the processor registered by its backend module."""
    pytest.importorskip("lazyslide")

    processor = build_slide_processor("lazyslide")

    assert isinstance(processor, SlideProcessorBase)


def test_build_slide_processor_rejects_missing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared factory reports an unavailable backend before construction."""
    monkeypatch.setattr(registries_module, "SLIDE_PROCESSORS", Registry())
    import pathforge.core.slide_processing.factory as factory_module

    monkeypatch.setattr(
        factory_module,
        "import_module",
        lambda module_name: (_ for _ in ()).throw(ModuleNotFoundError(module_name)),
    )

    with pytest.raises(ValueError, match="backend 'missing' is not available"):
        build_slide_processor("missing")


def test_config_accepts_lazyslide_backend_for_lazyslide_extractors(
    minimal_fe_config: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LazySlide-only feature extractors should validate when the backend matches."""
    cfg_dict = copy.deepcopy(minimal_fe_config)
    cfg_dict["benchmark_parameters"]["feature_extraction"] = ["lazy_backbone"]

    monkeypatch.setattr(
        config_module,
        "resolve_feature_extractor_source",
        lambda backend_name, name: (
            "processor-native"
            if (backend_name, name) == ("lazyslide", "lazy_backbone")
            else pytest.fail(f"unexpected extractor resolution: {backend_name}/{name}")
        ),
    )

    cfg = Config.model_validate(cfg_dict)

    assert cfg.slide_processing.backend == "lazyslide"
    assert cfg.benchmark_parameters.feature_extraction == ["lazy_backbone"]


def test_config_does_not_resolve_processor_without_feature_extractors(
    minimal_fe_config: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty feature grids do not require an installed slide processor at config time."""
    cfg_dict = copy.deepcopy(minimal_fe_config)
    cfg_dict["slide_processing"]["backend"] = "openslide"
    cfg_dict["benchmark_parameters"]["feature_extraction"] = []
    monkeypatch.setattr(
        config_module,
        "resolve_feature_extractor_source",
        lambda backend_name, name: pytest.fail(
            f"unexpected extractor resolution: {backend_name}/{name}"
        ),
    )

    cfg = Config.model_validate(cfg_dict)

    assert cfg.slide_processing.backend == "openslide"


def test_available_feature_extractors_combine_selected_processor_and_pathforge_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The selected processor's names are combined transiently with PathForge names."""

    class ProcessorWithEncoders:
        def native_feature_extractor_names(self) -> set[str]:
            return {"processor_encoder", "shared_encoder"}

        def supports_pathforge_feature_extractors(self) -> bool:
            return True

    processor_registry = Registry()
    processor_registry.register("test-processor")(ProcessorWithEncoders)
    monkeypatch.setattr(registries_module, "SLIDE_PROCESSORS", processor_registry)
    monkeypatch.setattr(
        registries_module,
        "registered_feature_extractor_names",
        lambda: {"pathforge_encoder", "shared_encoder"},
    )

    assert available_feature_extractor_names("test-processor") == {
        "pathforge_encoder",
        "processor_encoder",
        "shared_encoder",
    }


def test_available_feature_extractors_reject_unavailable_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown processor names fail before feature-extractor validation proceeds."""
    monkeypatch.setattr(registries_module, "SLIDE_PROCESSORS", Registry())

    def raise_missing_module(module_name: str) -> None:
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(registries_module, "import_module", raise_missing_module)

    with pytest.raises(ValueError, match="backend 'missing-processor' is not available"):
        available_feature_extractor_names("missing-processor")


def test_available_feature_extractors_reject_processor_module_without_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Imported backend modules must register the requested processor name."""
    monkeypatch.setattr(registries_module, "SLIDE_PROCESSORS", Registry())
    monkeypatch.setattr(registries_module, "import_module", lambda module_name: None)

    with pytest.raises(ValueError, match="backend 'unregistered-processor' is not registered"):
        available_feature_extractor_names("unregistered-processor")


def test_resolve_feature_extractor_source_prefers_processor_native_on_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A processor-native name wins even when PathForge registers the same name."""

    class ProcessorWithCollision:
        def native_feature_extractor_names(self) -> set[str]:
            return {"shared_encoder"}

        def supports_pathforge_feature_extractors(self) -> bool:
            return True

    processor_registry = Registry()
    processor_registry.register("test-processor")(ProcessorWithCollision)
    monkeypatch.setattr(registries_module, "SLIDE_PROCESSORS", processor_registry)
    monkeypatch.setattr(
        registries_module,
        "is_registered_pathforge_feature_extractor",
        lambda name: name == "shared_encoder",
    )

    assert (
        resolve_feature_extractor_source("test-processor", "shared_encoder")
        == "processor-native"
    )


def test_resolve_feature_extractor_source_rejects_native_registry_for_unadapted_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registrations require an explicit processor adapter before validation accepts them."""

    class ProcessorWithoutAdapter:
        def native_feature_extractor_names(self) -> set[str]:
            return set()

        def supports_pathforge_feature_extractors(self) -> bool:
            return False

    processor_registry = Registry()
    processor_registry.register("test-processor")(ProcessorWithoutAdapter)
    monkeypatch.setattr(registries_module, "SLIDE_PROCESSORS", processor_registry)
    monkeypatch.setattr(
        registries_module,
        "is_registered_pathforge_feature_extractor",
        lambda name: name == "pathforge_encoder",
    )

    with pytest.raises(ValueError, match="pathforge_encoder.*test-processor"):
        resolve_feature_extractor_source("test-processor", "pathforge_encoder")


def test_lazyslide_processor_lists_lazyslide_and_timm_encoders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LazySlide advertises both catalogs that its extraction API accepts."""
    pytest.importorskip("lazyslide")
    lazyslide_module = import_module("pathforge.core.slide_processing.lazyslide")
    monkeypatch.setattr(registries_module, "lazyslide_model_names", lambda: {"lazy"})
    monkeypatch.setattr(registries_module, "timm_model_names", lambda: {"timm"})

    assert lazyslide_module.LazySlideProcessor().native_feature_extractor_names() == {
        "lazy",
        "timm",
    }


def test_pathforge_feature_extractor_population_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Known native extractor modules are imported once without full population."""
    imported_modules: list[str] = []
    monkeypatch.setattr(registries_module, "_feature_extractors_populated", False)
    monkeypatch.setattr(
        registries_module,
        "_NATIVE_FEATURE_EXTRACTOR_MODULES",
        ("pathforge.test_feature_extractor",),
    )
    monkeypatch.setattr(
        registries_module,
        "import_module",
        lambda module_name: imported_modules.append(module_name),
    )

    registries_module.populate_pathforge_feature_extractors()
    registries_module.populate_pathforge_feature_extractors()

    assert imported_modules == ["pathforge.test_feature_extractor"]


def test_dynamic_population_does_not_register_external_feature_extractors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """timm and LazySlide discovery must not mutate the PathForge extractor registry."""
    monkeypatch.setattr(registries_module, "_populated", False)
    monkeypatch.setattr(registries_module, "_import_native_model_modules", lambda: None)
    monkeypatch.setattr(
        registries_module,
        "populate_pathforge_feature_extractors",
        lambda: None,
    )
    monkeypatch.setattr(registries_module, "is_torchmil_available", lambda: False)
    monkeypatch.setattr(registries_module, "is_mil_lab_available", lambda: False)
    monkeypatch.setattr(registries_module, "is_torchmetrics_available", lambda: False)
    monkeypatch.setattr(registries_module, "is_torchsurv_available", lambda: False)
    monkeypatch.setattr(registries_module, "timm_model_names", lambda: {"timm_encoder"})
    monkeypatch.setattr(registries_module, "lazyslide_model_names", lambda: {"lazy_encoder"})

    names_before = set(FEATURE_EXTRACTORS.list_plugins())
    registries_module.populate_dynamic_registries()

    assert set(FEATURE_EXTRACTORS.list_plugins()) == names_before


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
