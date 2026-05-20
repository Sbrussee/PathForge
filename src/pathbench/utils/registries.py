from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any

from pathbench.adapters.losses import register_builtin_loss_factories
from pathbench.utils.registry import Registry
from pathbench.core.base import CoreRegistries
from pathbench.utils.optional.mil_lab import is_mil_lab_available
from pathbench.utils.optional.torchmil import (
    is_torchmetrics_available,
    is_torchmil_available,
    is_torchsurv_available,
)

# Define Registries
REGISTRIES = CoreRegistries(
    datasets=Registry(),
    models=Registry(),
    losses=Registry(),
    tasks=Registry(),
    explainers=Registry(),
    feature_extractors=Registry(),
    normalizers=Registry(),
    augmentation_methods=Registry(),
)

# Shortcuts
DATASETS = REGISTRIES.datasets
MODELS = REGISTRIES.models
LOSSES = REGISTRIES.losses
TASKS = REGISTRIES.tasks
EXPLAINERS = REGISTRIES.explainers
FEATURE_EXTRACTORS = REGISTRIES.feature_extractors
NORMALIZERS = REGISTRIES.normalizers
AUGMENTATION_METHODS = REGISTRIES.augmentation_methods

# Infrastructure
SLIDE_PROCESSORS = Registry()
TRAINERS = Registry()
CLASSIFICATION_METRICS = Registry()
SURVIVAL_METRICS = Registry()
SURVIVAL_LOSSES = Registry()

register_builtin_loss_factories(LOSSES)

# Track Lazyslide-specific models for validation (filled by populate_dynamic_registries)
LAZYSLIDE_MODEL_NAMES: set[str] = set()


@dataclass(frozen=True)
class BackendCatalogEntry:
    """One backend-aware catalog entry for user-selectable components.

    Attributes:
        name: User-facing model or extractor name.
        backend: Backend required to use this entry, such as ``native``,
            ``torchmil``, ``mil-lab``, ``timm``, or ``lazyslide``.
        config_field: Config field used to select this entry.
        source: Origin namespace that provides the entry.
        available: Whether the required backend is currently installed and the
            entry can be selected in this environment.

    Example:
        ```python
        entries = list_mil_models()
        torchmil_names = [item.name for item in entries if item.backend == "torchmil"]
        ```
    """

    name: str
    backend: str
    config_field: str
    source: str
    available: bool


# ---------------------------------------------------------------------------
# Optional dependency discovery (safe, lazy)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _timm_module():
    try:
        import timm  # noqa: WPS433

        return timm
    except Exception:
        return None


@lru_cache(maxsize=1)
def _lazyslide_module():
    try:
        import lazyslide as zs  # noqa: WPS433

        return zs
    except Exception:
        return None


def _normalize_model_names(items: Any) -> set[str]:
    """
    Convert list_models() output to a set[str], even if entries are not plain strings.
    """
    names: set[str] = set()
    if items is None:
        return names

    for item in items:
        if isinstance(item, str):
            names.add(item)
            continue

        if hasattr(item, "key"):
            try:
                names.add(str(item.key))
                continue
            except Exception:
                pass

        if isinstance(item, dict) and "key" in item:
            try:
                names.add(str(item["key"]))
                continue
            except Exception:
                pass

        try:
            names.add(str(item))
        except Exception:
            continue

    return names


@lru_cache(maxsize=1)
def timm_model_names() -> set[str]:
    timm = _timm_module()
    if timm is None:
        return set()
    try:
        return _normalize_model_names(timm.list_models())
    except Exception:
        return set()


@lru_cache(maxsize=1)
def lazyslide_model_names() -> set[str]:
    zs = _lazyslide_module()
    if zs is None:
        return set()
    try:
        return _normalize_model_names(zs.models.list_models())
    except Exception:
        return set()


def registered_feature_extractor_names() -> set[str]:
    """
    Best-effort extraction of names currently registered in PathBench FEATURE_EXTRACTORS.
    """
    # Prefer public APIs if your Registry exposes them
    for attr in ("keys", "list", "names"):
        fn = getattr(FEATURE_EXTRACTORS, attr, None)
        if callable(fn):
            try:
                return set(fn())
            except Exception:
                pass

    # Fallback to common internal dict names
    for attr in ("_registry", "_items", "registry"):
        obj = getattr(FEATURE_EXTRACTORS, attr, None)
        if isinstance(obj, dict):
            return set(obj.keys())

    return set()


def available_feature_extractor_names() -> dict[str, set[str]]:
    return {
        "pathbench": registered_feature_extractor_names(),
        "timm": timm_model_names(),
        "lazyslide": lazyslide_model_names(),
    }


def all_feature_extractor_names() -> set[str]:
    grouped = available_feature_extractor_names()
    return grouped["pathbench"] | grouped["timm"] | grouped["lazyslide"]


def is_feature_extractor_available(name: str) -> bool:
    """
    Used by config validation without importing timm/torchvision at module import time.
    """
    if FEATURE_EXTRACTORS.is_available(name):
        return True
    if name in timm_model_names():
        return True
    if name in lazyslide_model_names():
        return True
    return False


# ---------------------------------------------------------------------------
# Dynamic registry population (explicit, not at import time)
# ---------------------------------------------------------------------------

_populated = False

_NATIVE_MODEL_MODULES: tuple[str, ...] = (
    "pathbench.core.models.perceiver_mil",
    "pathbench.core.models.prototype_mil",
    "pathbench.core.models.slide_mlp",
    "pathbench.core.models.var_mil",
)

_OPTIONAL_NATIVE_MODEL_MODULES: dict[str, str] = {
    "pathbench.core.models.mamba_mil": "mamba",
}

_NATIVE_MIL_MODEL_NAMES: tuple[str, ...] = (
    "PerceiverMIL",
    "PrototypeMIL",
    "SlideVectorMLP",
    "VarMIL",
)

_OPTIONAL_NATIVE_MIL_MODELS: dict[str, str] = {
    "MambaMIL": "mamba",
}


def _import_native_model_modules() -> None:
    """Import native PathBench model modules so their registry decorators run."""

    for module_name in _NATIVE_MODEL_MODULES:
        import_module(module_name)

    for module_name, dependency_name in _OPTIONAL_NATIVE_MODEL_MODULES.items():
        try:
            __import__(dependency_name)
        except Exception:
            continue
        import_module(module_name)


def populate_dynamic_registries() -> None:
    """
    Populate optional backend registries with entries from installed packages.

    IMPORTANT:
    - This is NOT called automatically at import time.
    - Call it explicitly in CLI/policy paths that require optional backends.
    """
    global _populated
    if _populated:
        return

    _import_native_model_modules()

    timm = _timm_module()
    if timm is not None:
        for model_name in timm_model_names():
            if not FEATURE_EXTRACTORS.is_available(model_name):

                @FEATURE_EXTRACTORS.register(model_name)
                def _timm_factory(name=model_name, pretrained=True, **kwargs):
                    return timm.create_model(name, pretrained=pretrained, **kwargs)

    zs = _lazyslide_module()
    if zs is not None:
        for model_name in lazyslide_model_names():
            LAZYSLIDE_MODEL_NAMES.add(model_name)
            if not FEATURE_EXTRACTORS.is_available(model_name):

                @FEATURE_EXTRACTORS.register(model_name)
                def _zs_factory(name=model_name, **kwargs):
                    return name

    if is_torchmil_available():
        from pathbench.adapters.torchmil.backend import register_torchmil_backend
        from pathbench.adapters.torchmil.heatmap_explainer import (
            register_torchmil_heatmap_explainer,
        )
        from pathbench.adapters.mil_lab.backend import (
            register_torchmil_fallback_aliases,
        )

        register_torchmil_backend()
        register_torchmil_fallback_aliases()
        register_torchmil_heatmap_explainer()

    if is_mil_lab_available():
        from pathbench.adapters.mil_lab.backend import register_mil_lab_backend

        register_mil_lab_backend()

    if is_torchmetrics_available() and not CLASSIFICATION_METRICS.is_available(
        "torchmetrics"
    ):
        from pathbench.adapters.metrics.classification import (
            TorchMetricsClassificationBackend,
        )

        CLASSIFICATION_METRICS.register("torchmetrics")(
            TorchMetricsClassificationBackend
        )

    if is_torchsurv_available():
        from pathbench.adapters.metrics.survival import TorchSurvBackend

        if not SURVIVAL_METRICS.is_available("torchsurv"):
            SURVIVAL_METRICS.register("torchsurv")(TorchSurvBackend)
        if not SURVIVAL_LOSSES.is_available("torchsurv"):
            SURVIVAL_LOSSES.register("torchsurv")(TorchSurvBackend)

    _populated = True


def _sorted_catalog(entries: list[BackendCatalogEntry]) -> list[BackendCatalogEntry]:
    """Return catalog entries sorted by backend and then name."""

    return sorted(entries, key=lambda item: (item.backend, item.name.casefold()))


def list_feature_extractors() -> list[BackendCatalogEntry]:
    """List user-selectable feature extractors across supported backends.

    Returns:
        list[BackendCatalogEntry]: Catalog entries sorted by backend and name.
            For optional backends such as ``timm`` and ``lazyslide``, only
            installed catalogs can be enumerated because their model lists come
            from the backend package itself.

    Example:
        ```python
        entries = list_feature_extractors()
        lazyslide_only = [item.name for item in entries if item.backend == "lazyslide"]
        ```
    """

    config_field = "benchmark_parameters.feature_extraction"
    entries = [
        BackendCatalogEntry(
            name=name,
            backend="native",
            config_field=config_field,
            source="pathbench",
            available=True,
        )
        for name in registered_feature_extractor_names()
    ]
    entries.extend(
        BackendCatalogEntry(
            name=name,
            backend="timm",
            config_field=config_field,
            source="timm",
            available=True,
        )
        for name in timm_model_names()
    )
    entries.extend(
        BackendCatalogEntry(
            name=name,
            backend="lazyslide",
            config_field=config_field,
            source="lazyslide",
            available=True,
        )
        for name in lazyslide_model_names()
    )
    return _sorted_catalog(entries)


def list_mil_models() -> list[BackendCatalogEntry]:
    """List user-selectable MIL models across native and adapter backends.

    Returns:
        list[BackendCatalogEntry]: Catalog entries sorted by backend and name.
            Native PathBench MIL models are always listed. Backend-adapter model
            catalogs are listed even when their backend is unavailable so the
            caller can present supported choices together with installation
            requirements.

    Example:
        ```python
        entries = list_mil_models()
        mil_lab_models = [item.name for item in entries if item.backend == "mil-lab"]
        ```
    """

    from pathbench.adapters.mil_lab.backend import MILLAB_MODEL_SPECS
    from pathbench.adapters.torchmil.backend import TORCHMIL_MODEL_SPECS

    _import_native_model_modules()

    entries = [
        BackendCatalogEntry(
            name=name,
            backend="native",
            config_field="benchmark_parameters.mil",
            source="pathbench",
            available=MODELS.is_available(name),
        )
        for name in _NATIVE_MIL_MODEL_NAMES
    ]

    entries.extend(
        BackendCatalogEntry(
            name=name,
            backend="native",
            config_field="benchmark_parameters.mil",
            source="pathbench",
            available=MODELS.is_available(name),
        )
        for name in _OPTIONAL_NATIVE_MIL_MODELS
    )
    entries.extend(
        BackendCatalogEntry(
            name=name,
            backend="torchmil",
            config_field="mil.torchmil_model",
            source="torchmil",
            available=is_torchmil_available(),
        )
        for name in TORCHMIL_MODEL_SPECS
    )
    entries.extend(
        BackendCatalogEntry(
            name=name,
            backend="mil-lab",
            config_field="mil.mil_lab_model",
            source="mil-lab",
            available=is_mil_lab_available(),
        )
        for name in MILLAB_MODEL_SPECS
    )
    return _sorted_catalog(entries)
