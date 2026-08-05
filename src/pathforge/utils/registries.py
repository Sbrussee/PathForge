from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import Any

from pathforge.adapters.losses import register_builtin_loss_factories
from pathforge.core.base import CoreRegistries
from pathforge.utils.optional.mil_lab import is_mil_lab_available
from pathforge.utils.optional.torchmil import (
    is_torchmetrics_available,
    is_torchmil_available,
    is_torchsurv_available,
)
from pathforge.utils.registry import FeatureExtractorRegistry, Registry

# Define Registries
REGISTRIES = CoreRegistries(
    datasets=Registry(),
    models=Registry(),
    losses=Registry(),
    tasks=Registry(),
    explainers=Registry(),
    feature_extractors=FeatureExtractorRegistry(),
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
        .. code-block:: python

            entries = list_mil_models()
            torchmil_names = [item.name for item in entries if item.backend == "torchmil"]

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
    """Return the TIMM model names visible in the current Python environment."""
    timm = _timm_module()
    if timm is None:
        return set()
    try:
        return _normalize_model_names(timm.list_models())
    except Exception:
        return set()


@lru_cache(maxsize=1)
def lazyslide_model_names() -> set[str]:
    """Return the LazySlide model names visible in the current Python environment."""
    zs = _lazyslide_module()
    if zs is None:
        return set()
    try:
        return _normalize_model_names(zs.models.list_models())
    except Exception:
        return set()


def registered_feature_extractor_names() -> set[str]:
    """Return names currently registered in PathForge ``FEATURE_EXTRACTORS``.

    Returns:
        Registered PathForge-native extractor names in an unordered set.

    Example:
        >>> "my_extractor" in registered_feature_extractor_names()
        True
    """
    return set(FEATURE_EXTRACTORS.list_plugins())


def discovered_feature_extractor_names() -> dict[str, set[str]]:
    """Return extractor names grouped by the backend that provides them."""
    return {
        "pathforge": registered_feature_extractor_names(),
        "timm": timm_model_names(),
        "lazyslide": lazyslide_model_names(),
    }


def all_feature_extractor_names() -> set[str]:
    """Return the union of PathForge-native and dynamically discovered extractors."""
    grouped = discovered_feature_extractor_names()
    return grouped["pathforge"] | grouped["timm"] | grouped["lazyslide"]


def _build_slide_processor(backend_name: str) -> Any:
    """Load and construct the processor registered for ``backend_name``.

    Args:
        backend_name: Registry key from ``slide_processing.backend``.

    Returns:
        Configured-backend processor instance.

    Raises:
        ValueError: If the backend cannot be imported or has no processor registration.

    Example:
        >>> processor = _build_slide_processor("lazyslide")
    """
    if not SLIDE_PROCESSORS.is_available(backend_name):
        try:
            import_module(f"pathforge.core.slide_processing.{backend_name}")
        except ModuleNotFoundError as exc:
            raise ValueError(
                f"Slide processing backend '{backend_name}' is not available."
            ) from exc

    if not SLIDE_PROCESSORS.is_available(backend_name):
        raise ValueError(f"Slide processing backend '{backend_name}' is not registered.")

    return SLIDE_PROCESSORS.get(backend_name)()


def available_feature_extractor_names(backend_name: str) -> set[str]:
    """Return feature-extractor names valid for one slide-processing backend.

    Args:
        backend_name: Selected ``slide_processing.backend`` registry key.

    Returns:
        Union of the processor-native encoder names and registered PathForge
        feature-extractor names. The union is transient and never mutates the
        global PathForge registry.

    Raises:
        ValueError: If ``backend_name`` cannot be resolved to a processor.

    Example:
        >>> names = available_feature_extractor_names("lazyslide")
    """
    populate_pathforge_feature_extractors()
    processor = _build_slide_processor(backend_name)
    native_names = processor.native_feature_extractor_names()
    return set(native_names) | registered_feature_extractor_names()


# ---------------------------------------------------------------------------
# Dynamic registry population (explicit, not at import time)
# ---------------------------------------------------------------------------

_populated = False

_feature_extractors_populated = False

_NATIVE_FEATURE_EXTRACTOR_MODULES: tuple[str, ...] = ()

_NATIVE_MODEL_MODULES: tuple[str, ...] = (
    "pathforge.core.models.perceiver_mil",
    "pathforge.core.models.prototype_mil",
    "pathforge.core.models.slide_mlp",
    "pathforge.core.models.var_mil",
)

_OPTIONAL_NATIVE_MODEL_MODULES: dict[str, str] = {
    "pathforge.core.models.mamba_mil": "mamba",
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
    """Import native PathForge model modules so their registry decorators run."""

    for module_name in _NATIVE_MODEL_MODULES:
        import_module(module_name)

    for module_name, dependency_name in _OPTIONAL_NATIVE_MODEL_MODULES.items():
        try:
            __import__(dependency_name)
        except Exception:
            continue
        import_module(module_name)


def populate_pathforge_feature_extractors() -> None:
    """Import built-in PathForge extractor modules so their decorators register them.

    The PathForge extractor registry intentionally contains only native
    ``FeatureExtractorBase`` implementations. Processor-owned encoders, such
    as LazySlide and timm models, are discovered by the selected processor and
    are never registered here.

    Example:
        >>> populate_pathforge_feature_extractors()
    """
    global _feature_extractors_populated
    if _feature_extractors_populated:
        return

    for module_name in _NATIVE_FEATURE_EXTRACTOR_MODULES:
        import_module(module_name)

    _feature_extractors_populated = True


def populate_dynamic_registries() -> None:
    """
    Populate all optional backend registries with entries from installed packages.

    IMPORTANT:
    - This is NOT called automatically at import time.
    - Call it explicitly only in paths that require the complete catalog.
    - Feature-extractor-only paths should call
      ``available_feature_extractor_names()`` instead.
    """
    global _populated
    if _populated:
        return

    populate_pathforge_feature_extractors()
    _import_native_model_modules()

    if is_torchmil_available():
        from pathforge.adapters.mil_lab.backend import (
            register_torchmil_fallback_aliases,
        )
        from pathforge.adapters.torchmil.backend import register_torchmil_backend
        from pathforge.adapters.torchmil.heatmap_explainer import (
            register_torchmil_heatmap_explainer,
        )

        register_torchmil_backend()
        register_torchmil_fallback_aliases()
        register_torchmil_heatmap_explainer()

    if is_mil_lab_available():
        from pathforge.adapters.mil_lab.backend import register_mil_lab_backend

        register_mil_lab_backend()

    if is_torchmetrics_available() and not CLASSIFICATION_METRICS.is_available(
        "torchmetrics"
    ):
        from pathforge.adapters.metrics.classification import (
            TorchMetricsClassificationBackend,
        )

        CLASSIFICATION_METRICS.register("torchmetrics")(
            TorchMetricsClassificationBackend
        )

    if is_torchsurv_available():
        from pathforge.adapters.metrics.survival import TorchSurvBackend

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
        .. code-block:: python

            entries = list_feature_extractors()
            lazyslide_only = [item.name for item in entries if item.backend == "lazyslide"]

    """

    config_field = "benchmark_parameters.feature_extraction"
    entries = [
        BackendCatalogEntry(
            name=name,
            backend="native",
            config_field=config_field,
            source="pathforge",
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
            Native PathForge MIL models are always listed. Backend-adapter model
            catalogs are listed even when their backend is unavailable so the
            caller can present supported choices together with installation
            requirements.

    Example:
        .. code-block:: python

            entries = list_mil_models()
            mil_lab_models = [item.name for item in entries if item.backend == "mil-lab"]

    """

    from pathforge.adapters.mil_lab.backend import MILLAB_MODEL_SPECS
    from pathforge.adapters.torchmil.backend import TORCHMIL_MODEL_SPECS

    _import_native_model_modules()

    entries = [
        BackendCatalogEntry(
            name=name,
            backend="native",
            config_field="benchmark_parameters.mil",
            source="pathforge",
            available=MODELS.is_available(name),
        )
        for name in _NATIVE_MIL_MODEL_NAMES
    ]

    entries.extend(
        BackendCatalogEntry(
            name=name,
            backend="native",
            config_field="benchmark_parameters.mil",
            source="pathforge",
            available=MODELS.is_available(name),
        )
        for name in _OPTIONAL_NATIVE_MIL_MODELS
    )
    entries.extend(
        BackendCatalogEntry(
            name=name,
            backend="torchmil",
            config_field="benchmark_parameters.mil",
            source="torchmil",
            available=is_torchmil_available(),
        )
        for name in TORCHMIL_MODEL_SPECS
    )
    entries.extend(
        BackendCatalogEntry(
            name=name,
            backend="mil-lab",
            config_field="benchmark_parameters.mil",
            source="mil-lab",
            available=is_mil_lab_available(),
        )
        for name in MILLAB_MODEL_SPECS
    )
    return _sorted_catalog(entries)


def resolve_mil_model_backend(name: str) -> str:
    """Return the backend providing a selectable MIL model name.

    The generic ``torchmil`` and ``mil-lab`` keys remain supported for legacy
    configs. New benchmark grids should use a concrete name returned by
    :func:`list_mil_models`.
    """

    normalized_name = str(name).strip()
    if normalized_name in {"torchmil", "mil-lab"}:
        return normalized_name
    matches = {entry.backend for entry in list_mil_models() if entry.name == normalized_name}
    if not matches:
        if MODELS.is_available(normalized_name):
            return "native"
        raise ValueError(f"MIL model '{normalized_name}' is not in the backend catalog.")
    if len(matches) > 1:
        raise ValueError(
            f"MIL model '{normalized_name}' is ambiguous across backends: "
            f"{sorted(matches)}. Use a backend-specific model name."
        )
    return matches.pop()
