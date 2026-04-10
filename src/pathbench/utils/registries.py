from __future__ import annotations

from functools import lru_cache
from typing import Any

from pathbench.utils.registry import Registry
from pathbench.core.base import CoreRegistries

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

# Track Lazyslide-specific models for validation (filled by populate_dynamic_registries)
LAZYSLIDE_MODEL_NAMES: set[str] = set()

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


def populate_dynamic_registries() -> None:
    """
    Populate FEATURE_EXTRACTORS with entries from timm and lazyslide.

    IMPORTANT:
    - This is NOT called automatically at import time.
    - Call it explicitly in CLI/policy paths that require timm/lazyslide.
    """
    global _populated
    if _populated:
        return

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

    _populated = True