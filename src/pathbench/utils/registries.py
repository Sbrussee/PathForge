# src/pathbench/utils/registries.py
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

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

# Track Lazyslide-specific models for validation
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


@lru_cache(maxsize=1)
def timm_model_names() -> set[str]:
    timm = _timm_module()
    if timm is None:
        return set()
    try:
        return set(timm.list_models())
    except Exception:
        return set()


@lru_cache(maxsize=1)
def lazyslide_model_names() -> set[str]:
    zs = _lazyslide_module()
    if zs is None:
        return set()
    try:
        return set(zs.models.list_models())
    except Exception:
        return set()


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
                    # Import torch lazily as well (only when actually constructing models)
                    return timm.create_model(name, pretrained=pretrained, **kwargs)

    zs = _lazyslide_module()
    if zs is not None:
        for model_name in lazyslide_model_names():
            LAZYSLIDE_MODEL_NAMES.add(model_name)
            if not FEATURE_EXTRACTORS.is_available(model_name):
                @FEATURE_EXTRACTORS.register(model_name)
                def _zs_factory(name=model_name, **kwargs):
                    # Lazyslide backends often take model name strings
                    return name

    _populated = True
