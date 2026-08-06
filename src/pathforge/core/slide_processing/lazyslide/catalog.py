"""Discovery helpers for feature extractors executable by LazySlide."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


def _normalize_model_names(items: Any) -> set[str]:
    """Convert a backend model listing into a set of configured model names."""
    names: set[str] = set()
    for item in items or ():
        if isinstance(item, str):
            names.add(item)
        elif hasattr(item, "key"):
            names.add(str(item.key))
        elif isinstance(item, dict) and "key" in item:
            names.add(str(item["key"]))
        else:
            names.add(str(item))
    return names


@lru_cache(maxsize=1)
def timm_model_names() -> set[str]:
    """Return TIMM model names visible in the current Python environment."""
    try:
        import timm

        return _normalize_model_names(timm.list_models())
    except Exception:
        return set()


@lru_cache(maxsize=1)
def lazyslide_model_names() -> set[str]:
    """Return LazySlide model names visible in the current Python environment."""
    try:
        import lazyslide as zs

        return _normalize_model_names(zs.models.list_models())
    except Exception:
        return set()
