"""Discovery helpers for feature extractors executable by LazySlide."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pathforge.core.feature_extractors.selection import (
    FeatureOutputLevel,
    determine_feature_extractor_output_level,
    parse_feature_extractor_name,
)


def _supports_patch_embeddings(model_class: type) -> bool:
    """Check registry class metadata and image encoding without loading weights.

    Args:
        model_class: A LazySlide registry class with scalar or list task metadata.
    Returns:
        Whether the class exposes image encoding for vision/multimodal embeddings.
    Example:
        >>> _supports_patch_embeddings(MODEL_REGISTRY["conch"])
        True
    """
    tasks = getattr(model_class, "task", ())
    if not isinstance(tasks, (list, tuple, set)):
        tasks = (tasks,)
    tasks = {getattr(task, "value", task) for task in tasks}
    return bool(tasks.intersection({"vision", "multimodal"})) and callable(
        getattr(model_class, "encode_image", None)
    )


def resolve_lazyslide_embedding(
    name: str,
    *,
    output_level: FeatureOutputLevel | None = None,
) -> tuple[str, str | None]:
    """Resolve one registered embedding request without constructing models.

    Args:
        name: Backend registry key when output_level is supplied; otherwise a
            PathForge configured name for catalog validation.
        output_level: Explicit requested level from the extraction policy.
    Returns:
        ``(patch_encoder, slide_encoder)`` names; the second is ``None`` for
        patch output. Both encoders must expose the required encoding methods.
    Raises:
        ValueError: The requested model cannot produce the requested embeddings.
    Example:
        >>> resolve_lazyslide_embedding("titan-slide")
        ('titan', 'titan')
    """
    from lazyslide_models import MODEL_REGISTRY

    if output_level is None:
        output_level = determine_feature_extractor_output_level(name)
        model_name = parse_feature_extractor_name(name)
    else:
        model_name = name
    slide_output = output_level == "slide"
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown LazySlide embedding model '{model_name}'.")
    model_class = MODEL_REGISTRY[model_name]
    tasks = getattr(model_class, "task", ())
    if not isinstance(tasks, (list, tuple, set)):
        tasks = (tasks,)
    tasks = {getattr(task, "value", task) for task in tasks}

    if slide_output:
        if "slide_encoder" not in tasks or not callable(
            getattr(model_class, "encode_slide", None)
        ):
            raise ValueError(
                f"LazySlide model '{model_name}' does not support slide embeddings."
            )
        patch_name = getattr(model_class, "vision_encoder", None)
        if not isinstance(patch_name, str) or not patch_name:
            raise ValueError(f"LazySlide model '{model_name}' has no patch encoder metadata.")
        # Inspect the patch class directly; avoid following cyclic encoder references.
        if patch_name not in MODEL_REGISTRY or not _supports_patch_embeddings(
            MODEL_REGISTRY[patch_name]
        ):
            raise ValueError(
                f"LazySlide model '{model_name}' requires an unavailable "
                f"patch encoder '{patch_name}'."
            )
        return patch_name, model_name

    if not _supports_patch_embeddings(model_class):
        raise ValueError(
            f"LazySlide model '{model_name}' does not support patch embeddings."
        )
    return model_name, None


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

        names = {
            name
            for name in _normalize_model_names(timm.list_models())
            if not name.endswith("-slide")
        }
        # LazySlide dispatches registry keys before timm, so colliding unsupported
        # registry models must not reappear via the timm catalog.
        try:
            from lazyslide_models import MODEL_REGISTRY
        except ImportError:
            return names
        return names - set(MODEL_REGISTRY)
    except Exception:
        return set()


@lru_cache(maxsize=1)
def lazyslide_model_names() -> set[str]:
    """Return valid patch names and ``-slide`` requests from the installed registry.

    Returns:
        Names usable for embeddings only; prediction, segmentation and image
        generation models are excluded without loading their weights.
    Example:
        >>> "titan-slide" in lazyslide_model_names()
        True
    """
    try:
        from lazyslide_models import MODEL_REGISTRY
    except ImportError:
        return set()
    names = set()
    for key in MODEL_REGISTRY:
        for name in (key, f"{key}-slide"):
            try:
                resolve_lazyslide_embedding(name)
            except ValueError:
                continue
            names.add(name)
    return names
