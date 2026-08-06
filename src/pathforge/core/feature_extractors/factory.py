"""Factory operations over the shared PathForge feature-extractor registry."""

from __future__ import annotations

from collections.abc import Callable
from pathforge.core.feature_extractors.base import FeatureExtractorBase


def _registry():
    """Return the shared registry after its global owner has initialized it."""
    from pathforge.utils.registries import FEATURE_EXTRACTORS

    return FEATURE_EXTRACTORS


def _is_native_feature_extractor(entry: object) -> bool:
    """Return whether one registry entry is a native extractor class."""
    return isinstance(entry, type) and issubclass(entry, FeatureExtractorBase)


def register_feature_extractor(
    name: str,
) -> Callable[[type[FeatureExtractorBase]], type[FeatureExtractorBase]]:
    """Register a native extractor class under ``name``.

    Args:
        name: Unique configured feature-extractor name.

    Returns:
        Decorator accepting a ``FeatureExtractorBase`` subclass.

    Example:
        >>> @register_feature_extractor("my_extractor")
        ... class MyExtractor(FeatureExtractorBase):
        ...     ...
    """
    def register(
        extractor_class: type[FeatureExtractorBase],
    ) -> type[FeatureExtractorBase]:
        if not _is_native_feature_extractor(extractor_class):
            raise TypeError(
                "Feature extractors must be FeatureExtractorBase subclasses; "
                f"received {extractor_class!r}."
            )
        return _registry().register(name)(extractor_class)

    return register


def get_feature_extractor(name: str) -> type[FeatureExtractorBase]:
    """Return the registered native extractor class for ``name``.

    Args:
        name: Configured feature-extractor name.
    Returns:
        Registered ``FeatureExtractorBase`` subclass.

    Example:
        >>> extractor_class = get_feature_extractor("my_extractor")
    """
    extractor_class = _registry().get(name)
    if not _is_native_feature_extractor(extractor_class):
        raise TypeError(
            f"Registry entry '{name}' is not a native PathForge feature extractor."
        )
    return extractor_class


def build_feature_extractor(name: str) -> FeatureExtractorBase:
    """Construct a registered native extractor by its configured name.

    Args:
        name: Configured feature-extractor name.
    Returns:
        Constructed extractor with its model in evaluation mode.

    Example:
        >>> extractor = build_feature_extractor("my_extractor")
    """
    return get_feature_extractor(name)()


def is_native_feature_extractor_available(name: str) -> bool:
    """Return whether ``name`` identifies a registered native extractor class."""
    registry = _registry()
    return registry.is_available(name) and _is_native_feature_extractor(registry.get(name))


def list_native_feature_extractors() -> list[str]:
    """List registered native extractor names in insertion order."""
    return [
        name
        for name in _registry().list_plugins()
        if is_native_feature_extractor_available(name)
    ]


def registered_feature_extractor_names() -> set[str]:
    """Return registered PathForge-native extractor names as a set."""
    return set(list_native_feature_extractors())


def available_feature_extractor_names(backend_name: str) -> set[str]:
    """Return feature-extractor names supported by one selected processor.

    Processor-native names are always available. Registered PathForge names
    are included only when the processor supports wrapping native extractors.
    """
    from pathforge.core.slide_processing.factory import build_slide_processor
    from pathforge.utils.registries import populate_pathforge_feature_extractors

    populate_pathforge_feature_extractors()
    processor = build_slide_processor(backend_name)
    native_names = processor.native_feature_extractor_names()
    if processor.supports_pathforge_feature_extractors():
        return set(native_names) | registered_feature_extractor_names()
    return set(native_names)


def resolve_feature_extractor_source(backend_name: str, name: str) -> str:
    """Resolve one configured extractor name using processor-native precedence."""
    from pathforge.core.feature_extractors.selection import (
        resolve_feature_extractor_selection,
    )

    return resolve_feature_extractor_selection(backend_name, name).source
