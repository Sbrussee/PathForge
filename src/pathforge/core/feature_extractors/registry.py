"""Registry helpers for PathForge-native feature extractors.

The shared registry deliberately also retains entries owned by external
backends, such as timm and LazySlide. The helpers here provide a typed API for
PathForge-native extractors without changing those existing registrations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pathforge.core.feature_extractors.base import FeatureExtractorBase
from pathforge.utils.registries import FEATURE_EXTRACTORS


def _is_native_feature_extractor(entry: object) -> bool:
    """Return whether a registry entry is a native feature-extractor class.

    Args:
        entry: Object returned by the shared feature-extractor registry.

    Returns:
        ``True`` when ``entry`` is a subclass of ``FeatureExtractorBase``.

    Example:
        >>> _is_native_feature_extractor(FeatureExtractorBase)
        True
    """
    return isinstance(entry, type) and issubclass(entry, FeatureExtractorBase)


def register_feature_extractor(
    name: str,
) -> Callable[[type[FeatureExtractorBase]], type[FeatureExtractorBase]]:
    """Register a PathForge-native feature-extractor class under ``name``.

    Args:
        name: Unique registry key used to select the extractor.

    Returns:
        A decorator accepting and returning a ``FeatureExtractorBase`` subclass.

    Raises:
        TypeError: If the decorated object is not a native extractor class.
        KeyError: If ``name`` is already present in the shared registry.

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
        return FEATURE_EXTRACTORS.register(name)(extractor_class)

    return register


def get_feature_extractor(name: str) -> object:
    """Retrieve any entry from the shared feature-extractor registry.

    Args:
        name: Registry key for a native or external-backend extractor.

    Returns:
        The object registered for ``name``.

    Raises:
        KeyError: If ``name`` is not registered.

    Example:
        >>> extractor = get_feature_extractor("my_extractor")
    """
    return FEATURE_EXTRACTORS.get(name)


def build_feature_extractor(name: str, **kwargs: Any) -> FeatureExtractorBase:
    """Construct a registered PathForge-native feature extractor.

    Args:
        name: Registry key for a native PathForge feature extractor.
        **kwargs: Keyword arguments forwarded to the extractor constructor.

    Returns:
        A constructed ``FeatureExtractorBase`` instance.

    Raises:
        KeyError: If ``name`` is not registered.
        TypeError: If the registered entry belongs to an external backend.

    Example:
        >>> extractor = build_feature_extractor("my_extractor", weights="default")
    """
    extractor_class = get_feature_extractor(name)
    if not _is_native_feature_extractor(extractor_class):
        raise TypeError(
            f"Registry entry '{name}' is not a native PathForge feature extractor."
        )
    return extractor_class(**kwargs)


def is_native_feature_extractor_available(name: str) -> bool:
    """Return whether ``name`` identifies a registered native extractor.

    Args:
        name: Registry key to inspect.

    Returns:
        ``True`` only for registered ``FeatureExtractorBase`` subclasses.

    Example:
        >>> is_native_feature_extractor_available("my_extractor")
        True
    """
    return FEATURE_EXTRACTORS.is_available(name) and _is_native_feature_extractor(
        get_feature_extractor(name)
    )


def list_native_feature_extractors() -> list[str]:
    """List names of registered PathForge-native feature extractors.

    Returns:
        Native extractor registry keys in registry insertion order.

    Example:
        >>> list_native_feature_extractors()
        ["my_extractor"]
    """
    return [
        name
        for name in FEATURE_EXTRACTORS.list_plugins()
        if is_native_feature_extractor_available(name)
    ]
