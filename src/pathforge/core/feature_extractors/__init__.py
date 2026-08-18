"""Public API for PathForge-native feature extractors."""

from pathforge.core.feature_extractors.base import FeatureExtractorBase

__all__ = [
    "FEATURE_EXTRACTORS",
    "FeatureExtractorBase",
    "build_feature_extractor",
    "get_feature_extractor",
    "is_native_feature_extractor_available",
    "list_native_feature_extractors",
    "register_feature_extractor",
]


def __getattr__(name: str):
    """Lazily expose factory operations without creating registry cycles."""
    if name == "FEATURE_EXTRACTORS":
        from pathforge.utils.registries import FEATURE_EXTRACTORS

        return FEATURE_EXTRACTORS
    if name in __all__:
        from pathforge.core.feature_extractors import factory

        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
