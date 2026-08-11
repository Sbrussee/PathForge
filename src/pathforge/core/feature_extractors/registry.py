"""Registry type for PathForge-native feature extractors."""

from __future__ import annotations

from collections.abc import Callable
from pathforge.core.feature_extractors.base import FeatureExtractorBase
from pathforge.utils.registry import Registry


class FeatureExtractorRegistry(Registry):
    """Registry that accepts only ``FeatureExtractorBase`` subclasses.

    Example:
        >>> registry = FeatureExtractorRegistry()
        >>> registry.register("my_extractor")(MyExtractor)
    """

    def register(
        self, name: str
    ) -> Callable[[type[FeatureExtractorBase]], type[FeatureExtractorBase]]:
        """Return a decorator that registers one native extractor class."""
        register = super().register(name)

        def validate_and_register(
            extractor_class: type[FeatureExtractorBase],
        ) -> type[FeatureExtractorBase]:
            if not (
                isinstance(extractor_class, type)
                and issubclass(extractor_class, FeatureExtractorBase)
            ):
                raise TypeError(
                    "Feature extractors must be FeatureExtractorBase subclasses; "
                    f"received {extractor_class!r}."
                )
            return register(extractor_class)

        return validate_and_register
