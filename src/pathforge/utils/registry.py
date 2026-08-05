# pathforge/utils/registry.py
from collections.abc import Callable
from typing import Dict, Sequence, TypeVar

from pathforge.core.base import RegistryBase
from pathforge.core.feature_extractors.base import FeatureExtractorBase

T = TypeVar('T')

class Registry(RegistryBase):
    """Minimal string-to-callable registry used for runtime plugin lookup."""

    def __init__(self) -> None:
        self._f: Dict[str, Callable[..., T]] = {}

    def register(self, name: str):
        def deco(fn: Callable[..., T]) -> Callable[..., T]:
            if name in self._f:
                raise KeyError(f"Duplicate registration: {name}")
            self._f[name] = fn
            return fn
        return deco

    def get(self, name: str) -> Callable[..., T]:
        if name not in self._f:
            raise KeyError(f"Plugin '{name}' not found in registry")
        return self._f[name]

    def list_plugins(self) -> Sequence[str]:
        return list(self._f.keys())

    def is_available(self, key: str) -> bool:
        return key in self._f


class FeatureExtractorRegistry(Registry):
    """Registry that accepts only PathForge-native feature-extractor classes.

    Example:
        >>> @FeatureExtractorRegistry().register("my_extractor")
        ... class MyExtractor(FeatureExtractorBase):
        ...     ...
    """

    def register(
        self, name: str
    ) -> Callable[[type[FeatureExtractorBase]], type[FeatureExtractorBase]]:
        """Return a decorator that registers one native feature-extractor class.

        Args:
            name: Unique feature-extractor identifier.

        Returns:
            A decorator accepting a ``FeatureExtractorBase`` subclass.

        Raises:
            TypeError: If the decorated object is not a native extractor class.

        Example:
            >>> registry = FeatureExtractorRegistry()
            >>> registry.register("my_extractor")(MyExtractor)
        """
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
