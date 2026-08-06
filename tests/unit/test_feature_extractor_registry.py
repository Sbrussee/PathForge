"""Unit tests for PathForge-native feature-extractor registry helpers."""

from __future__ import annotations

import pytest

from pathforge.core.feature_extractors import (
    build_feature_extractor,
    get_feature_extractor,
    is_native_feature_extractor_available,
    list_native_feature_extractors,
    register_feature_extractor,
)
from pathforge.core.feature_extractors import factory as feature_factory
from pathforge.core.feature_extractors.base import FeatureExtractorBase
from pathforge.core.feature_extractors.registry import FeatureExtractorRegistry
from pathforge.core.feature_extractors.selection import (
    resolve_feature_extractor_selection,
)
from pathforge.utils.registries import FEATURE_EXTRACTORS as SHARED_FEATURE_EXTRACTORS
from pathforge.utils.registry import Registry


class ExampleFeatureExtractor(FeatureExtractorBase):
    """Minimal native extractor used to test registry behavior."""

    def __init__(self, embedding_size: int = 4) -> None:
        self.embedding_size = embedding_size
        super().__init__()

    def build_model(self) -> object:
        return object()

    def get_transform(self) -> None:
        return None

    def encode_images(self, images):  # type: ignore[no-untyped-def]
        return images


def test_selection_prefers_processor_native_without_constructing_an_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection reports processor-native precedence without building a model."""

    class Processor:
        def native_feature_extractor_names(self) -> set[str]:
            return {"shared"}

        def supports_pathforge_feature_extractors(self) -> bool:
            return True

    monkeypatch.setattr(
        "pathforge.core.slide_processing.factory.build_slide_processor",
        lambda backend_name: Processor(),
    )
    monkeypatch.setattr(
        "pathforge.utils.registries.populate_pathforge_feature_extractors",
        lambda: None,
    )
    monkeypatch.setattr(feature_factory, "is_native_feature_extractor_available", lambda name: True)

    selection = resolve_feature_extractor_selection("test-processor", "shared")

    assert selection.name == "shared"
    assert selection.source == "processor-native"
    assert selection.requires_pathforge_adapter is False


def test_selection_uses_pathforge_native_extractor_when_backend_supports_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection marks registered native extractors for backend adaptation."""

    class Processor:
        def native_feature_extractor_names(self) -> set[str]:
            return set()

        def supports_pathforge_feature_extractors(self) -> bool:
            return True

    monkeypatch.setattr(
        "pathforge.core.slide_processing.factory.build_slide_processor",
        lambda backend_name: Processor(),
    )
    monkeypatch.setattr(
        "pathforge.utils.registries.populate_pathforge_feature_extractors",
        lambda: None,
    )
    monkeypatch.setattr(
        feature_factory, "is_native_feature_extractor_available", lambda name: name == "native"
    )

    selection = resolve_feature_extractor_selection("test-processor", "native")

    assert selection.source == "pathforge-native"
    assert selection.requires_pathforge_adapter is True


def test_selection_rejects_pathforge_native_extractor_without_adapter_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backends without an adapter cannot select PathForge-native extractors."""

    class Processor:
        def native_feature_extractor_names(self) -> set[str]:
            return set()

        def supports_pathforge_feature_extractors(self) -> bool:
            return False

    monkeypatch.setattr(
        "pathforge.core.slide_processing.factory.build_slide_processor",
        lambda backend_name: Processor(),
    )
    monkeypatch.setattr(
        "pathforge.utils.registries.populate_pathforge_feature_extractors",
        lambda: None,
    )

    with pytest.raises(ValueError, match="not available"):
        resolve_feature_extractor_selection("test-processor", "native")


def test_registered_feature_extractor_names_excludes_non_native_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Availability helpers never advertise legacy factory registry entries."""
    registry = Registry()
    monkeypatch.setattr(feature_factory, "_registry", lambda: registry)
    registry.register("legacy")(lambda: object())
    feature_factory.register_feature_extractor("native")(ExampleFeatureExtractor)

    assert feature_factory.registered_feature_extractor_names() == {"native"}


def test_feature_extractor_registry_reexports_shared_registry_instance() -> None:
    """The feature-extractor API uses the existing global registry object."""
    assert feature_factory._registry() is SHARED_FEATURE_EXTRACTORS
    assert build_feature_extractor is feature_factory.build_feature_extractor
    assert get_feature_extractor is feature_factory.get_feature_extractor
    assert (
        is_native_feature_extractor_available
        is feature_factory.is_native_feature_extractor_available
    )
    assert list_native_feature_extractors is feature_factory.list_native_feature_extractors
    assert register_feature_extractor is feature_factory.register_feature_extractor


def test_register_get_and_build_native_feature_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native extractors can be registered, retrieved, and constructed by name."""
    registry = Registry()
    monkeypatch.setattr(feature_factory, "_registry", lambda: registry)

    feature_factory.register_feature_extractor("example")(ExampleFeatureExtractor)

    assert feature_factory.get_feature_extractor("example") is ExampleFeatureExtractor
    extractor = feature_factory.build_feature_extractor("example")
    assert isinstance(extractor, ExampleFeatureExtractor)
    assert extractor.embedding_size == 4


def test_feature_extractor_registry_rejects_external_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared registry cannot be populated with third-party factories."""
    registry = FeatureExtractorRegistry()
    monkeypatch.setattr(feature_factory, "_registry", lambda: registry)
    feature_factory.register_feature_extractor("native")(ExampleFeatureExtractor)

    assert feature_factory.is_native_feature_extractor_available("native") is True
    assert feature_factory.is_native_feature_extractor_available("missing") is False
    assert feature_factory.list_native_feature_extractors() == ["native"]

    with pytest.raises(TypeError, match="FeatureExtractorBase"):
        registry.register("timm_entry")(lambda: object())


def test_register_feature_extractor_rejects_non_native_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration rejects functions and classes outside the native base contract."""
    monkeypatch.setattr(feature_factory, "_registry", lambda: Registry())

    with pytest.raises(TypeError, match="FeatureExtractorBase"):
        feature_factory.register_feature_extractor("invalid")(object)

    with pytest.raises(TypeError, match="FeatureExtractorBase"):
        feature_factory.register_feature_extractor("invalid_factory")(lambda: object())


def test_build_feature_extractor_rejects_legacy_non_native_registry_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building still rejects non-native entries in a legacy untyped registry."""
    registry = Registry()
    monkeypatch.setattr(feature_factory, "_registry", lambda: registry)
    registry.register("timm_entry")(lambda: object())

    with pytest.raises(TypeError, match="not a native PathForge feature extractor"):
        feature_factory.build_feature_extractor("timm_entry")
