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
from pathforge.core.feature_extractors import registry as feature_registry
from pathforge.core.feature_extractors.base import FeatureExtractorBase
from pathforge.utils.registries import FEATURE_EXTRACTORS as SHARED_FEATURE_EXTRACTORS
from pathforge.utils.registry import FeatureExtractorRegistry, Registry


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


def test_feature_extractor_registry_reexports_shared_registry_instance() -> None:
    """The feature-extractor API uses the existing global registry object."""
    assert feature_registry.FEATURE_EXTRACTORS is SHARED_FEATURE_EXTRACTORS
    assert build_feature_extractor is feature_registry.build_feature_extractor
    assert get_feature_extractor is feature_registry.get_feature_extractor
    assert (
        is_native_feature_extractor_available
        is feature_registry.is_native_feature_extractor_available
    )
    assert list_native_feature_extractors is feature_registry.list_native_feature_extractors
    assert register_feature_extractor is feature_registry.register_feature_extractor


def test_register_get_and_build_native_feature_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native extractors can be registered, retrieved, and constructed with kwargs."""
    registry = Registry()
    monkeypatch.setattr(feature_registry, "FEATURE_EXTRACTORS", registry)

    feature_registry.register_feature_extractor("example")(ExampleFeatureExtractor)

    assert feature_registry.get_feature_extractor("example") is ExampleFeatureExtractor
    extractor = feature_registry.build_feature_extractor("example", embedding_size=8)
    assert isinstance(extractor, ExampleFeatureExtractor)
    assert extractor.embedding_size == 8


def test_feature_extractor_registry_rejects_external_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared registry cannot be populated with third-party factories."""
    registry = FeatureExtractorRegistry()
    monkeypatch.setattr(feature_registry, "FEATURE_EXTRACTORS", registry)
    feature_registry.register_feature_extractor("native")(ExampleFeatureExtractor)

    assert feature_registry.is_native_feature_extractor_available("native") is True
    assert feature_registry.is_native_feature_extractor_available("missing") is False
    assert feature_registry.list_native_feature_extractors() == ["native"]

    with pytest.raises(TypeError, match="FeatureExtractorBase"):
        registry.register("timm_entry")(lambda: object())


def test_register_feature_extractor_rejects_non_native_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registration rejects functions and classes outside the native base contract."""
    monkeypatch.setattr(feature_registry, "FEATURE_EXTRACTORS", Registry())

    with pytest.raises(TypeError, match="FeatureExtractorBase"):
        feature_registry.register_feature_extractor("invalid")(object)

    with pytest.raises(TypeError, match="FeatureExtractorBase"):
        feature_registry.register_feature_extractor("invalid_factory")(lambda: object())


def test_build_feature_extractor_rejects_legacy_non_native_registry_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building still rejects non-native entries in a legacy untyped registry."""
    registry = Registry()
    monkeypatch.setattr(feature_registry, "FEATURE_EXTRACTORS", registry)
    registry.register("timm_entry")(lambda: object())

    with pytest.raises(TypeError, match="not a native PathForge feature extractor"):
        feature_registry.build_feature_extractor("timm_entry")
