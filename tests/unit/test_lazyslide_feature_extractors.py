"""Unit tests for PathForge-native extractor execution through LazySlide."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch
from torch import Tensor

from pathforge.core.feature_extractors.base import FeatureExtractorBase
from pathforge.core.slide_processing.lazyslide.feature_extractors import (
    LazySlideFeatureExtractorAdapter,
)


class _ExampleExtractor(FeatureExtractorBase):
    """Small native extractor used to exercise LazySlide adapter behavior."""

    def build_model(self) -> torch.nn.Module:
        return torch.nn.Identity()

    def get_transform(self) -> str:
        return "example-transform"

    def encode_images(self, images: Tensor) -> Tensor:
        return images + 1


def test_lazyslide_adapter_exposes_image_model_protocol() -> None:
    """The adapter delegates transformation, device movement, and batch encoding."""
    extractor = _ExampleExtractor()
    adapter = LazySlideFeatureExtractorAdapter("example", extractor)
    images = torch.zeros((2, 3, 4, 4))

    assert adapter.name == "example"
    assert adapter.get_transform() == "example-transform"
    assert torch.equal(adapter.encode_image(images), torch.ones_like(images))
    assert adapter.to("cpu") is adapter


def test_lazyslide_adapter_normalizes_backend_tiles_before_native_transform() -> None:
    """LazySlide supplies PathForge transforms with canonical RGB NumPy patches."""
    extractor = _ExampleExtractor()
    adapter = LazySlideFeatureExtractorAdapter("example", extractor)

    def transform(patch: np.ndarray) -> torch.Tensor:
        assert patch.dtype == np.uint8
        assert patch.shape == (4, 5, 3)
        return torch.from_numpy(patch).permute(2, 0, 1)

    extractor.get_transform = lambda: transform
    transformed = adapter.get_transform()(np.ones((3, 4, 5), dtype=np.float32))

    assert transformed.shape == (3, 4, 5)
    assert transformed.dtype == torch.uint8


def test_lazyslide_processor_uses_adapter_for_pathforge_extractors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PathForge resolutions construct an extractor and return a LazySlide adapter."""
    pytest.importorskip("lazyslide")
    import pathforge.core.feature_extractors.factory as feature_factory
    import pathforge.core.slide_processing.lazyslide as lazyslide_module

    extractor = _ExampleExtractor()
    monkeypatch.setattr(
        feature_factory,
        "resolve_feature_extractor_source",
        lambda backend_name, name: "pathforge-native",
    )
    build_calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        lazyslide_module,
        "build_feature_extractor",
        lambda name, **kwargs: build_calls.append((name, kwargs)) or extractor,
    )

    model = lazyslide_module.LazySlideProcessor()._resolve_feature_extractor(
        "example",
        {"weights": "test"},
    )

    assert isinstance(model, LazySlideFeatureExtractorAdapter)
    assert model.name == "example"
    assert build_calls == [("example", {"weights": "test"})]


def test_lazyslide_processor_prefers_processor_native_collision(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Processor-native names win collisions and announce the ignored registration."""
    pytest.importorskip("lazyslide")
    import pathforge.core.feature_extractors.factory as feature_factory
    import pathforge.core.slide_processing.lazyslide as lazyslide_module

    monkeypatch.setattr(
        feature_factory,
        "resolve_feature_extractor_source",
        lambda backend_name, name: "processor-native",
    )
    monkeypatch.setattr(
        feature_factory,
        "registered_feature_extractor_names",
        lambda: {"shared"},
    )
    monkeypatch.setattr(
        lazyslide_module,
        "build_feature_extractor",
        lambda name, **kwargs: pytest.fail("native names must not be constructed"),
    )

    with caplog.at_level(logging.INFO, logger=lazyslide_module.__name__):
        model = lazyslide_module.LazySlideProcessor()._resolve_feature_extractor(
            "shared",
            {},
        )

    assert model == "shared"
    assert "takes precedence" in caplog.text
