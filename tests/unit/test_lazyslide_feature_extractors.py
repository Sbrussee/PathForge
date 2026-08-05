"""Unit tests for PathForge-native extractor execution through LazySlide."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor

from pathforge.core.feature_extractors.base import FeatureExtractorBase
from pathforge.core.slide_processing.lazyslide_feature_extractors import (
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


class _FakeWsiObject(dict[str, object]):
    """Dictionary-like WSI object with the attrs mapping LazySlide expects."""

    def __init__(self) -> None:
        super().__init__()
        self.attrs: dict[str, object] = {}


def test_lazyslide_adapter_exposes_image_model_protocol() -> None:
    """The adapter delegates transformation, device movement, and batch encoding."""
    extractor = _ExampleExtractor()
    adapter = LazySlideFeatureExtractorAdapter("example", extractor)
    images = torch.zeros((2, 3, 4, 4))

    assert adapter.name == "example"
    assert adapter.get_transform() == "example-transform"
    assert torch.equal(adapter.encode_image(images), torch.ones_like(images))
    assert adapter.to("cpu") is adapter


def test_lazyslide_adapter_propagates_wrapped_extractor_errors() -> None:
    """Adapter failures retain the native extractor's error for callers to diagnose."""

    class FailingExtractor(_ExampleExtractor):
        def encode_images(self, images: Tensor) -> Tensor:
            raise RuntimeError("encoding failed")

    adapter = LazySlideFeatureExtractorAdapter("failing", FailingExtractor())

    with pytest.raises(RuntimeError, match="encoding failed"):
        adapter.encode_image(torch.zeros((1, 3, 4, 4)))


def test_lazyslide_processor_uses_adapter_for_pathforge_extractors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PathForge resolutions construct an extractor and return a LazySlide adapter."""
    pytest.importorskip("lazyslide")
    import pathforge.core.slide_processing.lazyslide as lazyslide_module
    import pathforge.utils.registries as registries_module

    extractor = _ExampleExtractor()
    monkeypatch.setattr(
        registries_module,
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
    import pathforge.core.slide_processing.lazyslide as lazyslide_module
    import pathforge.utils.registries as registries_module

    monkeypatch.setattr(
        registries_module,
        "resolve_feature_extractor_source",
        lambda backend_name, name: "processor-native",
    )
    monkeypatch.setattr(
        registries_module,
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


@pytest.mark.parametrize(
    ("resolved_model", "expected_model_params"),
    [
        ("native", {"pretrained": False}),
        (LazySlideFeatureExtractorAdapter("pathforge", _ExampleExtractor()), {}),
    ],
)
def test_lazyslide_processor_passes_resolved_model_to_feature_extraction(
    monkeypatch: pytest.MonkeyPatch,
    resolved_model: str | LazySlideFeatureExtractorAdapter,
    expected_model_params: dict[str, object],
) -> None:
    """Execution sends native strings and PathForge adapters to LazySlide correctly."""
    pytest.importorskip("lazyslide")
    import pathforge.core.slide_processing.lazyslide as lazyslide_module

    processor = lazyslide_module.LazySlideProcessor()
    wsi_object = _FakeWsiObject()
    wsi_object["pathforge_tiles"] = SimpleNamespace(X=np.ones((2, 3)))
    wsi = SimpleNamespace(obj=wsi_object)
    coords = np.array([[0, 0, 4, 4, 0], [4, 0, 4, 4, 0]], dtype=np.int32)
    tiling_spec = {"tile_px": 4, "tile_mpp": 0.5, "stride_px": 4}

    monkeypatch.setattr(processor, "_reconstruct_tile_spec", lambda coords, spec: {})
    monkeypatch.setattr(
        processor,
        "_reconstruct_tiles_table",
        lambda coords, tile_px: object(),
    )
    monkeypatch.setattr(
        processor,
        "_resolve_feature_extractor",
        lambda model_name, model_params: resolved_model,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        lazyslide_module.zs.tl,
        "feature_extraction",
        lambda *, wsi, model, **params: captured.update(model=model, params=params),
    )

    feature_matrix = processor.extract_features(
        wsi,
        coords,
        tiling_spec,
        config={
            "model": "pathforge",
            "params": {"batch_size": 2},
            "model_params": {"pretrained": False},
        },
    )

    assert captured["model"] is resolved_model
    assert captured["params"] == {"batch_size": 2, **expected_model_params}
    assert feature_matrix.shape == (2, 3)
