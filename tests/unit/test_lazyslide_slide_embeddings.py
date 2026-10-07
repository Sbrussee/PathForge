"""Exercise the real normalization patch and conditional slide encoding together."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pathforge.core.feature_extractors.selection import FeatureExtractorSelection
from pathforge.core.slide_processing.base import FeatureExtractionRequest


@pytest.fixture
def embedding_backend(monkeypatch):
    from lazyslide_models import MODEL_REGISTRY

    from pathforge.core.slide_processing.lazyslide import feature_extraction_patch as patch
    from pathforge.core.slide_processing.lazyslide import processor as backend

    calls = {"dataset": [], "slide": [], "construct": 0}

    class PatchEncoder:
        task = "vision"
        name = "example_patch"

        def __init__(self, model_path=None, token=None):
            pass

        def to(self, device):
            return self

        def get_transform(self):
            def transform(tile):
                assert tile.dtype == np.uint8
                assert tile.shape == (4, 5, 3)
                return torch.from_numpy(tile).permute(2, 0, 1).float()
            return transform

        def encode_image(self, images):
            return images.mean(dim=(2, 3))

    class SlideEncoder(PatchEncoder):
        task = ["multimodal", "slide_encoder"]
        vision_encoder = "example"
        result = None

        def __init__(self, model_path=None, token=None):
            calls["construct"] += 1

        def encode_slide(self, embeddings, coords=None, base_tile_size=None):
            calls["slide"].append((embeddings.cpu().numpy(), coords.cpu().numpy(), base_tile_size))
            return {"embeddings": embeddings.mean(dim=1) if self.result is None else self.result}

    monkeypatch.setitem(MODEL_REGISTRY, "example", SlideEncoder)
    monkeypatch.setitem(MODEL_REGISTRY, "example_alias", SlideEncoder)
    monkeypatch.setitem(MODEL_REGISTRY, "example_patch", PatchEncoder)
    monkeypatch.setitem(MODEL_REGISTRY, "separate", type(
        "Separate", (SlideEncoder,), {"vision_encoder": "example_patch"},
    ))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    processor = backend.LazySlideProcessor()

    def tile_images(tile_key, transform, color_norm):
        calls["dataset"].append(color_norm)
        images = []
        for index in range(3):
            tile = np.full((4, 5, 3), index + 10, dtype=np.uint8)
            if color_norm:
                # Simulate WSIData's normalized CHW float output, retaining the
                # actual PathForge restoration/transform/extraction implementation.
                tile = torch.from_numpy(tile).permute(2, 0, 1).float() + 10
            images.append({"image": transform(tile)})
        return images

    class SlideData(dict):
        attrs = {}
        shapes = {"tiles": np.zeros(3)}
        properties = SimpleNamespace(level_downsample=[1.0, 4.0])
        ds = SimpleNamespace(tile_images=tile_images)

        def tile_spec(self, key):
            return SimpleNamespace(base_width=32)

    obj = SlideData()
    monkeypatch.setattr(patch, "add_features", lambda wsi, key, tile_key, features: wsi.__setitem__(key, SimpleNamespace(X=features)))
    monkeypatch.setattr(processor, "_reconstruct_tile_spec", lambda coords, spec: {})
    monkeypatch.setattr(processor, "_reconstruct_tiles_table", lambda coords, tile_px: coords)
    coords = np.array([[0, 0, 32, 32, 0], [32, 0, 32, 32, 0], [0, 32, 32, 32, 0]], dtype=np.int32)
    return processor, SimpleNamespace(obj=obj), coords, SlideEncoder, calls


@pytest.mark.parametrize("name", ["example", "example-slide", "example_alias-slide", "separate-slide"])
@pytest.mark.parametrize("color_norm", [None, "macenko"])
def test_requested_embedding_level_preserves_normalized_patch_flow(embedding_backend, name, color_norm):
    processor, wsi, coords, model, calls = embedding_backend
    level = "slide" if name.endswith("-slide") else "patch"
    request = FeatureExtractionRequest(
        FeatureExtractorSelection(name, "processor-native", level),
        {"device": "cpu", "amp": False, "batch_size": 2, "color_norm": color_norm},
    )
    output = processor.extract_features(wsi, coords, {"tile_px": 4}, request)
    expected_patches = np.repeat(np.arange(3, dtype=np.float32)[:, None] + (20 if color_norm else 10), 3, axis=1)
    assert calls["dataset"] == [color_norm]
    assert output.dtype == np.float32
    if level == "slide":
        assert output.shape == (1, 3)
        np.testing.assert_array_equal(output, expected_patches.mean(axis=0, keepdims=True))
        embeddings, positions, tile_size = calls["slide"][0]
        np.testing.assert_array_equal(embeddings[0], expected_patches)
        np.testing.assert_array_equal(positions[0], coords[:, :2])
        assert tile_size == 32
        assert calls["construct"] == 1
    else:
        np.testing.assert_array_equal(output, expected_patches)
        assert calls["slide"] == []
    assert f"{name}_tiles" in wsi.obj


@pytest.mark.parametrize("result", [np.zeros((2, 3)), np.zeros((1, 2, 3)), np.zeros((1, 0)), np.full((1, 3), np.nan)])
def test_slide_result_must_be_one_finite_vector(embedding_backend, result):
    processor, wsi, coords, model, calls = embedding_backend
    model.result = result
    request = FeatureExtractionRequest(
        FeatureExtractorSelection("example-slide", "processor-native", "slide"),
        {"device": "cpu", "amp": False},
    )
    with pytest.raises(ValueError, match="slide embedding|non-finite"):
        processor.extract_features(wsi, coords, {"tile_px": 4}, request)


def test_one_dimensional_slide_output_is_stored_as_one_row(embedding_backend):
    processor, wsi, coords, model, calls = embedding_backend
    model.result = torch.ones(3)
    request = FeatureExtractionRequest(
        FeatureExtractorSelection("example-slide", "processor-native", "slide"),
        {"device": "cpu", "amp": False},
    )
    assert processor.extract_features(wsi, coords, {"tile_px": 4}, request).shape == (1, 3)


def test_slide_encoder_receives_tile_extent_in_level_zero_coordinates(embedding_backend):
    """The slide encoder gets pyramid-scaled read size, not resized patch pixels."""
    processor, wsi, coords, model, calls = embedding_backend
    coords[:, 4] = 1
    request = FeatureExtractionRequest(
        FeatureExtractorSelection("example-slide", "processor-native", "slide"),
        {"device": "cpu", "amp": False},
    )
    processor.extract_features(wsi, coords, {"tile_px": 4}, request)
    assert calls["slide"][0][2] == 128


@pytest.mark.parametrize("coords", [np.empty((0, 5)), np.empty((2, 4))])
def test_invalid_tiles_are_rejected_before_loading_models(embedding_backend, coords):
    processor, wsi, _, model, calls = embedding_backend
    request = FeatureExtractionRequest(
        FeatureExtractorSelection("example-slide", "processor-native", "slide"), {},
    )
    with pytest.raises(ValueError, match="empty slide|coords must"):
        processor.extract_features(wsi, coords, {"tile_px": 4}, request)
    assert calls["construct"] == 0


def test_slide_encoding_obeys_explicit_level_without_suffix(embedding_backend):
    """The backend takes its output level from the request, never the model name."""
    processor, wsi, coords, model, calls = embedding_backend
    request = FeatureExtractionRequest(
        FeatureExtractorSelection("example", "processor-native", "slide"),
        {"device": "cpu", "amp": False, "batch_size": 2},
    )
    output = processor.extract_features(wsi, coords, {"tile_px": 4}, request)
    assert output.shape == (1, 3)
    assert len(calls["slide"]) == 1
