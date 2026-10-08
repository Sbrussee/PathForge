"""Exercise the real normalization patch and conditional slide encoding together."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pathforge.core.experiments.combo_ids import build_feature_name, build_tiling_id
from pathforge.core.experiments.combinations import ComboConfig
from pathforge.core.feature_extractors.selection import FeatureExtractorSelection
from pathforge.core.io.slide_artifacts import features as features_io
from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_artifacts.layout import DEFAULT_LAYOUT
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


@pytest.fixture
def cached_patch_artifact(embedding_backend, tmp_path):
    """Store distinct patch vectors so reuse is distinguishable from encoding."""
    processor, wsi, coords, model, calls = embedding_backend
    wsi.artifact_path = tmp_path / "slide.h5"
    spec = {"tile_px": 4, "tile_mpp": 0.5, "stride_px": 4, "coord_space": "level0"}
    combo = ComboConfig(tile_px=4, tile_mpp=0.5, feature_extraction="example")
    patches = np.arange(9, dtype=np.float32).reshape(3, 3)
    with FileHandleH5(wsi.artifact_path, mode="a") as artifact:
        tiles_io.write_coords(artifact, build_tiling_id(combo), coords)
        tiles_io.write_tiling_spec(artifact, build_tiling_id(combo), spec)
        features_io.write_features(
            artifact, build_tiling_id(combo), build_feature_name(combo), patches
        )
    return processor, wsi, coords, model, calls, spec, combo, patches


@pytest.mark.parametrize("name", ["example-slide", "separate-slide", "example"])
@pytest.mark.parametrize("color_norm", [None, "macenko"])
@pytest.mark.parametrize("rows", [1, 3])
def test_slide_aggregation_reuses_saved_patch_features(
    cached_patch_artifact, monkeypatch, name, color_norm, rows
):
    processor, wsi, coords, model, calls, spec, combo, patches = cached_patch_artifact
    # Separate slide encoders must look up their vision encoder's storage name.
    combo.feature_extraction = "example_patch" if name == "separate-slide" else "example"
    combo.color_norm = color_norm
    coords, patches = coords[:rows], patches[:rows]
    with FileHandleH5(wsi.artifact_path, mode="a") as artifact:
        tiles_io.write_coords(artifact, build_tiling_id(combo), coords)
        features_io.write_features(
            artifact, build_tiling_id(combo), build_feature_name(combo), patches
        )
    request = FeatureExtractionRequest(
        FeatureExtractorSelection(name, "processor-native", "slide"),
        {"device": "cpu", "amp": False, "color_norm": color_norm},
    )
    from pathforge.core.slide_processing.lazyslide import processor as backend

    def unexpected_extraction(**kwargs):
        pytest.fail("Cached patch features must skip tile extraction")

    monkeypatch.setattr(backend.zs.tl, "feature_extraction", unexpected_extraction)
    output = processor.extract_features(wsi, coords, spec, request)
    np.testing.assert_array_equal(output, patches.mean(axis=0, keepdims=True))
    embeddings, positions, tile_size = calls["slide"][0]
    np.testing.assert_array_equal(embeddings[0], patches)
    np.testing.assert_array_equal(positions[0], coords[:, :2])
    assert calls["dataset"] == []
    assert calls["construct"] == 1
    assert output.dtype == np.float32
    assert tile_size == 32
    # The read-only cache lookup preserves the saved patch matrix.
    with FileHandleH5(wsi.artifact_path, mode="r") as artifact:
        np.testing.assert_array_equal(
            features_io.read_features(
                artifact, build_tiling_id(combo), build_feature_name(combo)
            ),
            patches,
        )


@pytest.mark.parametrize(
    "mismatch",
    [
        "missing",
        "unreadable",
        "normalization",
        "resolution",
        "stride",
        "order",
        "coordinates",
        "rows",
        "nonfinite",
        "empty_dimensions",
        "incomplete",
        "missing_coords",
        "missing_spec",
        "missing_resolution",
    ],
)
def test_incompatible_patch_cache_falls_back_to_encoding(cached_patch_artifact, mismatch):
    processor, wsi, coords, model, calls, spec, combo, patches = cached_patch_artifact
    color_norm = "macenko" if mismatch == "normalization" else None
    if mismatch == "missing":
        wsi.artifact_path = wsi.artifact_path.with_name("missing.h5")
    elif mismatch == "unreadable":
        wsi.artifact_path.write_bytes(b"not an H5 artifact")
    elif mismatch == "resolution":
        spec = {**spec, "tile_mpp": 1.0}
    elif mismatch == "stride":
        spec = {**spec, "stride_px": 8}
    elif mismatch == "order":
        coords = coords[::-1].copy()
    elif mismatch == "coordinates":
        coords = coords.copy()
        coords[0, 0] += 1
    elif mismatch == "missing_resolution":
        spec = {key: value for key, value in spec.items() if key != "tile_mpp"}
    else:
        with FileHandleH5(wsi.artifact_path, mode="a") as artifact:
            tiling_id = build_tiling_id(combo)
            feature_name = build_feature_name(combo)
            if mismatch == "rows":
                features_io.write_features(artifact, tiling_id, feature_name, patches[:1])
            elif mismatch == "nonfinite":
                patches[0, 0] = np.nan
                features_io.write_features(artifact, tiling_id, feature_name, patches)
            elif mismatch == "empty_dimensions":
                features_io.write_features(artifact, tiling_id, feature_name, patches[:, :0])
            elif mismatch == "incomplete":
                dataset = artifact.h5[
                    DEFAULT_LAYOUT.features_dataset(tiling_id, feature_name)
                ]
                dataset.attrs["status"] = "writing"
            elif mismatch == "missing_coords":
                del artifact.h5[DEFAULT_LAYOUT.coords_dataset(tiling_id)]
            elif mismatch == "missing_spec":
                del artifact.h5[DEFAULT_LAYOUT.tiling_spec_dataset(tiling_id)]
    request = FeatureExtractionRequest(
        FeatureExtractorSelection("example-slide", "processor-native", "slide"),
        {"device": "cpu", "amp": False, "color_norm": color_norm},
    )
    output = processor.extract_features(wsi, coords, spec, request)
    assert calls["dataset"] == [color_norm]
    assert len(calls["slide"]) == 1
    assert output.shape == (1, 3)


def test_patch_requests_continue_to_encode_tiles(cached_patch_artifact):
    processor, wsi, coords, model, calls, spec, combo, patches = cached_patch_artifact
    request = FeatureExtractionRequest(
        FeatureExtractorSelection("example", "processor-native", "patch"),
        {"device": "cpu", "amp": False},
    )
    output = processor.extract_features(wsi, coords, spec, request)
    assert calls["dataset"] == [None]
    assert calls["slide"] == []
    assert output.shape == patches.shape
    assert not np.array_equal(output, patches)
