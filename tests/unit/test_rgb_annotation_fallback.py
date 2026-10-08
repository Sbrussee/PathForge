from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pathforge.core.io.slide_artifacts import tiles as tiles_io
from pathforge.core.io.slide_artifacts.base import FileHandleH5
from pathforge.core.io.slide_retrieval import descriptors as descriptors_io
from pathforge.core.slide_processing.lazyslide import LazySlideProcessor
from pathforge.slide_retrieval.representation_strategies import histogram_rgb, mean_rgb


@pytest.mark.parametrize("module,resolve,dimension", [
    (mean_rgb, mean_rgb.resolve_sample_patch_mean_rgb, 3),
    (histogram_rgb, histogram_rgb.resolve_sample_patch_histogram_rgb, 768),
])
@pytest.mark.parametrize("native_mpp,expected_mpp", [(None, 0.25), (0.5, 0.5)])
def test_rgb_uses_annotation_fallback_with_real_processor(tmp_path, monkeypatch, module, resolve, dimension, native_mpp, expected_mpp):
    bag_id = "256px_0.5mpp"
    artifact_path = tmp_path / "S1.h5"
    staged = tmp_path / "renamed.svs"
    staged.write_bytes(b"fake")
    with FileHandleH5(artifact_path, mode="a") as artifact:
        tiles_io.write_coords(artifact, bag_id, np.array([[0, 0, 2, 2, 0]], dtype=np.int32))
    annotation_file = tmp_path / "annotations.csv"
    pd.DataFrame([
        {"dataset": "wrong", "slide": "S1", "fallback_mpp": 8},
        {"dataset": "a", "slide": "S1", "fallback_mpp": "0.25", "patient": "P1", "category": "tumor"},
    ]).to_csv(annotation_file, index=False)
    cfg = SimpleNamespace(experiment=SimpleNamespace(annotation_file=annotation_file))
    sample = SimpleNamespace(slide_ids=["S1"], artifact_paths=[artifact_path], slide_paths=[staged], metadata={"dataset": "a"})
    closed = []
    opened = SimpleNamespace(properties=SimpleNamespace(mpp=native_mpp), close=lambda: closed.append(True))

    def read_region(*args, **kwargs):
        assert opened.properties.mpp == expected_mpp
        return np.full((2, 2, 3), 128, dtype=np.uint8)

    opened.read_region = read_region
    reads = []
    monkeypatch.setattr("pathforge.core.slide_processing.lazyslide.processor.open_wsi", lambda path, **kwargs: reads.append(path) or opened)
    processor = LazySlideProcessor()
    monkeypatch.setattr(module, "_build_slide_processor", lambda **kwargs: processor)
    result = resolve(sample=sample, bag_id=bag_id, config=cfg)
    assert result.shape == (1, dimension)
    assert reads == [staged.resolve()]
    assert closed == [True]
    if dimension == 3:
        np.testing.assert_allclose(result, 128 / 255)
    else:
        assert result[0, 128] == 256 * 256
        assert result.sum() == 3 * 256 * 256
    target = mean_rgb._slide_retrieval_artifact_path(slide_artifact_path=artifact_path, slide_id="S1")
    with FileHandleH5(target, mode="r") as artifact:
        np.testing.assert_array_equal(descriptors_io.read_descriptor(artifact, bag_id, module.MEAN_RGB_DESCRIPTOR_NAME if dimension == 3 else module.HISTOGRAM_RGB_DESCRIPTOR_NAME), result)
    annotation_file.unlink()
    np.testing.assert_array_equal(resolve(sample=sample, bag_id=bag_id, config=cfg), result)
    assert len(reads) == 1


@pytest.mark.parametrize("fallback", [None, "bad", 0, -1, np.inf])
def test_missing_native_mpp_and_invalid_fallback_fail_at_processor(monkeypatch, fallback):
    from pathforge.core.datasets.wsi_dataset import WSI

    closed = []
    opened = SimpleNamespace(
        properties=SimpleNamespace(mpp=None), close=lambda: closed.append(True)
    )
    monkeypatch.setattr(
        "pathforge.core.slide_processing.lazyslide.processor.open_wsi",
        lambda *args, **kwargs: opened,
    )
    wsi = WSI.from_annotation(
        {"slide": "S1", "fallback_mpp": fallback},
        slide_path="staged.svs", artifact_path="S1.h5",
    )
    with pytest.raises(RuntimeError, match="valid scalar base MPP"):
        LazySlideProcessor().load_wsi(wsi)
    assert wsi._obj is None
    assert closed == [True]
