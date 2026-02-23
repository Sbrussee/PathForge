# tests/unit/test_policy_resolve_tissue.py

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType

import numpy as np
import pytest

from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5 import tissue as tissue_io
from pathbench.policy.feature_extraction import FeatureExtractionPolicy


class DummyDataset:
    def __init__(self, tissue_annotations_dir: Path | None):
        self.tissue_annotations_dir = tissue_annotations_dir


class DummyWSI:
    def __init__(self, slide: str, path: Path):
        self.slide = slide
        self.path = path


class FakeSlideProcessor:
    def __init__(self) -> None:
        self.load_calls = 0
        self.close_calls = 0
        self.segment_calls = 0

    def load_wsi(self, wsi: DummyWSI) -> None:
        self.load_calls += 1

    def close_wsi(self, wsi: DummyWSI) -> None:
        self.close_calls += 1

    def segment_tissue(self, wsi: DummyWSI, config) -> list[np.ndarray]:
        self.segment_calls += 1
        return [np.array([[0, 0], [10, 0], [10, 10], [0, 0]], dtype=np.float32)]


def _make_policy_shim() -> FeatureExtractionPolicy:
    """
    Create a FeatureExtractionPolicy instance without running __init__
    (so we can unit-test helper methods in isolation).
    """
    policy = FeatureExtractionPolicy.__new__(FeatureExtractionPolicy)

    # Bind the real _find_external_roi_file method (so we test your filtering).
    policy._find_external_roi_file = MethodType(FeatureExtractionPolicy._find_external_roi_file, policy)

    return policy


def test_find_external_roi_file_filters_supported_suffixes(tmp_path: Path) -> None:
    roi_dir = tmp_path / "rois"
    roi_dir.mkdir()

    # Unsupported file should be ignored
    (roi_dir / "S1.txt").write_text("no", encoding="utf-8")

    # Supported geojson
    geo = roi_dir / "S1.geojson"
    geo.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    policy = _make_policy_shim()
    ds = DummyDataset(tissue_annotations_dir=roi_dir)

    found = policy._find_external_roi_file(dataset=ds, slide_id="S1")
    assert found == geo


def test_resolve_tissue_cache_first_does_not_touch_external_or_backend(tmp_path: Path, monkeypatch) -> None:
    h5_path = tmp_path / "S1.h5"
    roi_dir = tmp_path / "rois"
    roi_dir.mkdir()

    # External geojson exists but should NOT be used because cache exists
    (roi_dir / "S1.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}),
        encoding="utf-8",
    )

    cached = [np.array([[0, 0], [2, 0], [2, 2], [0, 0]], dtype=np.float32)]

    # If external loader gets called, fail
    def _boom(_p: Path):
        raise AssertionError("External ROI loader should not be called when cache exists.")

    monkeypatch.setattr(tissue_io, "load_external_tissue_polygons", _boom)

    policy = _make_policy_shim()
    ds = DummyDataset(tissue_annotations_dir=roi_dir)
    wsi = DummyWSI("S1", tmp_path / "S1.svs")
    backend = FakeSlideProcessor()

    with FileHandleH5(h5_path, mode="a") as f:
        tissue_io.write_tissue(f, cached)

        out = FeatureExtractionPolicy._resolve_tissue_polygons(
            policy,
            dataset=ds,
            wsi=wsi,
            slide_artifact=f,
            slide_processor=backend,
            segmentation_config={"method": "otsu", "params": {}},
        )

    assert backend.segment_calls == 0
    assert backend.load_calls == 0
    assert backend.close_calls == 0
    assert len(out) == 1
    np.testing.assert_allclose(out[0], cached[0])


def test_resolve_tissue_uses_external_when_no_cache_and_writes_to_h5(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    roi_dir = tmp_path / "rois"
    roi_dir.mkdir()

    geo = roi_dir / "S1.geojson"
    geo.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 0]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    policy = _make_policy_shim()
    ds = DummyDataset(tissue_annotations_dir=roi_dir)
    wsi = DummyWSI("S1", tmp_path / "S1.svs")
    backend = FakeSlideProcessor()

    with FileHandleH5(h5_path, mode="a") as f:
        out = FeatureExtractionPolicy._resolve_tissue_polygons(
            policy,
            dataset=ds,
            wsi=wsi,
            slide_artifact=f,
            slide_processor=backend,
            segmentation_config={"method": "otsu", "params": {}},
        )

        # Should be cached now
        assert tissue_io.tissue_exists(f) is True
        cached = tissue_io.read_tissue(f)

    assert backend.segment_calls == 0
    assert len(out) == 1
    assert len(cached) == 1
    np.testing.assert_allclose(out[0], cached[0])


def test_resolve_tissue_falls_back_to_backend_when_no_cache_and_no_external(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"
    roi_dir = tmp_path / "rois"
    roi_dir.mkdir()

    policy = _make_policy_shim()
    ds = DummyDataset(tissue_annotations_dir=roi_dir)
    wsi = DummyWSI("S1", tmp_path / "S1.svs")
    backend = FakeSlideProcessor()

    with FileHandleH5(h5_path, mode="a") as f:
        out = FeatureExtractionPolicy._resolve_tissue_polygons(
            policy,
            dataset=ds,
            wsi=wsi,
            slide_artifact=f,
            slide_processor=backend,
            segmentation_config={"method": "otsu", "params": {}},
        )

        assert tissue_io.tissue_exists(f) is True
        cached = tissue_io.read_tissue(f)

    assert backend.segment_calls == 1
    assert backend.load_calls == 1
    assert backend.close_calls == 1
    assert len(out) == 1
    np.testing.assert_allclose(out[0], cached[0])
