# tests/unit/test_lazyslide_backend_visualization.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pathforge.core.datasets.wsi_dataset import WSI
from pathforge.core.slide_processing.lazyslide import LazySlideProcessor


class _FakeProperties:
    def __init__(self, shape):
        self.shape = shape


class _FakePropertiesNoShape:
    pass


class _FakeWSIObj(dict):
    """
    Minimal fake object that behaves like the parts of LazySlide/WSIData
    used by LazySlideProcessor.get_thumbnail():
    - mapping access for 'wsi_thumbnail'
    - .properties.shape
    """
    def __init__(self, thumbnail: np.ndarray | None, properties_obj: object):
        super().__init__()
        if thumbnail is not None:
            self["wsi_thumbnail"] = thumbnail
        self.properties = properties_obj


def _make_wsi(tmp_path: Path, fake_obj: object) -> WSI:
    wsi = WSI(
        slide="S1",
        patient="P1",
        category="C1",
        path=tmp_path / "slide.svs",
        artifact_path=tmp_path / "S1.h5",
    )
    wsi._obj = fake_obj
    return wsi


def test_lazyslide_get_level0_shape_parses_properties_shape_hw_order(tmp_path: Path) -> None:
    processor = LazySlideProcessor()

    fake_obj = _FakeWSIObj(
        thumbnail=np.zeros((64, 128, 3), dtype=np.uint8),
        properties_obj=_FakeProperties(shape=[15616, 16384]),  # (H, W)
    )

    h0, w0 = processor._get_level0_shape(fake_obj)
    assert h0 == 15616
    assert w0 == 16384


def test_lazyslide_get_thumbnail_returns_image_and_downscale_factors(tmp_path: Path) -> None:
    processor = LazySlideProcessor()

    thumb = np.zeros((100, 200, 3), dtype=np.uint8)  # H=100, W=200
    fake_obj = _FakeWSIObj(
        thumbnail=thumb,
        properties_obj=_FakeProperties(shape=[1000, 4000]),  # level-0 (H, W)
    )
    wsi = _make_wsi(tmp_path, fake_obj)

    thumbnail_out, downscale_x, downscale_y = processor.get_thumbnail(wsi, level=-1)

    assert isinstance(thumbnail_out, np.ndarray)
    assert thumbnail_out.dtype == np.uint8
    assert thumbnail_out.shape == (100, 200, 3)

    # downscale_x = W0 / Wthumb, downscale_y = H0 / Hthumb
    assert downscale_x == pytest.approx(4000 / 200)  # 20.0
    assert downscale_y == pytest.approx(1000 / 100)  # 10.0


def test_lazyslide_get_thumbnail_accepts_grayscale_thumbnail(tmp_path: Path) -> None:
    processor = LazySlideProcessor()

    thumb_gray = np.zeros((80, 160), dtype=np.uint8)  # HxW grayscale
    fake_obj = _FakeWSIObj(
        thumbnail=thumb_gray,
        properties_obj=_FakeProperties(shape=[800, 1600]),  # (H, W)
    )
    wsi = _make_wsi(tmp_path, fake_obj)

    thumbnail_out, downscale_x, downscale_y = processor.get_thumbnail(wsi, level=-1)

    # helper converts grayscale -> RGB uint8
    assert isinstance(thumbnail_out, np.ndarray)
    assert thumbnail_out.dtype == np.uint8
    assert thumbnail_out.shape == (80, 160, 3)

    assert downscale_x == pytest.approx(1600 / 160)  # 10.0
    assert downscale_y == pytest.approx(800 / 80)    # 10.0


def test_lazyslide_get_level0_shape_raises_when_properties_shape_missing(tmp_path: Path) -> None:
    processor = LazySlideProcessor()

    fake_obj = _FakeWSIObj(
        thumbnail=np.zeros((32, 32, 3), dtype=np.uint8),
        properties_obj=_FakePropertiesNoShape(),
    )

    with pytest.raises(RuntimeError) as excinfo:
        _ = processor._get_level0_shape(fake_obj)

    msg = str(excinfo.value)
    assert "level-0 shape" in msg
    assert "properties.shape" in msg


def test_lazyslide_get_thumbnail_raises_when_thumbnail_missing(tmp_path: Path) -> None:
    processor = LazySlideProcessor()

    fake_obj = _FakeWSIObj(
        thumbnail=None,  # no 'wsi_thumbnail' key
        properties_obj=_FakeProperties(shape=[1000, 1000]),
    )
    wsi = _make_wsi(tmp_path, fake_obj)

    with pytest.raises(RuntimeError) as excinfo:
        _ = processor.get_thumbnail(wsi, level=-1)

    assert "wsi_thumbnail" in str(excinfo.value)