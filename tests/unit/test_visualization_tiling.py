# tests/unit/test_visualization_tiling.py

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from pathbench.core.visualization.tiles_overview import render_tiles_overview_image


def _open_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()  # force decode
    return img


def test_render_tiles_overview_returns_non_empty_jpeg_bytes_rgb_thumbnail() -> None:
    # Simple RGB thumbnail (HxWx3)
    thumbnail = np.full((120, 160, 3), 200, dtype=np.uint8)

    # coords: [x, y, read_w, read_h, read_level]
    coords = np.array(
        [
            [0, 0, 32, 32, 0],
            [64, 32, 32, 32, 0],
            [96, 64, 32, 32, 0],
        ],
        dtype=np.int32,
    )

    out = render_tiles_overview_image(
        thumbnail_image=thumbnail,
        coords_array=coords,
        downscale_x=4.0,
        downscale_y=4.0,
        slide_id="S1",
        tiling_spec={"tile_px": 32, "tile_mpp": 0.5},
    )

    assert isinstance(out, bytes)
    assert len(out) > 0

    # JPEG SOI marker
    assert out[:2] == b"\xff\xd8"

    img = _open_image_from_bytes(out)
    assert img.size == (160, 120)  # PIL reports (W, H)
    assert img.mode == "RGB"


def test_render_tiles_overview_accepts_grayscale_numpy_thumbnail() -> None:
    # Grayscale thumbnail (HxW)
    thumbnail = np.full((80, 100), 180, dtype=np.uint8)

    coords = np.array(
        [
            [0, 0, 20, 20, 0],
            [40, 20, 20, 20, 0],
        ],
        dtype=np.int32,
    )

    out = render_tiles_overview_image(
        thumbnail_image=thumbnail,
        coords_array=coords,
        downscale_x=2.0,
        downscale_y=2.0,
        slide_id="gray_slide",
        tiling_spec={"tile_px": 20, "tile_mpp": 1.0},
    )

    assert isinstance(out, bytes)
    assert len(out) > 0

    img = _open_image_from_bytes(out)
    assert img.size == (100, 80)
    assert img.mode == "RGB"  # renderer converts to grayscale background then RGB for overlay/text


def test_render_tiles_overview_accepts_pil_image_input() -> None:
    thumbnail_pil = Image.new("RGB", (64, 48), color=(220, 220, 220))
    coords = np.array([[0, 0, 16, 16, 0]], dtype=np.int32)

    out = render_tiles_overview_image(
        thumbnail_image=thumbnail_pil,
        coords_array=coords,
        downscale_x=2.0,
        downscale_y=2.0,
        slide_id="pil_input",
        tiling_spec={"tile_px": 16, "tile_mpp": 0.5},
    )

    assert isinstance(out, bytes)
    assert len(out) > 0
    img = _open_image_from_bytes(out)
    assert img.size == (64, 48)


def test_render_tiles_overview_works_with_zero_tiles() -> None:
    thumbnail = np.full((90, 120, 3), 150, dtype=np.uint8)
    coords = np.empty((0, 5), dtype=np.int32)

    out = render_tiles_overview_image(
        thumbnail_image=thumbnail,
        coords_array=coords,
        downscale_x=1.0,
        downscale_y=1.0,
        slide_id="empty_tiles",
        tiling_spec={"tile_px": 256, "tile_mpp": 0.5},
    )

    assert isinstance(out, bytes)
    assert len(out) > 0
    img = _open_image_from_bytes(out)
    assert img.size == (120, 90)


def test_render_tiles_overview_rejects_bad_coords_shape() -> None:
    thumbnail = np.full((50, 60, 3), 200, dtype=np.uint8)

    bad_coords_1d = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(ValueError):
        render_tiles_overview_image(
            thumbnail_image=thumbnail,
            coords_array=bad_coords_1d,
            downscale_x=1.0,
            downscale_y=1.0,
            slide_id="bad1",
        )

    bad_coords_wrong_cols = np.array([[0, 0, 16, 16]], dtype=np.int32)  # only 4 columns
    bad_coords_not_2d = np.zeros((2, 2, 2), dtype=np.int32)
    with pytest.raises(ValueError):
        render_tiles_overview_image(
            thumbnail_image=thumbnail,
            coords_array=bad_coords_not_2d,
            downscale_x=1.0,
            downscale_y=1.0,
            slide_id="bad2",
        )


def test_render_tiles_overview_rejects_non_positive_downscale() -> None:
    thumbnail = np.full((50, 60, 3), 200, dtype=np.uint8)
    coords = np.array([[0, 0, 16, 16, 0]], dtype=np.int32)

    with pytest.raises(ValueError):
        render_tiles_overview_image(
            thumbnail_image=thumbnail,
            coords_array=coords,
            downscale_x=0.0,
            downscale_y=1.0,
            slide_id="bad_downscale_x",
        )

    with pytest.raises(ValueError):
        render_tiles_overview_image(
            thumbnail_image=thumbnail,
            coords_array=coords,
            downscale_x=1.0,
            downscale_y=-1.0,
            slide_id="bad_downscale_y",
        )