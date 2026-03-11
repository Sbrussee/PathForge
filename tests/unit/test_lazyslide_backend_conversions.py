from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from pathbench.utils.registries import SLIDE_PROCESSORS


shapely_geometry = pytest.importorskip("shapely.geometry")
Polygon = shapely_geometry.Polygon


def _get_processor():
    import_module("pathbench.core.slide_processing.lazyslide")  # registers via decorator
    ProcessorClass = SLIDE_PROCESSORS.get("lazyslide")
    return ProcessorClass()


def test_backend_tile_spec_to_policy_tiling_spec() -> None:
    proc = _get_processor()

    tile_spec_obj = {
        "tiles": {
            "width": 256,
            "height": 256,
            "mpp": 0.5,
            "stride_width": 128,
            "ops_level": 0,
            "ops_downsample": 1.0,
        }
    }
    config = {"tile_px": 999, "tile_mpp": 9.9}  # should be ignored when backend values exist

    spec = proc._backend_tile_spec_to_policy_tiling_spec(config=config, tile_spec_obj=tile_spec_obj)

    assert spec == {
        "tile_px": 256,
        "tile_mpp": 0.5,
        "stride_px": 128,
        "coord_space": "level0",
        "backend": "lazyslide",
    }


def test_backend_tile_spec_to_policy_tiling_spec_missing_values_raises() -> None:
    proc = _get_processor()

    # Missing both backend and config tile_mpp -> should fail
    with pytest.raises(ValueError):
        _ = proc._backend_tile_spec_to_policy_tiling_spec(
            config={"tile_px": 256},
            tile_spec_obj={"tiles": {"width": 256}},
        )


def test_backend_tiles_to_policy_coords_sorting_and_values() -> None:
    proc = _get_processor()

    # Two tiles, intentionally out-of-order tile_id so we test sorting
    geoms = [
        Polygon([(256, 0), (512, 0), (512, 256), (256, 256)]),
        Polygon([(0, 0), (256, 0), (256, 256), (0, 256)]),
    ]
    df = pd.DataFrame(
        {
            "tile_id": [2, 1],
            "geometry": geoms,
        }
    )

    tile_spec_obj = {
        "tiles": {
            "width": 256,
            "height": 256,
            "mpp": 0.5,
            "ops_level": 0,
            "ops_downsample": 2.0,  # read window becomes 512 x 512
        }
    }

    coords = proc._backend_tiles_to_policy_coords(df, tile_spec_obj=tile_spec_obj)

    assert coords.shape == (2, 5)
    assert coords.dtype == np.int32

    # Sorted by tile_id numeric -> tile_id 1 first => x0=0,y0=0
    assert coords[0, 0] == 0
    assert coords[0, 1] == 0

    # Second is x0=256,y0=0
    assert coords[1, 0] == 256
    assert coords[1, 1] == 0

    # read_w, read_h are derived from tile_px * ops_downsample = 256 * 2 = 512
    assert np.all(coords[:, 2] == 512)
    assert np.all(coords[:, 3] == 512)

    # read level = ops_level
    assert np.all(coords[:, 4] == 0)


def test_backend_tissues_to_policy() -> None:
    proc = _get_processor()

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    df = pd.DataFrame({"geometry": [poly]})

    tissues = proc._backend_tissues_to_policy(df)

    assert isinstance(tissues, list)
    assert len(tissues) == 1

    polygon = tissues[0]
    assert isinstance(polygon, list)
    assert len(polygon) == 1

    outer_ring = polygon[0]
    assert isinstance(outer_ring, list)
    assert len(outer_ring) >= 4
    assert all(len(pt) == 2 for pt in outer_ring)


def test_validate_tile_spec() -> None:
    proc = _get_processor()

    valid_spec = {
        "tile_px": 256,
        "tile_mpp": 0.5,
        "stride_px": 256,
        "coord_space": "level0",
        "backend": "lazyslide",
    }

    assert proc.validate_tile_spec(valid_spec, {"tile_px": 256, "tile_mpp": 0.5}) is True
    assert proc.validate_tile_spec(valid_spec, {"tile_px": 512, "tile_mpp": 0.5}) is False
    assert proc.validate_tile_spec(None, {"tile_px": 256, "tile_mpp": 0.5}) is False
    assert proc.validate_tile_spec({"tile_px": 256}, {"tile_px": 256, "tile_mpp": 0.5}) is False
    assert proc.validate_tile_spec(
        {"tile_px": 256, "tile_mpp": 0.5, "stride_px": 256, "coord_space": "not_level0"},
        {"tile_px": 256, "tile_mpp": 0.5},
    ) is False

    # If no config is provided, only structural validation is checked
    assert proc.validate_tile_spec(valid_spec, None) is True