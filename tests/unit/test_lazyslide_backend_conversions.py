# tests/unit/test_lazyslide_backend_conversions.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pandas as pd

from importlib import import_module
from pathbench.utils.registries import SLIDE_PROCESSORS


shapely_geometry = pytest.importorskip("shapely.geometry")
Polygon = shapely_geometry.Polygon


def _get_processor():
    import_module("pathbench.core.slide_processing.lazyslide")  # registers via decorator
    ProcessorClass = SLIDE_PROCESSORS.get("lazyslide")
    return ProcessorClass()

def test_tiling_spec_from_config() -> None:
    proc = _get_processor()

    spec = proc._tiling_spec_from_config({"tile_px": 256, "tile_mpp": 0.5})
    assert spec == {"tile_px": 256, "tile_mpp": 0.5}

    with pytest.raises(ValueError):
        _ = proc._tiling_spec_from_config({"tile_px": 256})  # missing tile_mpp


def test_tiles_table_to_coords_sorting_and_values() -> None:
    proc = _get_processor()

    # Two tiles, intentionally out-of-order tile_id so we test sorting
    geoms = [
        Polygon([(256, 0), (512, 0), (512, 256), (256, 0)]),
        Polygon([(0, 0), (256, 0), (256, 256), (0, 0)]),
    ]
    df = pd.DataFrame(
        {
            "tile_id": [2, 1],
            "geometry": geoms,
        }
    )

    tile_spec_obj = {"tiles": {"width": 256, "height": 256, "ops_level": 0}}
    coords = proc._tiles_table_to_coords(df, tile_spec_obj=tile_spec_obj)

    assert coords.shape == (2, 5)
    assert coords.dtype == np.int32

    # Sorted by tile_id numeric -> tile_id 1 first => x0=0,y0=0
    assert coords[0, 0] == 0
    assert coords[0, 1] == 0

    # Second is x0=256,y0=0
    assert coords[1, 0] == 256
    assert coords[1, 1] == 0

    # read_w, read_h, level
    assert np.all(coords[:, 2] == 256)
    assert np.all(coords[:, 3] == 256)
    assert np.all(coords[:, 4] == 0)


def test_tissues_table_to_policy() -> None:
    proc = _get_processor()

    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 0)])
    df = pd.DataFrame({"geometry": [poly]})

    tissues = proc._tissues_table_to_policy(df)

    assert isinstance(tissues, list)
    assert len(tissues) == 1
    arr = tissues[0]
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2 and arr.shape[1] == 2
    assert arr.dtype == np.float32


def test_validate_tile_spec() -> None:
    proc = _get_processor()

    assert proc.validate_tile_spec({"tile_px": 256, "tile_mpp": 0.5}, {"tile_px": 256, "tile_mpp": 0.5}) is True
    assert proc.validate_tile_spec({"tile_px": 256, "tile_mpp": 0.5}, {"tile_px": 512, "tile_mpp": 0.5}) is False
    assert proc.validate_tile_spec(None, {"tile_px": 256, "tile_mpp": 0.5}) is False
    assert proc.validate_tile_spec({"tile_px": 256}, {"tile_px": 256, "tile_mpp": 0.5}) is False
