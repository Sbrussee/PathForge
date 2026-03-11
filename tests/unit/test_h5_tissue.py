from __future__ import annotations

import json
from pathlib import Path

import pytest

from pathbench.core.io.h5.base import FileHandleH5
from pathbench.core.io.h5 import tissue as tissue_io


def test_h5_tissue_roundtrip(tmp_path: Path) -> None:
    h5_path = tmp_path / "S1.h5"

    polys_in = [
        [
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 0.0]],
        ],
        [
            [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 5.0]],
        ],
    ]

    with FileHandleH5(h5_path, mode="a") as f:
        tissue_io.write_tissue(f, polys_in)
        assert tissue_io.tissue_exists(f) is True

        polys_out = tissue_io.read_tissue(f)
        assert isinstance(polys_out, list)
        assert len(polys_out) == 2

        assert polys_out == polys_in


def test_load_external_tissue_polygons_geojson_polygon_and_multipolygon(tmp_path: Path) -> None:
    geojson_path = tmp_path / "S1.geojson"

    # One Polygon + one MultiPolygon (two polygons inside)
    obj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "poly"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0, 0], [10, 0], [10, 10], [0, 0]],
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"name": "mpoly"},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [
                            [[20, 20], [30, 20], [30, 30], [20, 20]],
                        ],
                        [
                            [[40, 40], [50, 40], [50, 50], [40, 40]],
                        ],
                    ],
                },
            },
        ],
    }

    geojson_path.write_text(json.dumps(obj), encoding="utf-8")

    polys = tissue_io.load_external_tissue_polygons(geojson_path)

    assert len(polys) == 3
    for poly in polys:
        assert isinstance(poly, list)
        assert len(poly) >= 1

        outer_ring = poly[0]
        assert isinstance(outer_ring, list)
        assert len(outer_ring) >= 4
        assert all(len(pt) == 2 for pt in outer_ring)


def test_load_external_tissue_polygons_unsupported_suffix_raises(tmp_path: Path) -> None:
    p = tmp_path / "S1.txt"
    p.write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        tissue_io.load_external_tissue_polygons(p)

    assert "Unsupported external tissue format" in str(excinfo.value)