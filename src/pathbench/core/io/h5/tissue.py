# src/pathbench/core/io/h5/tissue.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, TypeAlias

import json

from pathbench.core.io.h5.base import (
    FileHandleH5,
    exists,
    read_json_dataset,
    write_json_dataset,
)
from pathbench.core.io.h5.layout import DEFAULT_LAYOUT, H5Layout

# -----------------------------------------------------------------------------
# Internal tissue contract
# -----------------------------------------------------------------------------
# Stored in H5 as JSON:
#
# [
#     [  # polygon 1
#         [[x, y], ...]  # outer ring
#     ],
#     [  # polygon 2
#         [[x, y], ...],  # outer ring
#         [[x, y], ...],  # hole 1
#     ],
# ]
#
# This is equivalent to the "coordinates" part of a GeoJSON MultiPolygon,
# but without the outer {"type": "MultiPolygon", "coordinates": ...} wrapper.
# -----------------------------------------------------------------------------

Position: TypeAlias = list[float]
Ring: TypeAlias = list[Position]
PolygonRings: TypeAlias = list[Ring]
TissuePolygons: TypeAlias = list[PolygonRings]


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _coerce_position(obj: Any, *, context: str) -> Position:
    if not isinstance(obj, (list, tuple)) or len(obj) < 2:
        raise ValueError(f"{context}: expected [x, y], got {obj!r}")

    x = float(obj[0])
    y = float(obj[1])
    return [x, y]


def _coerce_ring(obj: Any, *, context: str) -> Ring:
    if not isinstance(obj, (list, tuple)):
        raise TypeError(f"{context}: expected ring as list, got {type(obj)}")

    ring = [_coerce_position(p, context=f"{context} point {i}") for i, p in enumerate(obj)]

    # GeoJSON linear ring rules: at least 4 positions, first == last
    if len(ring) < 4:
        raise ValueError(f"{context}: linear ring must contain at least 4 positions, got {len(ring)}")

    if ring[0][0] != ring[-1][0] or ring[0][1] != ring[-1][1]:
        raise ValueError(f"{context}: linear ring must be closed (first position must equal last position)")

    return ring


def _ensure_tissue_polygons(obj: Any) -> TissuePolygons:
    """
    Normalize and validate tissue polygons to the internal contract:

        list[polygon]
        polygon = list[ring]
        ring = list[[x, y], ...]

    Notes:
    - No geometry is altered.
    - No holes are dropped.
    - Rings must already be closed.
    """
    if obj is None:
        return []

    if not isinstance(obj, (list, tuple)):
        raise TypeError(f"tissue must be a list of polygons, got {type(obj)}")

    polygons: TissuePolygons = []

    for poly_idx, poly in enumerate(obj):
        if not isinstance(poly, (list, tuple)):
            raise TypeError(
                f"tissue polygon at index {poly_idx} must be a list of rings, got {type(poly)}"
            )
        if len(poly) == 0:
            raise ValueError(f"tissue polygon at index {poly_idx} must contain at least one ring")

        rings: PolygonRings = []
        for ring_idx, ring in enumerate(poly):
            kind = "outer ring" if ring_idx == 0 else f"hole {ring_idx}"
            rings.append(_coerce_ring(ring, context=f"polygon {poly_idx} {kind}"))

        polygons.append(rings)

    return polygons


# -----------------------------------------------------------------------------
# H5 tissue storage
# -----------------------------------------------------------------------------

def tissue_exists(slide_artifact: FileHandleH5, *, layout: H5Layout = DEFAULT_LAYOUT) -> bool:
    return exists(slide_artifact.h5, layout.tissue_dataset)


def read_tissue(
    slide_artifact: FileHandleH5,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> TissuePolygons:
    """
    Read tissue polygons from H5.

    Returns:
        TissuePolygons:
            [
                [ring0, hole1, ...],   # polygon 1
                [ring0],               # polygon 2
            ]
    """
    obj = read_json_dataset(slide_artifact.h5, layout.tissue_dataset)
    return _ensure_tissue_polygons(obj)


def write_tissue(
    slide_artifact: FileHandleH5,
    tissue_polygons: Any,
    *,
    layout: H5Layout = DEFAULT_LAYOUT,
) -> None:
    """
    Write tissue polygons to H5 as JSON using the internal nested-rings contract.

    Accepts:
        [
            [ring0, hole1, ...],   # polygon 1
            [ring0],               # polygon 2
        ]
    """
    polygons = _ensure_tissue_polygons(tissue_polygons)
    write_json_dataset(slide_artifact.h5, layout.tissue_dataset, polygons)


# -----------------------------------------------------------------------------
# External tissue loading (by file suffix)
# -----------------------------------------------------------------------------

ExternalTissueLoader = Callable[[Path], TissuePolygons]
EXTERNAL_TISSUE_LOADERS: Dict[str, ExternalTissueLoader] = {}


def register_external_tissue_loader(*suffixes: str):
    """
    Decorator to register an external tissue loader for one or more suffixes.
    """
    def _decorator(func: ExternalTissueLoader) -> ExternalTissueLoader:
        for suf in suffixes:
            if not suf.startswith("."):
                raise ValueError(f"Suffix must start with '.', got: {suf!r}")
            EXTERNAL_TISSUE_LOADERS[suf.lower()] = func
        return func

    return _decorator


def load_external_tissue_polygons(path: str | Path) -> TissuePolygons:
    """
    Load tissue polygons from an external file (currently supports .geojson).

    Returns:
        TissuePolygons using the internal nested-rings contract.
    """
    p = Path(path)
    suf = p.suffix.lower()

    loader = EXTERNAL_TISSUE_LOADERS.get(suf)
    if loader is None:
        raise ValueError(
            f"Unsupported external tissue format: {suf!r}. "
            f"Supported: {sorted(EXTERNAL_TISSUE_LOADERS.keys())}"
        )

    if not p.exists():
        raise FileNotFoundError(f"External tissue file not found: {p}")

    polygons = loader(p)
    return _ensure_tissue_polygons(polygons)


# -----------------------------------------------------------------------------
# External tissue loading implementations
# -----------------------------------------------------------------------------

def _coerce_geojson_polygon_coords(coords: Any, *, context: str) -> PolygonRings:
    """
    Convert a GeoJSON Polygon coordinates array into our internal polygon format.

    GeoJSON Polygon coordinates:
        [outer_ring, hole1, hole2, ...]
    """
    if not isinstance(coords, (list, tuple)) or len(coords) == 0:
        raise ValueError(f"{context}: polygon coordinates must contain at least one ring")

    rings: PolygonRings = []
    for ring_idx, ring in enumerate(coords):
        kind = "outer ring" if ring_idx == 0 else f"hole {ring_idx}"
        rings.append(_coerce_ring(ring, context=f"{context} {kind}"))

    return rings


@register_external_tissue_loader(".geojson")
def _load_geojson_polygons(path: Path) -> TissuePolygons:
    """
    Load tissues from a GeoJSON FeatureCollection.

    Supports:
      - Polygon geometries (preserves outer ring + holes)
      - MultiPolygon geometries (each polygon preserved separately)

    Returns:
      TissuePolygons in the internal nested-rings contract.
    """
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if obj.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON must be a FeatureCollection")

    polygons: TissuePolygons = []

    for feat_idx, feat in enumerate(obj.get("features", [])):
        geom = (feat or {}).get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")

        if gtype == "Polygon":
            polygons.append(
                _coerce_geojson_polygon_coords(
                    coords,
                    context=f"feature {feat_idx} polygon",
                )
            )

        elif gtype == "MultiPolygon":
            if not isinstance(coords, (list, tuple)):
                raise ValueError(f"feature {feat_idx}: MultiPolygon coordinates must be a list")

            for poly_idx, poly_coords in enumerate(coords):
                polygons.append(
                    _coerce_geojson_polygon_coords(
                        poly_coords,
                        context=f"feature {feat_idx} multipolygon polygon {poly_idx}",
                    )
                )

        else:
            raise ValueError(f"Unsupported geometry type in tissue geojson: {gtype}")

    return polygons