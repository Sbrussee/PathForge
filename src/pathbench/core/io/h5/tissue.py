# src/pathbench/core/io/h5/tissue.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import json
import numpy as np

from pathbench.core.io.h5.base import FileHandleH5, exists, read_json_dataset, write_json_dataset
from pathbench.core.io.h5.layout import DEFAULT_LAYOUT, H5Layout

# -----------------------------------------------------------------------------
# tissue polygon helpers
# -----------------------------------------------------------------------------

def _ensure_polygons_list(obj: Any) -> list[np.ndarray]:
    """
    Normalize tissue polygons to: list[np.ndarray] where each array is (N,2) float32.

    Accepts either:
      - list of numpy arrays
      - list of python lists [[x,y], ...]
    """
    if obj is None:
        return []

    if not isinstance(obj, (list, tuple)):
        raise TypeError(f"tissue must be a list of polygons, got {type(obj)}")

    polygons: list[np.ndarray] = []
    for i, poly in enumerate(obj):
        arr = np.asarray(poly, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            raise ValueError(
                f"Invalid tissue polygon at index {i}: expected (N,2) with N>=3, got {arr.shape}"
            )
        polygons.append(arr)

    return polygons


def _polygons_to_jsonable(polygons: list[np.ndarray]) -> list[list[list[float]]]:
    """
    Convert list[np.ndarray (N,2)] -> JSON-serializable nested lists.
    """
    out: list[list[list[float]]] = []
    for i, poly in enumerate(polygons):
        arr = np.asarray(poly, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            raise ValueError(
                f"Invalid tissue polygon at index {i}: expected (N,2) with N>=3, got {arr.shape}"
            )
        out.append(arr.tolist())
    return out
    
# -----------------------------------------------------------------------------
# H5 tissue storage
# -----------------------------------------------------------------------------

def tissue_exists(slide_artifact: FileHandleH5, *, layout: H5Layout = DEFAULT_LAYOUT) -> bool:
    return exists(slide_artifact.h5, layout.tissue_dataset)


def read_tissue(slide_artifact: FileHandleH5, *, layout: H5Layout = DEFAULT_LAYOUT) -> list[np.ndarray]:
    """
    Read tissue polygons from H5 and return as list[np.ndarray (N,2) float32].
    """
    obj = read_json_dataset(slide_artifact.h5, layout.tissue_dataset)
    return _ensure_polygons_list(obj)


def write_tissue(slide_artifact: FileHandleH5, tissue_polygons: Any, *, layout: H5Layout = DEFAULT_LAYOUT) -> None:
    """
    Write tissue polygons to H5 as JSON.

    Accepts list[np.ndarray] or list[list[list[float]]].
    Stores JSON-serializable nested lists; read_tissue() restores to np.ndarray.
    """
    polygons = _ensure_polygons_list(tissue_polygons)
    write_json_dataset(slide_artifact.h5, layout.tissue_dataset, _polygons_to_jsonable(polygons))


# -----------------------------------------------------------------------------
# External tissue loading (by file suffix)
# -----------------------------------------------------------------------------

ExternalTissueLoader = Callable[[Path], list[np.ndarray]]
EXTERNAL_TISSUE_LOADERS: Dict[str, ExternalTissueLoader] = {}


def register_external_tissue_loader(*suffixes: str):
    """
    Decorator to register an external tissue loader for one or more suffixes.
    Example:
        @register_external_tissue_loader(".geojson")
        def _load_geojson_polygons(path: Path) -> list[np.ndarray]: ...
    """
    def _decorator(func: ExternalTissueLoader) -> ExternalTissueLoader:
        for suf in suffixes:
            if not suf.startswith("."):
                raise ValueError(f"Suffix must start with '.', got: {suf!r}")
            EXTERNAL_TISSUE_LOADERS[suf.lower()] = func
        return func

    return _decorator


def load_external_tissue_polygons(path: str | Path) -> list[np.ndarray]:
    """
    Load tissue polygons from an external file (currently supports .geojson).

    Returns:
        list[np.ndarray] where each polygon is (N,2) float32.
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
    return _ensure_polygons_list(polygons)


# -----------------------------------------------------------------------------
# External tissue loading implementations
# -----------------------------------------------------------------------------

@register_external_tissue_loader(".geojson")
def _load_geojson_polygons(path: Path) -> list[np.ndarray]:
    """
    Load tissues from a GeoJSON FeatureCollection.

    Supports:
      - Polygon geometries (uses the first ring)
      - MultiPolygon geometries (each polygon becomes one entry)

    Returns:
      list[np.ndarray] (N,2) float32
    """
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if obj.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON must be a FeatureCollection")

    polygons: list[np.ndarray] = []

    for feat in obj.get("features", []):
        geom = (feat or {}).get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")

        if gtype == "Polygon":
            # coords: [ring0, ring1, ...] -> take ring0
            if not coords or not coords[0]:
                continue
            ring0 = np.asarray(coords[0], dtype=np.float32)
            polygons.append(ring0[:, :2])

        elif gtype == "MultiPolygon":
            # coords: [[[ring0,...]], [[ring0,...]], ...]
            if not coords:
                continue
            for poly in coords:
                if not poly or not poly[0]:
                    continue
                ring0 = np.asarray(poly[0], dtype=np.float32)
                polygons.append(ring0[:, :2])

        else:
            raise ValueError(f"Unsupported geometry type in tissue geojson: {gtype}")

    return polygons
