# src/pathbench/core/io/tissues.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
import numpy as np
import json

from pathbench.core.io.base import (
    ensure_base_path,
    ensure_parent_dir,
    ensure_suffix,
    detect_artifact_path
)

# ---- Registry (populated at bottom) ----
_TissuesLoader = Callable[[Path], list[np.ndarray]]
_TissuesSaver = Callable[[list[np.ndarray], Path], None]

TISSUES_CODECS: dict[str, tuple[_TissuesLoader, _TissuesSaver]] = {}

DEFAULT_SUFFIX = ".geojson"
SUPPORTED_SUFFIXES: tuple[str, ...]


def load_tissues(base: Path) -> Optional[list[np.ndarray]]:
    """Load tissues for a base path."""
    base = ensure_base_path(base)
    p = detect_artifact_path(base)
    if p is None:
        return None

    loader, _ = TISSUES_CODECS[p.suffix]
    return loader(p)


def save_tissues(
    tissues: list[np.ndarray],
    base: Path,
    *,
    suffix: str = DEFAULT_SUFFIX,
) -> Path:
    """Save tissues for a base path."""
    base = ensure_base_path(base)

    codec = TISSUES_CODECS.get(suffix)
    if codec is None:
        raise ValueError(f"Unsupported tissues suffix '{suffix}'. Supported: {list(SUPPORTED_SUFFIXES)}")

    out = base.with_suffix(suffix)
    ensure_parent_dir(out)

    _, saver = codec
    saver(tissues, out)
    return out

# ---- GeoJSON ----

def save_tissues_geojson(tissues: List[np.ndarray], path: str | Path) -> None:
    """
    Save tissues (list of Nx2 numpy arrays) as GeoJSON FeatureCollection.

    Each tissue polygon is written as a GeoJSON Polygon. Rings are closed if needed.

    Args:
        tissues: List of polygons, each shape (N, 2) in slide pixel coords.
        path: Output .geojson path.
    """
    path = Path(path)
    path = ensure_suffix(path, ".geojson")
    path.parent.mkdir(parents=True, exist_ok=True)

    features: List[Dict[str, Any]] = []
    for i, poly in enumerate(tissues):
        arr = np.asarray(poly)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            raise ValueError(f"Invalid tissue polygon at index {i}: expected (N,2) with N>=3, got {arr.shape}")

        # Ensure ring is closed
        if not np.allclose(arr[0], arr[-1]):
            arr = np.vstack([arr, arr[0]])

        coords = arr.tolist()  # [[x,y], ...]
        features.append(
            {
                "type": "Feature",
                "properties": {"tissue_idx": i},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )

    fc = {"type": "FeatureCollection", "features": features}

    with path.open("w", encoding="utf-8") as f:
        json.dump(fc, f)
        f.write("\n")


def load_tissues_geojson(path: str | Path) -> List[np.ndarray]:
    """
    Load tissues from GeoJSON FeatureCollection produced by save_tissues_geojson().

    Supports:
      - Polygon geometries (uses the first ring)
      - MultiPolygon geometries (flattens: each polygon becomes one entry)

    Args:
        path: .geojson path

    Returns:
        List of polygons as numpy arrays (N,2) float32.
    """
    path = Path(path)
    path = ensure_suffix(path, ".geojson")

    if not path.exists():
        raise FileNotFoundError(f"Tissues geojson file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if obj.get("type") != "FeatureCollection":
        raise ValueError("GeoJSON must be a FeatureCollection")

    tissues: List[np.ndarray] = []

    for feat in obj.get("features", []):
        geom = (feat or {}).get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")

        if gtype == "Polygon":
            # coords: [ring0, ring1, ...] -> take ring0
            if not coords or not coords[0]:
                continue
            ring0 = np.asarray(coords[0], dtype=np.float32)
            tissues.append(ring0)

        elif gtype == "MultiPolygon":
            # coords: [[[ring0,...]], [[ring0,...]], ...]
            if not coords:
                continue
            for poly in coords:
                if not poly or not poly[0]:
                    continue
                ring0 = np.asarray(poly[0], dtype=np.float32)
                tissues.append(ring0)
        else:
            raise ValueError(f"Unsupported geometry type in tissues geojson: {gtype}")

    return tissues

# ---- Register codecs ----
TISSUES_CODECS[".geojson"] = (load_tissues_geojson, save_tissues_geojson)
SUPPORTED_SUFFIXES = tuple(TISSUES_CODECS.keys())