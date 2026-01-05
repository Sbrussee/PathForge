# pathbench/utils/io_utils.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import torch

def _ensure_suffix(path: str | Path, suffix: str) -> Path:
    p = Path(path)
    return p if p.suffix else p.with_suffix(suffix)

################################## SAVING/LOADING TISSUE GEOJSON ##################################

def save_tissues_geojson(tissues: List[np.ndarray], path: str | Path) -> None:
    """
    Save tissues (list of Nx2 numpy arrays) as GeoJSON FeatureCollection.

    Each tissue polygon is written as a GeoJSON Polygon. Rings are closed if needed.

    Args:
        tissues: List of polygons, each shape (N, 2) in slide pixel coords.
        path: Output .geojson path.
    """
    path = Path(path)
    path = _ensure_suffix(path, ".geojson")
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
    path = _ensure_suffix(path, ".geojson")

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

################################ SAVING/LOADING TILES NPZ ###################################

def save_tiles_npz(tiles_df: pd.DataFrame, tile_spec: str, path: Path) -> None:
    """
    Save tiles + tile_spec (JSON string) to a compressed NPZ.

    Policy contract:
      - tile_spec must be a Python str containing valid JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    required = {"tile_id", "x", "y"}
    missing = required - set(tiles_df.columns)
    if missing:
        raise ValueError(f"tiles_df missing required columns: {sorted(missing)}")

    if not isinstance(tile_spec, str):
        raise TypeError(f"tile_spec must be a JSON string (str), got {type(tile_spec)}")

    # Validate tile_spec is JSON (keys can be variable)
    try:
        json.loads(tile_spec)
    except Exception as e:
        raise ValueError(f"tile_spec is not valid JSON: {e}") from e

    tile_id_series = tiles_df["tile_id"]

    # Prefer numeric tile_id if possible; otherwise store as strings
    try:
        tile_id_arr = tile_id_series.astype(np.int64).to_numpy()
        tile_id_kind = "int"
    except Exception:
        tile_id_arr = tile_id_series.astype(str).to_numpy()
        tile_id_kind = "str"

    x_arr = tiles_df["x"].to_numpy(dtype=np.int64)
    y_arr = tiles_df["y"].to_numpy(dtype=np.int64)

    np.savez_compressed(
        path,
        tile_id=tile_id_arr,
        tile_id_kind=np.array(tile_id_kind),
        x=x_arr,
        y=y_arr,
        tile_spec=np.array(tile_spec, dtype=np.str_),
    )


def load_tiles_npz(path: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load tiles + tile_spec from a compressed NPZ created by save_tiles_npz().

    Returns:
      (tiles_df, tile_spec) or (None, None) if file not found.

    Policy contract guaranteed:
      - tile_spec is a Python str
      - tile_spec contains valid JSON
    """
    if not path.exists():
        return None, None

    z = np.load(path, allow_pickle=False)

    tile_id_kind = str(z["tile_id_kind"])
    tile_id_arr = z["tile_id"]
    if tile_id_kind == "int":
        tile_id = tile_id_arr.astype(np.int64)
    else:
        tile_id = tile_id_arr.astype(str)

    tiles_df = pd.DataFrame(
        {
            "tile_id": tile_id,
            "x": z["x"].astype(np.int64),
            "y": z["y"].astype(np.int64),
        }
    )

    # STRICT: extract the stored scalar value, don't use str(np_array_repr)
    tile_spec_raw = z["tile_spec"]
    if isinstance(tile_spec_raw, np.ndarray):
        tile_spec_raw = tile_spec_raw.item()

    if not isinstance(tile_spec_raw, str):
        raise TypeError(f"Loaded tile_spec must be str, got {type(tile_spec_raw)}")

    # Strict JSON validation
    try:
        json.loads(tile_spec_raw)
    except Exception as e:
        raise ValueError(f"Loaded tile_spec is not valid JSON: {e}") from e

    return tiles_df, tile_spec_raw


################################ SAVING/LOADING FEATURES ANNADATA ##################################

def load_h5ad(path: str | Path) -> ad.AnnData:
    """
    Load an AnnData (.h5ad) file.

    Args:
        path: Path to the .h5ad file.

    Returns:
        AnnData object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AnnData .h5ad file not found: {path}")
    
    path = _ensure_suffix(path, ".h5ad")
    return ad.read_h5ad(path)


def save_h5ad_atomic(adata: ad.AnnData, path: str | Path) -> None:
    """
    Atomically save an AnnData (.h5ad) file by writing to a temporary file
    in the same directory and then replacing the target path.

    Args:
        adata: AnnData object to save.
        path: Destination path ('.h5ad').
    """
    path = Path(path)
    path = _ensure_suffix(path, ".h5ad")
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        adata.write_h5ad(tmp_path)
        tmp_path.replace(path)  # atomic on POSIX; replaces if exists
    finally:
        # If something failed before replace, try to clean up tmp file
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

def save_features_pt(adata: ad.AnnData, path: str | Path) -> None:
    """
    Save feature AnnData as:
      - <path>.pt: tensor only (N, D)
      - <path>.index.npz: obs fields needed to map rows (tile_id, optionally library_id)

    Required:
      - adata.obs contains 'tile_id'
    Optional:
      - adata.obs contains 'library_id' (will be stored if present)
    """
    path = Path(path)
    pt_path = _ensure_suffix(path, ".pt")
    idx_path = pt_path.with_suffix(".index.npz")

    # Ensure directory of actual output path exists
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    if adata.X is None:
        raise ValueError("AnnData.X is None; expected feature matrix.")
    if "tile_id" not in adata.obs.columns:
        raise ValueError("AnnData must contain obs['tile_id'].")

    # ---- X -> numpy float32 ----
    X = adata.X
    # support sparse matrices
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"AnnData.X must be 2D (N, D). Got shape={X.shape}")

    X = X.astype(np.float32, copy=False)

    # ---- obs fields ----
    tile_id = adata.obs["tile_id"].astype(str).to_numpy()
    if tile_id.shape[0] != X.shape[0]:
        raise ValueError(f"obs['tile_id'] length ({tile_id.shape[0]}) != X rows ({X.shape[0]})")

    have_library = "library_id" in adata.obs.columns
    if have_library:
        library_id = adata.obs["library_id"].astype(str).to_numpy()
        if library_id.shape[0] != X.shape[0]:
            raise ValueError(
                f"obs['library_id'] length ({library_id.shape[0]}) != X rows ({X.shape[0]})"
            )

    pt_tmp = pt_path.with_name(f".{pt_path.name}.tmp")

    # Ensure tmp ends with .npz so numpy doesn't append another suffix
    idx_tmp = idx_path.with_name(f".{idx_path.name}.tmp.npz")

    try:
        # tensor only
        torch.save(torch.from_numpy(X), pt_tmp)
        pt_tmp.replace(pt_path)

        # index mapping (tile_id is required; library_id optional)
        if have_library:
            np.savez_compressed(idx_tmp, tile_id=tile_id, library_id=library_id)
        else:
            np.savez_compressed(idx_tmp, tile_id=tile_id)

        idx_tmp.replace(idx_path)

    finally:
        for tmp in (pt_tmp, idx_tmp):
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass


def load_features_pt(path: str | Path) -> ad.AnnData:
    """
    Load features from:
      - <path>.pt
      - <path>.index.npz

    Returns AnnData with:
      - X: float32 (N, D)
      - obs['tile_id']: str
      - obs['library_id']: str (if present in index)
    """
    path = Path(path)
    pt_path = _ensure_suffix(path, ".pt")
    idx_path = pt_path.with_suffix(".index.npz")

    if not pt_path.exists():
        raise FileNotFoundError(f"Features file not found: {pt_path}")
    if not idx_path.exists():
        raise FileNotFoundError(f"Index file not found: {idx_path}")

    tensor = torch.load(pt_path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor in {pt_path}, got {type(tensor)}")

    X = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    if X.ndim != 2:
        raise ValueError(f"Loaded features must be 2D (N, D). Got shape={X.shape}")

    z = np.load(idx_path, allow_pickle=False)
    tile_id = z["tile_id"].astype(str)

    if tile_id.shape[0] != X.shape[0]:
        raise ValueError(f"tile_id length ({tile_id.shape[0]}) != X rows ({X.shape[0]})")

    obs = {"tile_id": tile_id}

    if "library_id" in z.files:
        library_id = z["library_id"].astype(str)
        if library_id.shape[0] != X.shape[0]:
            raise ValueError(f"library_id length ({library_id.shape[0]}) != X rows ({X.shape[0]})")
        obs["library_id"] = library_id

    return ad.AnnData(X=X, obs=pd.DataFrame(obs))

################################ SAVING/LOADING GENERIC JSON ##################################

def load_json(path: str | Path) -> Dict[str, Any]:
    """
    Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON as a dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str | Path, *, indent: int = 2, sort_keys: bool = True) -> None:
    """
    Save a dict to JSON.

    Args:
        data: Dictionary to write.
        path: Destination path.
        indent: JSON indentation.
        sort_keys: Whether to sort keys.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)
        f.write("\n")

################################ PARAMETER COMBINATIONS ##################################

def calculate_combinations(params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Calculate all combinations of the given parameters."""
    from itertools import product

    keys = params.keys()
    values = params.values()
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return combinations

class ComboConfig:
    """
    Generic, dynamically-populated combo.

    Any key you pass in becomes an attribute:
        Combo(feature_extraction="virchow", tile_px=256)
        -> combo.feature_extraction, combo.tile_px
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_keys_values(cls, keys: list[str], values: list[object]) -> "ComboConfig":
        data = {k: v for k, v in zip(keys, values)}
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)  
