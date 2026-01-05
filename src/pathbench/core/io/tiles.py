# src/pathbench/core/io/tiles.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Callable

import pandas as pd
import json
import numpy as np

from pathbench.core.io.base import (
    ensure_base_path,
    ensure_parent_dir,
    detect_artifact_path,
    ensure_suffix,
)

# ---- Registry (populated at bottom) ----
_TilesLoader = Callable[[Path], Tuple[pd.DataFrame, str]]
_TilesSaver = Callable[[pd.DataFrame, str, Path], None]

TILES_CODECS: dict[str, tuple[_TilesLoader, _TilesSaver]] = {}

DEFAULT_SUFFIX = ".npz"
SUPPORTED_SUFFIXES: tuple[str, ...]  # set after registration


# ---- Public API ----

def load_tiles(base: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load tiles for a base path."""
    base = ensure_base_path(base)
    p = detect_artifact_path(base)
    if p is None:
        return None, None

    loader, _ = TILES_CODECS[p.suffix]
    return loader(p)


def save_tiles(
    tiles_df: pd.DataFrame,
    tile_spec: str,
    base: Path,
    *,
    suffix: str = DEFAULT_SUFFIX,
) -> Path:
    """Save tiles for a base path."""
    base = ensure_base_path(base)

    codec = TILES_CODECS.get(suffix)
    if codec is None:
        raise ValueError(f"Unsupported tiles suffix '{suffix}'. Supported: {list(SUPPORTED_SUFFIXES)}")

    out = base.with_suffix(suffix)
    ensure_parent_dir(out)

    _, saver = codec
    saver(tiles_df, tile_spec, out)
    return out


# ---- NPZ ----

def save_tiles_npz(tiles_df: pd.DataFrame, tile_spec: str, path: Path) -> None:
    """
    Save tiles + tile_spec (JSON string) to a compressed NPZ.

    Policy contract:
      - tile_spec must be a Python str containing valid JSON.
    """
    path = Path(path)
    path = ensure_suffix(path, ".npz")

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

    x_arr = tiles_df["x"].to_numpy(dtype=np.float32)
    y_arr = tiles_df["y"].to_numpy(dtype=np.float32)

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
    path = Path(path)
    path = ensure_suffix(path, ".npz")

    if not path.exists():
        raise FileNotFoundError(f"Tiles npz file not found: {path}")

    z = np.load(path, allow_pickle=False)

    tile_id_kind = z["tile_id_kind"]
    if isinstance(tile_id_kind, np.ndarray):
        tile_id_kind = tile_id_kind.item()  # robust for 0-D scalars
    tile_id_kind = str(tile_id_kind)

    tile_id_arr = z["tile_id"]
    if tile_id_kind == "int":
        tile_id = tile_id_arr.astype(np.int64)
    else:
        tile_id = tile_id_arr.astype(str)

    tiles_df = pd.DataFrame(
        {
            "tile_id": tile_id,
            "x": z["x"].astype(np.float32),
            "y": z["y"].astype(np.float32),
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


# ---- Register codecs ----
TILES_CODECS[".npz"] = (load_tiles_npz, save_tiles_npz)
SUPPORTED_SUFFIXES = tuple(TILES_CODECS.keys())
