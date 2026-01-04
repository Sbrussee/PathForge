# src/pathbench/core/io/features.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

import logging
import anndata as ad

from pathbench.core.io.base import (
    ensure_base_path,
    detect_artifact_path,
    ensure_parent_dir,
    ensure_suffix,
)

logger = logging.getLogger(__name__)

_FeaturesLoader = Callable[[Path], ad.AnnData]
_FeaturesSaver = Callable[[ad.AnnData, Path], None]

FEATURES_CODECS: dict[str, tuple[_FeaturesLoader, _FeaturesSaver]] = {}

DEFAULT_SUFFIX = ".pt"
SUPPORTED_SUFFIXES: tuple[str, ...]  # set after registration

def load_features(base: Path) -> Optional[ad.AnnData]:
    """Load features for a base path. Incomplete caches are treated as missing."""
    base = ensure_base_path(base)
    p = detect_artifact_path(base)
    if p is None:
        return None

    codec = FEATURES_CODECS.get(p.suffix)
    if codec is None:
        raise ValueError(f"Unsupported features suffix '{p.suffix}'. Supported: {list(SUPPORTED_SUFFIXES)}")
    loader, _ = codec
    
    try:
        return loader(p)
    except FileNotFoundError as e:
        # Most important case: .pt exists but .index.npz missing (partial write / interrupted run)
        logger.info("[io.features] Incomplete features cache at %s (%s). Treating as missing.", p, e)
        return None


def save_features(
    adata: ad.AnnData,
    base: Path,
    *,
    suffix: str = DEFAULT_SUFFIX,
) -> Path:
    base = ensure_base_path(base)

    codec = FEATURES_CODECS.get(suffix)
    if codec is None:
        raise ValueError(f"Unsupported features suffix '{suffix}'. Supported: {list(SUPPORTED_SUFFIXES)}")

    out = base.with_suffix(suffix)
    ensure_parent_dir(out)

    _, saver = codec
    saver(adata, out)
    return out

# ---- H5AD ----

def load_h5ad(path: str | Path) -> ad.AnnData:
    path = ensure_suffix(path, ".h5ad")
    if not path.exists():
        raise FileNotFoundError(f"AnnData .h5ad file not found: {path}")
    return ad.read_h5ad(path)


def save_h5ad_atomic(adata: ad.AnnData, path: str | Path) -> None:
    path = ensure_suffix(path, ".h5ad")
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        adata.write_h5ad(tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


# ---- PT + INDEX ----

def save_features_pt(adata: ad.AnnData, path: str | Path) -> None:
    """
    Save feature AnnData as:
      - <path>.pt: tensor only (N, D)
      - <path>.index.npz: obs fields to map rows (tile_id, optionally library_id)
    """
    import numpy as np
    import torch

    path = Path(path)
    pt_path = ensure_suffix(path, ".pt")
    idx_path = pt_path.with_suffix(".index.npz")

    pt_path.parent.mkdir(parents=True, exist_ok=True)

    if adata.X is None:
        raise ValueError("AnnData.X is None; expected feature matrix.")
    if "tile_id" not in adata.obs.columns:
        raise ValueError("AnnData must contain obs['tile_id'].")

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"AnnData.X must be 2D (N, D). Got shape={X.shape}")
    X = X.astype(np.float32, copy=False)

    tile_id = adata.obs["tile_id"].astype(str).to_numpy()
    if tile_id.shape[0] != X.shape[0]:
        raise ValueError(f"obs['tile_id'] length ({tile_id.shape[0]}) != X rows ({X.shape[0]})")

    have_library = "library_id" in adata.obs.columns
    if have_library:
        library_id = adata.obs["library_id"].astype(str).to_numpy()
        if library_id.shape[0] != X.shape[0]:
            raise ValueError(f"obs['library_id'] length ({library_id.shape[0]}) != X rows ({X.shape[0]})")

    pt_tmp = pt_path.with_name(f".{pt_path.name}.tmp")
    idx_tmp = idx_path.with_name(f".{idx_path.name}.tmp.npz")

    try:
        torch.save(torch.from_numpy(X), pt_tmp)
        pt_tmp.replace(pt_path)

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
    """
    import numpy as np
    import torch
    import pandas as pd

    path = Path(path)
    pt_path = ensure_suffix(path, ".pt")
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

    obs: dict[str, object] = {"tile_id": tile_id}

    if "library_id" in z.files:
        library_id = z["library_id"].astype(str)
        if library_id.shape[0] != X.shape[0]:
            raise ValueError(f"library_id length ({library_id.shape[0]}) != X rows ({X.shape[0]})")
        obs["library_id"] = library_id

    return ad.AnnData(X=X, obs=pd.DataFrame(obs))


# ---- Register codecs ----
FEATURES_CODECS[".pt"] = (load_features_pt, save_features_pt)
FEATURES_CODECS[".h5ad"] = (load_h5ad, save_h5ad_atomic)  # optional; remove if you don't want it
SUPPORTED_SUFFIXES = tuple(FEATURES_CODECS.keys())
