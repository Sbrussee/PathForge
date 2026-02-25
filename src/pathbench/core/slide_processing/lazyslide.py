# src/pathbench/core/slide_processing/lazyslide.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
import logging

import lazyslide as zs
from wsidata import open_wsi

import numpy as np
import pandas as pd
import geopandas as gpd
import anndata as ad  # noqa: F401  # returned by lazyslide (kept for clarity)
import timm
import torch
from shapely.geometry import Polygon
from spatialdata.models import ShapesModel

from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.utils.registries import SLIDE_PROCESSORS
from pathbench.core.datasets.wsi_dataset import WSI

logger = logging.getLogger(__name__)


@SLIDE_PROCESSORS.register("lazyslide")
class LazySlideProcessor(SlideProcessorBase):
    """
    Policy contracts (canonical):
    - segment_tissue(...) -> list[np.ndarray]          # polygons, each (N,2) float32, level-0 pixels
    - extract_patches(...) -> (coords, tiling_spec)    # coords: (N,5) int32, tiling_spec: dict (stored in H5)
    - validate_tile_spec(...) -> bool                  # validates tiling_spec dict vs config
    - extract_features(...) -> np.ndarray              # (N,D) float32 row-aligned with coords

    Storage contract (H5):
    - coords: (N,5) int32 columns:
        [x_level0, y_level0, read_w_at_level, read_h_at_level, level]
    - tiling_spec (JSON, backend-agnostic):
        {tile_px, tile_mpp, stride_px, coord_space="level0", backend?}

    Lazyslide/wsidata-specific tile_spec (wsi.obj.attrs["tile_spec"]) is reconstructed
    from coords + tiling_spec when extracting features.
    """

    BACKEND_NAME = "lazyslide"
    COORD_SPACE = "level0"

    def __init__(self) -> None:
        super().__init__()

    # ---------------------------------------------------------------------
    # Conversions: backend -> policy
    # ---------------------------------------------------------------------

    def _backend_tissues_to_policy(self, tissues_table: Any) -> List[np.ndarray]:
        """Convert lazyslide tissues table -> policy tissues list[np.ndarray] (N,2), level-0 pixel coords."""
        df = tissues_table
        if "geometry" not in df.columns:
            raise ValueError("[LazySlide] wsi['tissues'] missing 'geometry' column.")

        polygons: List[np.ndarray] = []
        for geom in df["geometry"].tolist():
            if geom is None:
                continue
            coords = np.asarray(geom.exterior.coords, dtype=np.float32)  # includes closing point
            polygons.append(coords[:, :2])
        return polygons


    def _backend_tiles_to_policy_coords(self, tiles_table: Any, tile_spec_obj: dict) -> np.ndarray:
        """
        Convert lazyslide tiles table + wsidata tile_spec -> policy coords (N,5) int32:
        [x_level0, y_level0, read_w_at_level, read_h_at_level, read_level]

        - x/y come from tile polygon bounds (base/level-0 coordinate space)
        - read_level is ops_level
        - read_w/read_h are operation-level read window sizes:
            read_w = round(tile_px * ops_downsample)
        """
        df = tiles_table.copy()

        if "tile_id" not in df.columns:
            df["tile_id"] = df.index.astype(int)

        # deterministic row order
        tile_id_numeric = pd.to_numeric(df["tile_id"], errors="coerce")
        df = df.iloc[np.argsort(tile_id_numeric.to_numpy())]

        if "geometry" not in df.columns:
            raise ValueError("[LazySlide] wsi['tiles'] missing 'geometry' column.")

        bounds = df["geometry"].apply(
            lambda g: g.bounds if g is not None else (np.nan, np.nan, np.nan, np.nan)
        )
        bounds = np.asarray(list(bounds), dtype=np.float32)  # (N,4): (minx,miny,maxx,maxy)

        tiles_meta = (tile_spec_obj or {}).get("tiles", {})
        tile_px = int(tiles_meta.get("width", 0))  # destined/output tile size
        tile_mpp = tiles_meta.get("mpp", None)
        ops_level = int(tiles_meta.get("ops_level", -1))
        ops_downsample = float(tiles_meta.get("ops_downsample", 1.0))

        if tile_px <= 0:
            raise ValueError("[LazySlide] tile_spec_obj['tiles']['width'] must be positive.")
        if tile_mpp is None:
            raise ValueError("[LazySlide] tile_spec_obj['tiles']['mpp'] is missing.")
        if ops_level < 0:
            raise ValueError("[LazySlide] tile_spec_obj['tiles']['ops_level'] is missing/invalid.")
        if ops_downsample <= 0:
            raise ValueError("[LazySlide] tile_spec_obj['tiles']['ops_downsample'] must be > 0.")

        x0 = np.rint(bounds[:, 0]).astype(np.int32)
        y0 = np.rint(bounds[:, 1]).astype(np.int32)

        read_w = int(round(tile_px * ops_downsample))
        read_h = int(round(tile_px * ops_downsample))
        if read_w <= 0 or read_h <= 0:
            raise ValueError("[LazySlide] Derived read_w/read_h must be positive.")

        return np.stack(
            [
                x0,
                y0,
                np.full(len(df), read_w, dtype=np.int32),
                np.full(len(df), read_h, dtype=np.int32),
                np.full(len(df), ops_level, dtype=np.int32),
            ],
            axis=1,
        ).astype(np.int32)

    def _backend_tile_spec_to_policy_tiling_spec(self, *, config: Dict[str, Any], tile_spec_obj: dict) -> dict:
        """
        Build the backend-agnostic tiling_spec we store in H5 for a bag.

        Stored keys:
        - tile_px (output tile size)
        - tile_mpp (target mpp)
        - stride_px (output stride)
        - coord_space ("level0")
        - backend ("lazyslide")
        """
        tiles_meta = (tile_spec_obj or {}).get("tiles", {})

        tile_px = tiles_meta.get("width", config.get("tile_px"))
        tile_mpp = tiles_meta.get("mpp", config.get("tile_mpp"))
        stride_px = tiles_meta.get("stride_width", tile_px)

        if tile_px is None or tile_mpp is None:
            raise ValueError("[LazySlide] Cannot build H5 tiling_spec: missing tile_px/tile_mpp.")
        if int(tile_px) <= 0 or float(tile_mpp) <= 0:
            raise ValueError("[LazySlide] Invalid tile_px/tile_mpp when building H5 tiling_spec.")

        return {
            "tile_px": int(tile_px),
            "tile_mpp": float(tile_mpp),
            "stride_px": int(stride_px),
            "coord_space": self.COORD_SPACE,
            "backend": self.BACKEND_NAME,
        }

    # ---------------------------------------------------------------------
    # Conversions: policy -> backend
    # ---------------------------------------------------------------------

    def _policy_tissues_to_backend(self, tissues: List[np.ndarray]):
        """Convert policy tissues list[np.ndarray] -> lazyslide ShapesModel."""
        rows: list[int] = []
        geoms: list[Polygon] = []
        for i, poly in enumerate(tissues):
            arr = np.asarray(poly, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
                raise ValueError(
                    f"[LazySlide] Invalid tissue polygon at index {i}: expected (N,2), got {arr.shape}"
                )
            if not np.allclose(arr[0], arr[-1]):
                arr = np.vstack([arr, arr[0]])
            rows.append(i)
            geoms.append(Polygon(arr))

        gdf = gpd.GeoDataFrame({"tissue_id": rows}, geometry=geoms)
        return ShapesModel.parse(gdf)


    def _xy_to_backend_tiles_table(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        tile_px: int,
        tile_id: Optional[np.ndarray] = None,
        template=None,
    ):
        """
        Convert tile top-left anchors (x,y) -> lazyslide ShapesModel tiles table.

        This is the *single* conversion: xy -> polygons -> ShapesModel.
        No intermediate DataFrame is required.

        Args:
            x, y: arrays of top-left coordinates in level-0 pixels
            tile_px: destined/output tile size (used to build tile polygons)
            tile_id: optional string ids; if None, uses 0..N-1
        """
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("[LazySlide] x and y must be 1D arrays with the same length.")
        if int(tile_px) <= 0:
            raise ValueError("[LazySlide] tile_px must be > 0.")

        if tile_id is None:
            tile_id_arr = np.arange(x.shape[0]).astype(str)
        else:
            tile_id_arr = np.asarray(tile_id).astype(str)
            if tile_id_arr.shape[0] != x.shape[0]:
                raise ValueError("[LazySlide] tile_id length must match x/y length.")

        geoms = [
            Polygon(
                [
                    (float(xi), float(yi)),
                    (float(xi + tile_px), float(yi)),
                    (float(xi + tile_px), float(yi + tile_px)),
                    (float(xi), float(yi + tile_px)),
                ]
            )
            for xi, yi in zip(x, y)
        ]

        gdf = gpd.GeoDataFrame(
            {
                "tile_id": tile_id_arr,
                "tissue_id": np.zeros(len(tile_id_arr), dtype=np.int32),
            },
            geometry=geoms,
        )

        if template is not None and hasattr(template, "crs") and template.crs is not None:
            gdf = gdf.set_crs(template.crs, allow_override=True)

        return ShapesModel.parse(gdf)


    def _reconstruct_tile_spec(self, coords: np.ndarray, tiling_spec: dict) -> dict:
        """
        Build a wsidata/lazyslide-compatible `wsi.obj.attrs["tile_spec"]` dict from:
        - coords: (N,5) [x0,y0,read_w,read_h,read_level]
        - tiling_spec: {tile_px,tile_mpp,stride_px,coord_space,...}
        """
        coords = np.asarray(coords, dtype=np.int32)
        if coords.ndim != 2 or coords.shape[1] != 5:
            raise ValueError(f"[LazySlide] coords must be (N,5) int32, got {coords.shape}")
        if coords.shape[0] == 0:
            raise ValueError("[LazySlide] Cannot reconstruct tile_spec from empty coords.")

        tile_px = int(tiling_spec.get("tile_px", 0))
        tile_mpp = tiling_spec.get("tile_mpp", None)
        stride_px = int(tiling_spec.get("stride_px", tile_px))

        if tile_px <= 0 or tile_mpp is None:
            raise ValueError("[LazySlide] tiling_spec must contain valid tile_px and tile_mpp.")
        if stride_px <= 0:
            raise ValueError("[LazySlide] tiling_spec.stride_px must be positive.")

        read_w = int(np.unique(coords[:, 2])[0])
        read_h = int(np.unique(coords[:, 3])[0])
        read_level = int(np.unique(coords[:, 4])[0])

        if read_w <= 0 or read_h <= 0:
            raise ValueError("[LazySlide] coords read_w/read_h must be positive.")
        if read_level < 0:
            raise ValueError("[LazySlide] coords read_level must be >= 0.")

        ops_downsample_w = float(read_w) / float(tile_px)
        ops_downsample_h = float(read_h) / float(tile_px)
        if abs(ops_downsample_w - ops_downsample_h) > 1e-6:
            logger.warning(
                "[LazySlide] read_w/read_h imply anisotropic ops_downsample (w=%s,h=%s). "
                "Using width-derived value for ops_downsample.",
                ops_downsample_w,
                ops_downsample_h,
            )
        ops_downsample = float(ops_downsample_w)

        return {
            "tiles": {
                "width": tile_px,
                "height": tile_px,
                "stride_width": stride_px,
                "stride_height": stride_px,
                "mpp": float(tile_mpp),
                "ops_level": read_level,
                "ops_downsample": ops_downsample,
                "tissue_name": "tissues",
            }
        }


    def _reconstruct_tiles_table(self, coords: np.ndarray, *, tile_px: int):
        """
        Rebuild lazyslide `wsi.obj["tiles"]` table from coords.

        No DataFrame: directly transform (x,y) anchors into polygons.
        """
        coords = np.asarray(coords, dtype=np.int32)
        if coords.ndim != 2 or coords.shape[1] != 5:
            raise ValueError(f"[LazySlide] coords must be (N,5) int32, got {coords.shape}")

        x = coords[:, 0].astype(np.float32)
        y = coords[:, 1].astype(np.float32)
        tile_id = np.arange(coords.shape[0]).astype(str)

        return self._xy_to_backend_tiles_table(x, y, tile_px=int(tile_px), tile_id=tile_id)
    
    # ---------------------------------------------------------------------
    # Thumnail helpers
    # ---------------------------------------------------------------------

    def _get_level0_shape(self, wsi_obj: Any) -> tuple[int, int]:
        """
        Return level-0 shape as (height, width) from LazySlide/WSIData.

        In our environment, `wsi_obj.properties.shape` is in (H, W) order.
        """
        props = getattr(wsi_obj, "properties", None)
        if props is None or not hasattr(props, "shape"):
            raise RuntimeError("[LazySlide] Could not determine level-0 shape: missing wsi.obj.properties.shape.")

        shape = getattr(props, "shape")
        try:
            h0 = int(shape[0])
            w0 = int(shape[1])
        except Exception as e:
            raise RuntimeError(
                f"[LazySlide] Invalid wsi.obj.properties.shape format: {shape!r}"
            ) from e

        if h0 <= 0 or w0 <= 0:
            raise RuntimeError(f"[LazySlide] Invalid level-0 shape values: {shape!r}")

        return h0, w0


    def _thumbnail_to_rgb_uint8_numpy(self, thumb_obj: Any) -> np.ndarray:
        """
        Convert LazySlide/SpatialData thumbnail object into HxWx3 uint8 numpy array.
        """
        # Common wrappers (xarray / image models)
        src = thumb_obj
        if hasattr(src, "values"):
            src = src.values
        elif hasattr(src, "data"):
            src = src.data

        arr = np.asarray(src)

        # If channel-first (C,H,W), convert to H,W,C
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)

        # grayscale -> RGB
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            raise RuntimeError(
                f"[LazySlide] Unsupported thumbnail array shape for visualization: {arr.shape}"
            )

        # RGBA -> RGB
        if arr.shape[-1] == 4:
            arr = arr[..., :3]

        # Normalize dtype to uint8
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
                # If likely 0..1 floats, scale to 0..255
                if arr.size > 0:
                    max_val = float(np.max(arr))
                    if max_val <= 1.5:
                        arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr

    # ---------------------------------------------------------------------
    # SlideProcessorBase API
    # ---------------------------------------------------------------------

    def load_wsi(self, wsi: WSI) -> None:
        if getattr(wsi, "_obj", None) is not None:
            return
        wsi._obj = open_wsi(wsi.path)

    def close_wsi(self, wsi: WSI) -> None:
        obj = getattr(wsi, "_obj", None)
        if obj is None:
            return
        try:
            obj.close()
        finally:
            wsi._obj = None

    # ---------------------------------------------------------------------
    # Policy methods
    # --------------------------------------------------------------------- 

    def get_thumbnail(self, wsi: WSI, level: int = -1) -> Tuple[Any, float, float]:
        """
        Return a thumbnail image and downscale factors relative to level-0 coords.

        Returns:
            (thumbnail_image, downscale_x, downscale_y)

        Notes:
        - Current implementation prefers LazySlide's existing 'wsi_thumbnail'.
        - `level` is accepted for API compatibility; for now we use the stored thumbnail.
        """
        if getattr(wsi, "_obj", None) is None:
            raise RuntimeError("[LazySlide] WSI not loaded. Call load_wsi(wsi) first.")

        if "wsi_thumbnail" not in wsi.obj:
            raise RuntimeError("[LazySlide] 'wsi_thumbnail' not found on loaded WSI object.")

        thumb_obj = wsi.obj["wsi_thumbnail"]
        thumb = self._thumbnail_to_rgb_uint8_numpy(thumb_obj)  # HxWx3 uint8

        h_thumb, w_thumb = thumb.shape[:2]
        if h_thumb <= 0 or w_thumb <= 0:
            raise RuntimeError("[LazySlide] Invalid thumbnail dimensions.")

        h0, w0 = self._get_level0_shape(wsi.obj)

        downscale_x = float(w0) / float(w_thumb)
        downscale_y = float(h0) / float(h_thumb)

        return thumb, downscale_x, downscale_y

    def segment_tissue(self, wsi: WSI, config: Dict[str, Any]) -> List[np.ndarray]:
        method = config.get("method", "otsu")
        params = dict(config.get("params", {}))

        logger.info("[LazySlide] Tissue segmentation: method='%s', params=%s", method, params)

        if method == "otsu":
            zs.pp.find_tissues(wsi=wsi.obj, **params)
        else:
            raise NotImplementedError(f"[LazySlide] Tissue segmentation method '{method}' not implemented.")

        if "tissues" not in wsi.obj:
            raise RuntimeError("[LazySlide] Segmentation ran but no 'tissues' table found on WSI.")

        return self._backend_tissues_to_policy(wsi.obj["tissues"])

    def extract_patches(self, wsi: WSI, tissues: List[np.ndarray], config: Dict[str, Any]):
        """
        Produce:
        - coords: (N,5) int32 [x0,y0,read_w,read_h,level]
        - tiling_spec: dict written to H5 (backend-agnostic)
        """
        # Policy is ground truth: inject tissues
        wsi.obj["tissues"] = self._policy_tissues_to_backend(tissues)

        params = dict(config.get("params", {}))
        tile_px = config.get("tile_px")
        tile_mpp = config.get("tile_mpp")

        if tile_px is not None:
            params.setdefault("tile_px", tile_px)
        if tile_mpp is not None:
            params.setdefault("mpp", tile_mpp)

        logger.info("[LazySlide] Tiling tissues with params=%s", params)
        zs.pp.tile_tissues(wsi=wsi.obj, **params)

        if "tiles" not in wsi.obj:
            raise RuntimeError("[LazySlide] Tiling ran but no 'tiles' table found on WSI.")

        if not hasattr(wsi.obj, "attrs") or "tile_spec" not in wsi.obj.attrs:
            raise RuntimeError("[LazySlide] Tiling finished but wsi.obj.attrs['tile_spec'] is missing.")

        tile_spec_obj = wsi.obj.attrs["tile_spec"]
        coords = self._backend_tiles_to_policy_coords(wsi.obj["tiles"], tile_spec_obj=tile_spec_obj)

        tiling_spec_h5 = self._backend_tile_spec_to_policy_tiling_spec(config=config, tile_spec_obj=tile_spec_obj)

        return coords, tiling_spec_h5

    def validate_tile_spec(self, tiling_spec: Optional[dict], config: Optional[Dict[str, Any]] = None) -> bool:
        if tiling_spec is None or not isinstance(tiling_spec, dict):
            return False

        required_keys = {"tile_px", "tile_mpp", "stride_px", "coord_space"}
        if not required_keys.issubset(set(tiling_spec.keys())):
            return False
        if str(tiling_spec.get("coord_space")) != self.COORD_SPACE:
            return False

        if config is None:
            return True

        try:
            # Validate the "intent" fields against the run config (cache identity)
            return (
                int(tiling_spec["tile_px"]) == int(config.get("tile_px"))
                and abs(float(tiling_spec["tile_mpp"]) - float(config.get("tile_mpp"))) < 1e-6
            )
        except Exception:
            return False

    def extract_features(
        self,
        wsi: WSI,
        coords: np.ndarray,
        tiling_spec: dict,
        config: Dict[str, Any],
    ) -> np.ndarray:
        # Require an explicit model name (never default silently).
        if "model" not in config or not config["model"]:
            raise ValueError("[LazySlide] Feature extraction requires config['model'] (no default).")
        model_name = str(config["model"])
        params = dict(config.get("params", {}))

        coords = np.asarray(coords, dtype=np.int32)
        if coords.ndim != 2 or coords.shape[1] != 5:
            raise ValueError(f"[LazySlide] coords must be (N,5) int32, got {coords.shape}")

        # Warn (do not block) if features are extracted with a backend different than the tiling backend.
        spec_backend = tiling_spec.get("backend")
        if spec_backend and spec_backend != self.BACKEND_NAME:
            logger.warning(
                "[LazySlide] tiling_spec.backend=%r differs from current backend=%r. "
                "Attempting feature extraction anyway using cached coords.",
                spec_backend,
                self.BACKEND_NAME,
            )

        # ---- Rebuild wsidata/lazyslide tiling context ----
        wsi.obj.attrs["tile_spec"] = self._reconstruct_tile_spec(coords, tiling_spec)

        tile_px = int(tiling_spec["tile_px"])
        wsi.obj["tiles"] = self._reconstruct_tiles_table(coords, tile_px=tile_px)

        # ---- Device default ----
        if "device" not in params and torch.cuda.is_available():
            params["device"] = "cuda"

        # ---- Validate model availability ----
        available = zs.models.list_models() + timm.list_models()
        if model_name not in available:
            raise ValueError(f"[LazySlide] Model '{model_name}' not found in LazySlide/timm.")

        logger.info(
            "[LazySlide] Feature extraction: model=%s, device=%s, params=%s",
            model_name,
            params.get("device", "default"),
            {k: v for k, v in params.items() if k != "device"},
        )

        # ---- Run feature extraction ----
        zs.tl.feature_extraction(wsi=wsi.obj, model=model_name, **params)

        key = f"{model_name}_tiles"
        if key not in wsi.obj:
            raise RuntimeError(f"[LazySlide] Feature extraction finished but '{key}' not found on WSI.")

        feats = wsi.obj[key]  # AnnData
        X = feats.X

        # Convert sparse-like matrices to dense without importing scipy (predictable + minimal deps).
        if hasattr(X, "toarray"):
            X = X.toarray()

        features_matrix = np.asarray(X, dtype=np.float32)
        if features_matrix.ndim != 2:
            raise ValueError(f"[LazySlide] Expected 2D feature matrix, got shape {features_matrix.shape}.")

        if features_matrix.shape[0] != coords.shape[0]:
            raise ValueError(
                f"[LazySlide] Features rows ({features_matrix.shape[0]}) != coords rows ({coords.shape[0]})."
            )

        return features_matrix

    # ---------------------------------------------------------------------
    # Optional / legacy methods kept as-is
    # ---------------------------------------------------------------------

    def extract_cells(self, wsi: WSI, config: Dict[str, Any]) -> Any:
        cell_cfg = config.get("cell_segmentation", None)
        if cell_cfg is None:
            raise ValueError("[LazySlide] extract_cells requires config['cell_segmentation'].")

        cell_seg_model = cell_cfg.get("model", "instanseg")
        cell_type_classification = cell_cfg.get("cell_type_classification", False)
        cell_class_model = cell_cfg.get("cell_type_model", "histoplus") if cell_type_classification else None
        params = dict(cell_cfg.get("params", {}))

        logger.info(
            "[LazySlide] Cell segmentation: model='%s', cell_type_classification=%s, params=%s",
            cell_seg_model,
            cell_type_classification,
            params,
        )

        cells_key = f"{cell_seg_model}_cells"
        cell_types_key = f"{cell_class_model}_cell_types" if cell_type_classification else None

        zs.pp.find_tissues(wsi=wsi.obj)
        zs.pp.tile_tissues(
            wsi=wsi.obj,
            tile_size=params.get("tile_size", 512),
            overlap=params.get("overlap", 0.2),
            mpp=params.get("mpp", 0.5),
        )
        logger.info("[LazySlide] Tiled tissues for cell segmentation.")

        zs.seg.cells(
            wsi=wsi.obj,
            model=cell_seg_model,
            transform=params.get("transform", None),
            num_workers=params.get("num_workers", 0),
            batch_size=params.get("batch_size", 4),
            key_added=cells_key,
        )
        logger.info(
            "[LazySlide] Cell segmentation done, added to %s number of cells: %d",
            cells_key,
            len(wsi.obj[cells_key]),
        )

        if len(wsi.obj[cells_key]) == 0:
            raise RuntimeError("[LazySlide] No cells were segmented, cannot proceed to cell type classification.")

        if cell_type_classification:
            zs.seg.cell_types(
                wsi=wsi.obj,
                model=cell_class_model,
                magnification=params.get("mpp", None),
                transform=params.get("transform", None),
                batch_size=params.get("batch_size", 4),
                num_workers=params.get("num_workers", 0),
                key_added=cell_types_key,
            )
            logger.info(
                "[LazySlide] Cell type classification done, added to %s number of classified cells: %d",
                cell_types_key,
                len(wsi.obj[cell_types_key]),
            )

            if len(wsi.obj[cell_types_key]) == 0:
                raise RuntimeError("[LazySlide] No cell types were classified, something went wrong.")

    def inspect_slide(self, wsi: WSI) -> None:
        logger.info("[LazySlide] Inspecting slide object:")
        logger.info("  ID: %s", getattr(wsi.obj, "id", "unknown"))
        logger.info("  Name: %s", getattr(wsi.obj, "name", "unknown"))
        logger.info("  Properties: %s", getattr(wsi.obj, "properties", {}))
        logger.info("  Available tables: %s", list(wsi.obj.keys()))
        for key in wsi.obj.keys():
            table = wsi.obj[key]
            try:
                n_entries = len(table)
                logger.info("    Table '%s': %d entries", key, n_entries)
            except Exception:
                logger.info("    Table '%s': (unable to determine length)", key)

    def __repr__(self) -> str:
        return "LazySlideProcessor"

    def __str__(self) -> str:
        return "LazySlideProcessor"