from __future__ import annotations

from typing import Any, Dict, Optional, List
import logging

import lazyslide as zs
from wsidata import open_wsi

import numpy as np
import json
import pandas as pd
import geopandas as gpd
import anndata as ad
import timm
import torch
from shapely.geometry import Polygon
from spatialdata.models import ShapesModel

from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.utils.registries import SLIDE_PROCESSORS
from pathbench.core.datasets.wsi import WSI

logger = logging.getLogger(__name__)


@SLIDE_PROCESSORS.register("lazyslide")
class LazySlideProcessor(SlideProcessorBase):
    """
    Conforms to policy canonical structures:

    - segment_tissue(config) -> list[np.ndarray] (each polygon: (N,2) xy in pixels)
    - extract_patches(tissues, config) -> pd.DataFrame with columns: tile_id, x, y
    - extract_features(tiles_df, config) -> AnnData with obs['tile_id'] (and spatial can be added)
    """

    def __init__(self) -> None:
        super().__init__()

    # ----------------- Conversions (policy <-> lazyslide) -----------------
    def _tissues_table_to_policy(self, tissues_table: Any) -> List[np.ndarray]:
        """
        Convert LazySlide tissues table (with shapely geometry) to policy tissues list.
        """
        df = tissues_table
        if "geometry" not in df.columns:
            raise ValueError("[LazySlide] wsi['tissues'] missing 'geometry' column.")

        out: List[np.ndarray] = []
        for geom in df["geometry"].tolist():
            if geom is None:
                continue
            coords = np.asarray(geom.exterior.coords, dtype=np.float32)  # includes closing point
            out.append(coords[:, :2])
        return out

    def _policy_tissues_to_tissues_table(self, tissues: List[np.ndarray]):
        rows = []
        geoms = []
        for i, poly in enumerate(tissues):
            arr = np.asarray(poly, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
                raise ValueError(f"[LazySlide] Invalid tissue polygon at index {i}: expected (N,2), got {arr.shape}")
            if not np.allclose(arr[0], arr[-1]):
                arr = np.vstack([arr, arr[0]])
            rows.append(i)
            geoms.append(Polygon(arr))

        gdf = gpd.GeoDataFrame({"tissue_id": rows}, geometry=geoms)

        return ShapesModel.parse(gdf)

    def _tiles_table_to_policy(self, tiles_table: Any) -> pd.DataFrame:
        """
        Convert LazySlide tiles table (tile_id,tissue_id,geometry) to policy tiles_df (tile_id,x,y).
        """
        df = tiles_table.copy()

        if "tile_id" not in df.columns:
            df["tile_id"] = df.index.astype(int)

        if "geometry" not in df.columns:
            raise ValueError("[LazySlide] wsi['tiles'] missing 'geometry' column.")

        bounds = df["geometry"].apply(lambda g: g.bounds if g is not None else (np.nan, np.nan, np.nan, np.nan))
        bounds = np.asarray(list(bounds), dtype=np.float32)

        out = pd.DataFrame(
            {
                "tile_id": df["tile_id"].astype(str).to_numpy(),
                "x": bounds[:, 0],
                "y": bounds[:, 1],
            }
        )
        return out

    def _policy_tiles_to_tiles_table(self, tiles_df: pd.DataFrame, tile_px: int, template=None):
        required = {"tile_id", "x", "y"}
        missing = required - set(tiles_df.columns)
        if missing:
            raise ValueError(f"[LazySlide] tiles_df missing required columns: {sorted(missing)}")

        xs = tiles_df["x"].to_numpy(dtype=np.float32)
        ys = tiles_df["y"].to_numpy(dtype=np.float32)

        geoms = [
            Polygon([
                (float(x), float(y)),
                (float(x + tile_px), float(y)),
                (float(x + tile_px), float(y + tile_px)),
                (float(x), float(y + tile_px)),
            ])
            for x, y in zip(xs, ys)
        ]

        gdf = gpd.GeoDataFrame(
            {
                "tile_id": tiles_df["tile_id"].astype(str).to_numpy(),
                "tissue_id": np.zeros(len(tiles_df), dtype=np.int32),
            },
            geometry=geoms,  # <- key fix
        )

        if template is not None and hasattr(template, "crs") and template.crs is not None:
            gdf = gdf.set_crs(template.crs, allow_override=True)

        return ShapesModel.parse(gdf)
    
    def _tile_spec_to_obj(self, tile_spec: str) -> dict:
        """
        Strictly validate and parse tile_spec.

        Contract:
        - tile_spec must be a Python str
        - tile_spec must be valid JSON (json.loads succeeds)
        - must contain minimal lazyslide-required structure
        """
        import json

        if not isinstance(tile_spec, str):
            raise TypeError(f"[LazySlide] tile_spec must be a JSON string (str), got {type(tile_spec)}")

        try:
            spec = json.loads(tile_spec)
        except Exception as e:
            raise ValueError(f"[LazySlide] tile_spec is not valid JSON: {e}") from e

        tiles = spec.get("tiles")
        if not isinstance(tiles, dict):
            raise ValueError("[LazySlide] tile_spec JSON must contain object at key 'tiles'.")

        for k in ["width", "height", "mpp", "ops_level", "tissue_name"]:
            if k not in tiles:
                raise ValueError(f"[LazySlide] tile_spec JSON missing required key tiles['{k}'].")

        return spec
    
    def _tile_spec_from_obj(self, spec_obj: dict) -> str:
        """
        Strictly validate + serialize a backend tile_spec object to the policy contract format.

        Contract output:
        - returns a Python str
        - valid JSON (json.loads(...) works)
        - contains minimal lazyslide-required structure
        """
        import json

        if not isinstance(spec_obj, dict):
            raise TypeError(f"[LazySlide] tile_spec object must be dict, got {type(spec_obj)}")

        tiles = spec_obj.get("tiles")
        if not isinstance(tiles, dict):
            raise ValueError("[LazySlide] tile_spec object must contain dict at key 'tiles'.")

        for k in ["width", "height", "mpp", "ops_level", "tissue_name"]:
            if k not in tiles:
                raise ValueError(f"[LazySlide] tile_spec object missing required key tiles['{k}'].")

        # Strict JSON serialization (no numpy scalars, no non-serializable objects)
        try:
            tile_spec = json.dumps(spec_obj, separators=(",", ":"), sort_keys=True)
        except TypeError as e:
            raise TypeError(f"[LazySlide] tile_spec object is not JSON-serializable: {e}") from e

        # Round-trip validation
        json.loads(tile_spec)
        return tile_spec
    
    # ----------------- Policy methods -----------------
    def load_wsi(self, wsi: WSI) -> None:
        """
        LazySlide uses wsidata.open_wsi(...) which returns a wsidata.WSIData object.
        """
        if getattr(wsi, "_obj", None) is not None:
            return

        wsi._obj = open_wsi(wsi.path) 

    def close_wsi(self, wsi: WSI) -> None:
        """
        wsidata.WSIData has .close() which closes the underlying reader.
        """
        obj = getattr(wsi, "_obj", None)
        if obj is None:
            return

        try:
            obj.close()
        finally:
            wsi._obj = None

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

        return self._tissues_table_to_policy(wsi.obj["tissues"])

    def extract_patches(self, wsi: WSI, tissues: List[np.ndarray], config: Dict[str, Any]) -> pd.DataFrame:

        # Policy is ground truth: inject tissues
        wsi.obj["tissues"] = self._policy_tissues_to_tissues_table(tissues)

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
        
        # export tile_spec as JSON string (keys can be variable)
        if not hasattr(wsi.obj, "attrs") or "tile_spec" not in wsi.obj.attrs:
            raise RuntimeError("[LazySlide] Tiling finished but wsi.obj.attrs['tile_spec'] is missing.")
        tile_spec_obj = wsi.obj.attrs["tile_spec"]

        tile_spec = self._tile_spec_from_obj(tile_spec_obj)
        tiles_df = self._tiles_table_to_policy(wsi.obj["tiles"])

        return tiles_df, tile_spec
    
    def validate_tile_spec(self, tile_spec: Optional[str], config: Optional[Dict[str, Any]] = None) -> bool:
        if tile_spec is None:
            return False
        try:
            spec = self._tile_spec_to_obj(tile_spec)
        except Exception:
            return False

        # optional consistency checks with config (strict but minimal)
        if config is not None:
            tiles = spec["tiles"]
            tile_px = config.get("tile_px")
            if tile_px is not None and int(tiles["width"]) != int(tile_px):
                return False

            tile_mpp = config.get("tile_mpp")
            if tile_mpp is not None and abs(float(tiles["mpp"]) - float(tile_mpp)) > 1e-6:
                return False

        return True

    def extract_features(self, wsi: WSI, tiles_df: pd.DataFrame, tile_spec: str, config: Dict[str, Any]) -> ad.AnnData:

        model_name = config.get("model", "resnet50")
        params = dict(config.get("params", {}))

        tile_px = config.get("tile_px")
        if tile_px is None:
            raise ValueError("[LazySlide] extract_features requires tile_px in config to reconstruct tile geometries.")

        # apply tile_spec to restore wsidata/lazyslide internal specjson
        wsi.obj.attrs["tile_spec"] = self._tile_spec_to_obj(tile_spec)

        # Policy is ground truth: inject tiles (as before)
        wsi.obj["tiles"] = self._policy_tiles_to_tiles_table(tiles_df, tile_px=tile_px)

        if "device" not in params and torch.cuda.is_available():
            params["device"] = "cuda"

        available = zs.models.list_models() + timm.list_models()
        if model_name not in available:
            raise ValueError(f"[LazySlide] Model '{model_name}' not found in LazySlide/timm.")

        logger.info(
            "[LazySlide] Feature extraction: model=%s, device=%s, params=%s",
            model_name,
            params.get("device", "default"),
            {k: v for k, v in params.items() if k != "device"},
        )

        logger.info("[LazySlide] torch.cuda.is_available()=%s", torch.cuda.is_available())
        logger.info("[LazySlide] params['device']=%s", params.get("device", None))
        if torch.cuda.is_available():
            logger.info("[LazySlide] torch.cuda.current_device()=%s", torch.cuda.current_device())
            logger.info("[LazySlide] torch.cuda.get_device_name()=%s", torch.cuda.get_device_name(0))

        zs.tl.feature_extraction(wsi=wsi.obj, model=model_name, **params)

        key = f"{model_name}_tiles"
        if key not in wsi.obj:
            raise RuntimeError(f"[LazySlide] Feature extraction finished but '{key}' not found on WSI.")

        feats = wsi.obj[key]  # AnnData

        # Ensure spatial exists using tiles_df (unchanged)
        if "spatial" not in feats.obsm:
            tiles_key = tiles_df.set_index("tile_id")[["x", "y"]]
            feat_tile_ids = feats.obs["tile_id"].astype(str).to_numpy()
            try:
                feats.obsm["spatial"] = tiles_key.loc[feat_tile_ids].to_numpy(dtype=np.float32)
            except KeyError:
                logger.warning("[LazySlide] Could not align features tile_id to provided tiles_df for spatial.")

        return feats

    def extract_cells(self, wsi: WSI, config: Dict[str, Any]) -> Any:
        #Parameter handling
        cell_seg_model = config.get("cell_segmentation", None).get("model", "instanseg")
        cell_type_classification = config.get("cell_segmentation", None).get("cell_type_classification", False)
        if cell_type_classification:
            cell_class_model = config.get("cell_segmentation", None).get("cell_type_model", "histoplus")
        params = dict(config.get("cell_segmentation", None)).get("params", {})
        logger.info("[LazySlide] Cell segmentation: model='%s', cell_type_classification=%s, params=%s", 
                    cell_seg_model, cell_type_classification, params)
        
        #Define keys
        cells_key = "{model}_cells".format(model=cell_seg_model)
        cell_types_key = "{model}_cell_types".format(model=cell_class_model) if cell_type_classification else None
    
        # Tissue tiling with overlap for better cell segmentation
        zs.pp.find_tissues(wsi=wsi.obj)
        zs.pp.tile_tissues(wsi=wsi.obj, tile_size=params.get("tile_size", 512), overlap=params.get("overlap", 0.2), mpp=params.get("mpp", 0.5))
        logger.info("[LazySlide] Tiled tissues for cell segmentation.")
        
        #Cell type segmentation
        zs.seg.cells(wsi=wsi.obj, model=cell_seg_model, transform=params.get("transform", None), num_workers=params.get("num_workers", 0),
                     batch_size=params.get("batch_size", 4), key_added=cells_key)
        logger.info("[LazySlide] Cell segmentation done, added to %s number of cells: %d", cells_key, len(wsi.obj[cells_key]))
        
        #Assert cells were found
        assert len(wsi.obj[cells_key]) > 0, "[LazySlide] No cells were segmented, cannot proceed to cell type classification."
        
        if cell_type_classification:
            # Cell type classification
            zs.seg.cell_types(wsi=wsi.obj, model=cell_class_model, magnification=params.get("mpp", None), transform=params.get("transform", None),
                              batch_size=params.get("batch_size", 4), num_workers=params.get("num_workers", 0), key_added=cell_types_key)
            logger.info("[LazySlide] Cell type classification done, added to %s number of classified cells: %d", cell_types_key, len(wsi.obj[cell_types_key]))
            
            #Assert cell types were classified
            assert len(wsi.obj[cell_types_key]) > 0, "[LazySlide] No cell types were classified, something went wrong."
        
    def inspect_slide(self, wsi: WSI) -> None: #TODO: Not in line with policy as defined in this branch
        """
        Print a summary of the slide object for debugging.
        """
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
