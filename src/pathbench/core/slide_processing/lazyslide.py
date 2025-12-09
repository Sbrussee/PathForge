from __future__ import annotations
from typing import Any, Dict, Optional
import lazyslide as zs
from wsidata import open_wsi
import timm
import logging
import torch
import numpy as np
import os

logger = logging.getLogger(__name__)

from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.utils.registries import SLIDE_PROCESSORS

@SLIDE_PROCESSORS.register("lazyslide")
class LazySlideProcessor(SlideProcessorBase):
    """
    Concrete implementation of SlideProcessorBase using the LazySlide backend.
    """
    
    def load_slide(self, slide_path: str) -> zs.WSIData:
        logger.debug("[LazySlide] Loading WSI: %s", slide_path)
        return open_wsi(slide_path)
    
    def segment_tissue(self, slide_obj: zs.WSIData, config: Dict[str, Any]) -> Any:
        """
        Run tissue segmentation on the WSI.

        method = "otsu"      -> use zs.pp.find_tissues (simple threshold-based)
            = <modelname> -> use zs.seg.tissue with a learned model
        """
        method = config.get("method", "otsu")
        params = dict(config.get("params", {}))

        logger.info("[LazySlide] Tissue segmentation: method='%s', params=%s", method, params)

        if method == "otsu":
            # Basic Otsu-based segmentation
            zs.pp.find_tissues(wsi=slide_obj, **params)
        else:
            raise NotImplementedError(f"Tissue segmentation method '{method}' not implemented in LazySlideProcessor.")
        
        try:
            tissues = slide_obj["tissues"]
            n_tissues = len(tissues)
            logger.info("[LazySlide] Found %d tissue regions", n_tissues)
        except Exception:
            logger.warning("[LazySlide] No 'tissues' table found after segmentation")

        return slide_obj
    
    def _save_tiles_npz(self, slide_obj: zs.WSIData, tiles_out_path: str) -> None:
        """
        Extract tile top-left coordinates from LazySlide's tiles GeoDataFrame
        and save them as an .npz file.

        .npz structure:
            coords: (N, 2) float32, columns = [x, y]
            tile_ids: (N,) int32 (if available)
        """
        tiles_gdf = slide_obj["tiles"]  # standard Lazyslide tiling output

        coords = []
        tile_ids = []

        # Expect columns: tile_id, geometry (shapely polygon)
        for idx, row in tiles_gdf.iterrows():
            geom = row["geometry"]
            minx, miny, _, _ = geom.bounds  # top-left corner
            coords.append((minx, miny))

            if "tile_id" in tiles_gdf.columns:
                tile_ids.append(row["tile_id"])
            else:
                tile_ids.append(idx)  # fallback: DataFrame index

        coords = np.asarray(coords, dtype=np.float32)
        tile_ids = np.asarray(tile_ids, dtype=np.int32)

        os.makedirs(os.path.dirname(tiles_out_path), exist_ok=True)
        np.savez_compressed(tiles_out_path, coords=coords, tile_ids=tile_ids)

        logger.debug(
            "[LazySlide] Saved tiles: %s (coords shape=%s, n_tiles=%d)",
            tiles_out_path,
            coords.shape,
            coords.shape[0],
        )

    def extract_patches(self, slide_obj: zs.WSIData, config: Dict[str, Any]) -> Any:
        params = dict(config.get("params", {}))
        tile_px = config.get("tile_px")
        tile_mpp = config.get("tile_mpp")

        if tile_px is not None:
            params.setdefault("tile_px", tile_px)
        if tile_mpp is not None:
            params.setdefault("mpp", tile_mpp)

        logger.debug("[LazySlide] Tiling tissues with params=%s", params)
        zs.pp.tile_tissues(wsi=slide_obj, **params)
        return slide_obj
    
    def _save_features_pt(
        self,
        slide_obj: zs.WSIData,
        model_name: str,
        features_out_path: str,
    ) -> None:
        """
        Fetch tile-level features for `model_name` from LazySlide's AnnData
        and save them as a .pt tensor.

        .pt content:
            torch.Tensor of shape (N, D), dtype float32
        """
        # This uses LazySlide's feature API – adjust the call if your version differs.
        adata = slide_obj.fetch.features_anndata(model_name)
        feats = np.asarray(adata.X, dtype=np.float32)

        tensor = torch.from_numpy(feats)
        os.makedirs(os.path.dirname(features_out_path), exist_ok=True)
        torch.save(tensor, features_out_path)

        logger.debug(
            "[LazySlide] Saved features: %s (shape=%s)",
            features_out_path,
            tuple(tensor.shape),
        )


    def extract_features(self, slide_obj: zs.WSIData, config: Dict[str, Any]) -> Any:

        model_name = config.get("model", "resnet50")
        params = config.get("params", {})

        if "device" not in params and torch.cuda.is_available():
            params["device"] = "cuda"

        slide_id = getattr(slide_obj, "name", None) or getattr(slide_obj, "id", "unknown")

        # Check model name once (LazySlide + timm)
        available = zs.models.list_models() + timm.list_models()
        if model_name not in available:
            raise ValueError(f"Model {model_name} not found in LazySlide/timm.")

        logger.info(
            "[LazySlideProcessor] feature_extraction: slide=%s,model=%s, device=%s, extra_params=%s",
            slide_id,
            model_name,
            params.get("device", "default"),
            {k: v for k, v in params.items() if k != "device"},
        )

        # This writes features into the WSI's SpatialData store
        zs.tl.feature_extraction(wsi=slide_obj, model=model_name, **params)

        # LazySlide convention: "{model_name}_tiles" as AnnData table key
        key = f"{model_name}_tiles"
        try:
            # Option A: documented helper
            adata = slide_obj.fetch.features_anndata(model_name)
        except Exception:
            # Option B: direct table access by key
            try:
                adata = slide_obj[key]
            except Exception as e:
                logger.warning(
                    "[LazySlideProcessor] Features extracted but could not read AnnData "
                    "for slide=%s, model=%s (key=%s): %s",
                    slide_id,
                    model_name,
                    key,
                    e,
                )
                return slide_obj
        
        n_tiles, n_dim = adata.n_obs, adata.n_vars
        logger.info(
            "[LazySlideProcessor] Feature extraction done: slide=%s, model=%s, "
            "tiles=%d, dim=%d",
            slide_id,
            model_name,
            n_tiles,
            n_dim,
        )

        return slide_obj

    
    def save_slide(self, slide_obj: zs.WSIData, save_path: str) -> None:
        # LazySlide typically saves sidecar files, but this enforces a save trigger if needed
        # or saves the python object. For features, they are saved during extraction.
        # We might implement saving the WSIData object metadata here.
        slide_obj.save(save_path)

    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return "LazySlideProcessor"