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
    
    
    def check_for_existing_features(
        self,
        tile_config: Dict[str, Any],
        feat_config: Dict[str, Any],
        experiment_dir: str,
        features_dir: str,
        slide_id: str,
    ) -> bool:
        """
        Check if features for the given slide and configuration already exist.

        If they do, log and return True to indicate skipping extraction.
        """
        tile_px, tile_mpp, model_name = tile_config["tile_px"], tile_config["tile_mpp"], feat_config["model"]
        features_out_path = os.path.join(experiment_dir, features_dir,
                                         f"{tile_px}_{tile_mpp}_{model_name}",
                                         f"{slide_id}.pt")
        if os.path.exists(features_out_path):
            logger.info(f"Features already exist at {features_out_path}, skipping extraction.")
        return True
        
    def save_features(
        self,
        slide_obj: zs.WSIData,
        tile_config: Dict[str, Any],
        feat_config: Dict[str, Any],
        features_dir: str, experiment_dir: str,
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
        
        
        tile_px, tile_mpp, model_name = tile_config["tile_px"], tile_config["tile_mpp"], feat_config["model"]
        slide_id = getattr(slide_obj, "name", None) or getattr(slide_obj, "id", "unknown")
        features_out_path = os.path.join(experiment_dir, features_dir,
                                         f"{tile_px}_{tile_mpp}_{model_name}",
                                         f"{slide_id}.pt")
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


    def extract_cells(self, slide_obj: zs.WSIData, config: Dict[str, Any]) -> Any:
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
        zs.pp.find_tissues(slide_obj)
        zs.pp.tile_tissues(slide_obj, tile_size=params.get("tile_size", 512), overlap=params.get("overlap", 0.2), mpp=params.get("mpp", 0.5))
        logger.info("[LazySlide] Tiled tissues for cell segmentation.")
        
        #Cell type segmentation
        zs.seg.cells(slide_obj, model=cell_seg_model, transform=params.get("transform", None), num_workers=params.get("num_workers", 0),
                     batch_size=params.get("batch_size", 4), key_added=cells_key)
        logger.info("[LazySlide] Cell segmentation done, added to %s number of cells: %d", cells_key, len(slide_obj[cells_key]))
        
        #Assert cells were found
        assert len(slide_obj[cells_key]) > 0, "[LazySlide] No cells were segmented, cannot proceed to cell type classification."
        
        if cell_type_classification:
            # Cell type classification
            zs.seg.cell_types(slide_obj, model=cell_class_model, magnification=params.get("mpp", None), transform=params.get("transform", None),
                              batch_size=params.get("batch_size", 4), num_workers=params.get("num_workers", 0), key_added=cell_types_key)
            logger.info("[LazySlide] Cell type classification done, added to %s number of classified cells: %d", cell_types_key, len(slide_obj[cell_types_key]))
            
            #Assert cell types were classified
            assert len(slide_obj[cell_types_key]) > 0, "[LazySlide] No cell types were classified, something went wrong."
        
    def save_slide(self, slide_obj: zs.WSIData, save_path: str) -> None:
        # LazySlide typically saves sidecar files, but this enforces a save trigger if needed
        # or saves the python object. For features, they are saved during extraction.
        # We might implement saving the WSIData object metadata here.
        
        #This will write a .zarr file to the save_path
        slide_obj.write(save_path)
        
    def inspect_slide(self, slide_obj: zs.WSIData) -> None:
        """
        Print a summary of the slide object for debugging.
        """
        logger.info("[LazySlide] Inspecting slide object:")
        logger.info("  ID: %s", getattr(slide_obj, "id", "unknown"))
        logger.info("  Name: %s", getattr(slide_obj, "name", "unknown"))
        logger.info("  Properties: %s", getattr(slide_obj, "properties", {}))
        logger.info("  Available tables: %s", list(slide_obj.keys()))
        for key in slide_obj.keys():
            table = slide_obj[key]
            try:
                n_entries = len(table)
                logger.info("    Table '%s': %d entries", key, n_entries)
            except Exception:
                logger.info("    Table '%s': (unable to determine length)", key)
                
        

    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return "LazySlideProcessor"