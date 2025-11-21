from __future__ import annotations
from typing import Any, Dict, Optional
import lazyslide as zs
from wsidata import open_wsi
import timm

from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.utils.registries import SLIDE_PROCESSORS

@SLIDE_PROCESSORS.register("lazyslide")
class LazySlideProcessor(SlideProcessorBase):
    """
    Concrete implementation of SlideProcessorBase using the LazySlide backend.
    """
    
    def load_slide(self, slide_path: str) -> zs.WSIData:
        return open_wsi(slide_path)
    
    def segment_tissue(self, slide_obj: zs.WSIData, config: Dict[str, Any]) -> Any:
        # Extract specific parameters from the generic config dict
        # Assuming config passed here is the specific slide_processing.tissue_segmentation dict
        method = config.get("method", "otsu")
        params = config.get("params", {})
        
        if method == "otsu":
            zs.pp.find_tissues(wsi=slide_obj, **params)
        else:
            # Assume method is a model name compatible with zs.seg.tissue
            zs.seg.tissue(wsi=slide_obj, model=method, **params)
        return slide_obj
    
    def extract_patches(self, slide_obj: zs.WSIData, config: Dict[str, Any]) -> Any:
        params = config.get("params", {})
        zs.pp.tile_tissues(wsi=slide_obj, **params)
        return slide_obj
    
    def extract_features(self, slide_obj: zs.WSIData, config: Dict[str, Any]) -> Any:
        model_name = config.get("model", "resnet50")
        params = config.get("params", {})
        
        # Validation (optional, but good practice)
        available = zs.models.list_models() + timm.list_models()
        if model_name not in available:
            raise ValueError(f"Model {model_name} not found in LazySlide/timm.")
            
        zs.tl.extract_features(wsi=slide_obj, model=model_name, **params)
        return slide_obj
    
    def save_slide(self, slide_obj: zs.WSIData, save_path: str) -> None:
        # LazySlide typically saves sidecar files, but this enforces a save trigger if needed
        # or saves the python object. For features, they are saved during extraction.
        # We might implement saving the WSIData object metadata here.
        slide_obj.save(save_path)

    def __repr__(self) -> str:
        return "LazySlideProcessor"