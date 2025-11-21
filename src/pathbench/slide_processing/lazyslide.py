# pathbench/slide_processing/lazyslide.py
from ..config import Config
from typing import Optional, Union
from pathbench.slide_processing.base import SlideProcessorBase
import timm
import lazyslide as zs
from wsidata import open_wsi

#This code implements various functions for processing Whole Slide Images (WSI) using the LazySlide library.

class LazySlideProcessor(SlideProcessorBase):
    """
    Slide Processor implementation using LazySlide backend.
    """
    
    def load_slide(self, slide_path: str) -> zs.WSIData:
        """Load a slide from the given path."""
        return open_wsi(slide_path)
    
    def segment_tissue(self, slide_obj: zs.WSIData, config: Optional[Dict[str, Any]] = None) -> None:
        """Segment tissue regions from the slide object."""
        segment_tissue(config, slide_obj, method)
    
    def extract_patches(self, slide_obj: zs.WSIData, config: Optional[Dict[str, Any]] = None) -> None:
        """Extract patches from the slide object."""
        extract_tiles(config, slide_obj)
    
    def extract_features(self, slide_obj: zs.WSIData, config: Optional[Dict[str, Any]] = None) -> None:
        """Extract features from the slide object."""
        extract_features(config, slide_obj, model)
    
    def save_slide(self, slide_obj: zs.WSIData, save_path: str) -> None:
        """Save the processed slide object to the specified path."""
        slide_obj.save(save_path)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    def __str__(self) -> str:
        return self.__repr__()



def segment_tissue(config: Config, slide: zs.WSIData, method: str = "otsu") -> None:
    """
    Segment tissue regions in a Whole Slide Image (WSI).

    This function identifies tissue areas within a WSI using either the Otsu thresholding
    method or a specified deep learning segmentation model.

    Args:
        config (Config): Configuration object containing slide processing parameters.
        slide (zs.WSIData): The WSI object loaded via LazySlide.
        method (str, optional): Segmentation method. Defaults to "otsu".
            - "otsu": Uses traditional Otsu thresholding via `zs.pp.find_tissues`.
            - <model_name>: Uses a pre-trained tissue segmentation model via `zs.seg.tissue`.

    Returns:
        None
    """
    if method == "otsu":
        zs.pp.find_tissues(wsi=slide, **config.slide_processing.tissue_segmentation)
    else:
        zs.seg.tissue(wsi=slide, model=method, **config.slide_processing.tissue_segmentation)

def extract_tiles(config: Config, slide: zs.WSIData) -> None:
    """
    Extract tiles (patches) from tissue regions of a Whole Slide Image.

    Args:
        config (Config): Configuration object containing tile extraction parameters.
        slide (zs.WSIData): The WSI object with segmented tissue regions.

    Returns:
        None
    """
    zs.pp.tile_tissues(wsi=slide, **config.slide_processing.tile_extraction)

def extract_features(config: Config, slide: zs.WSIData, model: str) -> None:
    """
    Extract deep learning features from tiles of a Whole Slide Image.

    The function supports both LazySlide and timm model libraries. It ensures the specified
    model exists in either before feature extraction.

    Args:
        config (Config): Configuration object containing feature extraction parameters.
        slide (zs.WSIData): The WSI object with tiled regions.
        model (str): Model name used for feature extraction.
            Must be available in `zs.models.list_models()` or `timm.list_models()`.

    Raises:
        AssertionError: If the provided model is not found in LazySlide or timm model zoo.

    Returns:
        None
    """
    assert model in zs.models.list_models() or model in timm.list_models(), \
        f"Model '{model}' not found in LazySlide or timm model zoo."

    zs.tl.extract_features(wsi=slide, model=model, **config.slide_processing.feature_extraction)

def aggregate_features(config: Config, slide: zs.WSIData, 
                       encoder: str = "mean", by: Optional[str] = None) -> None:
    """
    Aggregate extracted features from tiles to obtain slide-level representations.

    Args:
        config (Config): Configuration object containing feature aggregation parameters.
        slide (zs.WSIData): The WSI object with extracted features.
        encoder (str | None, optional): Aggregation encoder or method name. Defaults to "mean".
        by (str | None, optional): Optional grouping key for aggregation. Defaults to None, corresponding to all tiles in the slide.

    Returns:
        None
    """
    zs.tl.feature_aggregation(wsi=slide, encoder=encoder, by=by, **config.slide_processing.feature_aggregation)



def load_slide(path: str) -> zs.WSIData:
    """
    Load a Whole Slide Image (WSI) using LazySlide.

    Args:
        path (str): Path to the WSI file.

    Returns:
        zs.WSIData: Loaded Whole Slide Image object.
    """
    return open_wsi(path)

