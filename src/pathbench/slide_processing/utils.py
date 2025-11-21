
from ..config import Config
from typing import Optional, Union
import timm
import lazyslide as zs
from wsidata import open_wsi


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

