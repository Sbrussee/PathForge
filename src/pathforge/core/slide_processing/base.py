from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd

from pathforge.core.datasets.wsi_dataset import WSI


class SlideProcessorBase(ABC):
    """Base class for slide processing backends."""

    @abstractmethod
    def load_wsi(self, wsi: WSI) -> None:
        """
        Load / open the backend-native slide object and store it on the WSI.
        """
        pass

    def close_wsi(self, wsi: WSI) -> None:
        """
        Close the backend-native slide object (if needed) and clear it from the WSI.
        """
        pass

    def get_base_mpp(self, wsi: WSI) -> float:
        """
        Return the level-0 microns-per-pixel (MPP) as a scalar.

        This should represent the slide's base resolution used to convert
        physical tile size to level-0 pixels.
        """
        raise NotImplementedError

    @abstractmethod
    def get_thumbnail(self, wsi: WSI, level: int = -1) -> Tuple[Any, float, float]:
        """
        Retrieve a thumbnail image for visualization and the downscale factors
        relative to level-0 (base resolution) coordinates.

        Args:
            wsi: Loaded WSI.
            level: Pyramid level to use for thumbnail retrieval. By convention,
                   level=-1 means the lowest-resolution level.

        Returns:
            (thumbnail_image, downscale_x, downscale_y)

            - thumbnail_image: backend-defined image object (e.g. PIL image / ndarray)
            - downscale_x: factor such that x_thumb = x_level0 / downscale_x
            - downscale_y: factor such that y_thumb = y_level0 / downscale_y
        """
        pass

    @abstractmethod
    def segment_tissue(self, wsi: WSI, config: Optional[Dict[str, Any]] = None) -> Any:
        """Segment tissue regions from the slide object."""
        pass

    @abstractmethod
    def extract_patches(self, wsi: WSI, tissues: List[np.ndarray], config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, str]:
        """
        Extract patches from the slide object.

        Returns:
            tiles_df: DataFrame with required columns: 'tile_id', 'x', 'y'
            tile_spec: JSON string (keys may vary; backend-defined)
        """
        pass

    @abstractmethod
    def validate_tile_spec(self, tile_spec: Optional[str], config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate whether the provided tile_spec (JSON string) is usable for this backend.

        IMPORTANT: This MUST be implemented by downstream backends. There is no default
        permissive implementation, because the policy relies on this to decide whether
        cached tiles can be reused safely.
        """
        pass

    @abstractmethod
    def extract_features(self, wsi: WSI, tiles: pd.DataFrame, tile_spec: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Extract features from the slide object.

        Args:
            tiles: DataFrame with required columns: 'tile_id', 'x', 'y'
            tile_spec: JSON string returned by extract_patches() (backend-defined)
        """
        pass

    @abstractmethod
    def read_patch_region(
        self,
        wsi: WSI,
        x: int,
        y: int,
        width: int,
        height: int,
        level: int,
    ) -> np.ndarray:
        """
        Read one patch region from the source slide as an RGB image array.

        Args:
            wsi: Loaded slide wrapper.
            x: Level-0 left coordinate.
            y: Level-0 top coordinate.
            width: Region width in pixels at ``level``.
            height: Region height in pixels at ``level``.
            level: Slide pyramid level used for the read.

        Returns:
            RGB ``uint8`` array with shape ``(H, W, 3)``.

        Example:
        .. code-block:: python

            patch = processor.read_patch_region(wsi, 0, 0, 256, 256, 0)

        """
        pass

    @abstractmethod
    def extract_cells(self, wsi: WSI, config: Optional[Dict[str, Any]] = None) -> Any:
        """Extract cells from the slide object."""
        pass

    @abstractmethod
    def inspect_slide(self, wsi: WSI) -> None:
        """Inspect the slide object for debugging or analysis."""
        pass
