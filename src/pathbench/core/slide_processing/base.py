
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd

from pathbench.core.datasets.wsi import WSI

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
    def extract_cells(self, wsi: WSI, config: Optional[Dict[str, Any]] = None) -> Any:
        """Extract cells from the slide object."""
        pass
    
    @abstractmethod
    def inspect_slide(self, wsi: WSI) -> None:
        """Inspect the slide object for debugging or analysis."""
        pass
    
    
    
    