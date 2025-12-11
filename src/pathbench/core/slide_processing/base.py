
from abc import ABC, abstractmethod

class SlideProcessorBase(ABC):
    """Base class for slide processing backends."""
    
    @abstractmethod
    def load_slide(self, slide_path: str) -> Any:
        """Load a slide from the given path."""
        pass
    
    @abstractmethod
    def segment_tissue(self, slide_obj: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Segment tissue regions from the slide object."""
        pass
    
    @abstractmethod
    def extract_patches(self, slide_obj: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Extract patches from the slide object."""
        pass
    
    @abstractmethod
    def extract_features(self, slide_obj: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Extract features from the slide object."""
        pass
    
    @abstractmethod
    def save_features(self, slide_obj: Any, save_path: str) -> None:
        """Save the extracted features to the specified path. Features
        should be saved in a .pt format under the save_path.
        
        save_path should be specified as {experiment_dir}/{features_dir}/{tile_px}_{tile_mpp}_{model_name}/{slide_id}.pt
        """
        pass
    
    @abstractmethod
    def save_slide(self, slide_obj: Any, save_path: str) -> None:
        """Save the processed slide object to the specified path."""
        pass
    
    @abstractmethod
    def extract_cells(self, slide_obj: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Extract cells from the slide object."""
        pass
    
    @abstractmethod
    def inspect_slide(self, slide_obj: Any) -> None:
        """Inspect the slide object for debugging or analysis."""
        pass
    
    
    
    