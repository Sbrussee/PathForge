
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
    def 