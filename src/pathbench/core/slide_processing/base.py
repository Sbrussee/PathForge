from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

class SlideProcessorBase(ABC):
    """
    Abstract class for Slide Processing backend interface.
    Should support various backends (e.g., OpenSlide, LazySlide, etc.)
    """
    
    @abstractmethod
    def load_slide(self, slide_path: str) -> Any:
        """Load a slide from the given path."""
        ...
        
    @abstractmethod
    def segment_tissue(self, slide_obj: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Segment tissue regions from the slide object."""
        ...
        
    @abstractmethod
    def extract_patches(self, slide_obj: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Extract patches from the slide object."""
        ...
        
    @abstractmethod
    def extract_features(self, slide_obj: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """Extract features from the slide object."""
        ...
    
    @abstractmethod
    def save_slide(self, slide_obj: Any, save_path: str) -> None:
        """Save the processed slide object to the specified path."""
        ...
    
    @abstractmethod ##TODO: Why is this abstractmethod?
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    @abstractmethod ##TODO: Why is this abstractmethod?
    def __str__(self) -> str:
        return self.__repr__()
