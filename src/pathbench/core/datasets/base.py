# src/core/datasets/base.py
from __future__ import annotations
from typing import Protocol, Iterable, Iterator, Sequence, Tuple, Any

class DatasetBase(ABC):
    """
    Generic dataset base class. Can represent tile-level, slide-level or other datasets.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def num_samples(self) -> int:
        ...

    def __len__(self) -> int:
        # enforce num_samples as the “truth”
        return self.num_samples

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        ...
    
class BagDatasetBase(DatasetBase):
    """
    Dataset base class for Multiple Instance Learning (MIL) bags.
    Each item is a bag of instances with an associated label.
    """
    
    @property
    @abstractmethod
    def num_bags(self) -> int:
        ...

    @property
    def num_samples(self) -> int:  # type: ignore[override]
        # alias: for bags, sample == bag
        return self.num_bags
    
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Iterable[Any], Any]:
        """Retrieve a bag of instances and its label by index.
        Should return a tuple (instances, label)."""
        pass
    
    
    
class TileDatasetBase(DatasetBase):
    """
    Dataset base class for tile-level datasets.
    Each item is a single tile with an associated label.
    """

    @property
    @abstractmethod
    def num_tiles(self) -> int:
        ...
         
    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Retrieve a tile and its label by index.
        Should return a tuple (tile, label)."""
        pass
    
