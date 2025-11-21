from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Sequence

class RegistryBase(ABC):
    """
    Plugin registry base class for managing different types of plugins.
    """
    
    @abstractmethod
    def register(self, key: str, obj: Any) -> None:
        """Register a plugin with a given key."""
        pass
    
    @abstractmethod
    def get(self, key: str) -> Any:
        """Retrieve a plugin by its key."""
        pass
    
    @abstractmethod
    def list_plugins(self) -> Sequence[str]:
        """List all registered plugin keys."""
        pass
    
    @abstractmethod
    def is_available(self, key: str) -> bool:
        """Check if a plugin is available."""
        pass
    
@dataclass
class CoreRegistries:
    """
    Container for core plugin registries.
    """
    datasets: RegistryBase
    models: RegistryBase
    losses: RegistryBase
    tasks: RegistryBase
    explainers: RegistryBase
    feature_extractors: RegistryBase
    normalizers: RegistryBase
    augmentation_methods: RegistryBase
    