from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


class ConfigBase(ABC):
    """
    Abstract base for all high-level configuration objects.

    Provides a standard API for converting to/from simple Python dicts
    and loading/saving YAML files.
    """

    # ---- dict <-> object -------------------------------------------------

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfigBase":
        """Create config object from a (possibly nested) dict."""
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert config object to a (YAML-serializable) dict."""
        ...

    # ---- YAML helpers ----------------------------------------------------

    @classmethod
    def load_yaml(cls, path: str | Path) -> "ConfigBase":
        path = Path(path)
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)