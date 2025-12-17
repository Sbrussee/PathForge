from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path

@dataclass
class WSI:
    """
    Minimal backend-dispatching WSI wrapper.

    Usage:
        wsi = WSI(path, backend="lazyslide").load()
        native = wsi.obj
    """
    path: str | Path
    backend: str

    _obj: Optional[Any] = None

    def load(self) -> "WSI":
        """
        Load the slide using load_{backend}() and cache the native slide object.
        """
        if self._obj is not None:
            return self

        fn_name = f"load_{self.backend}"
        fn = getattr(self, fn_name, None)
        if fn is None:
            raise ValueError(
                f"Unsupported backend '{self.backend}'. "
                f"Missing method: {fn_name}()"
            )

        self._obj = fn(str(self.path))
        return self

    @property
    def obj(self) -> Any:
        """
        Return the loaded backend-native slide object.

        Raises:
            RuntimeError: if load() was not called yet.
        """
        if self._obj is None:
            raise RuntimeError("WSI not loaded. Call .load() first.")
        return self._obj

    # ------------------------- Backend-specific loaders -------------------------

    def load_lazyslide(self, path: str) -> Any:
        from wsidata import open_wsi
        return open_wsi(path)