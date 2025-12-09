# pathbench/utils/registry.py
from pathbench.core.base import RegistryBase
from typing import Callable, Dict, Sequence, TypeVar

T = TypeVar('T')

class Registry(RegistryBase):
    def __init__(self) -> None:
        self._f: Dict[str, Callable[..., T]] = {}

    def register(self, name: str):
        def deco(fn: Callable[..., T]) -> Callable[..., T]:
            if name in self._f:
                raise KeyError(f"Duplicate registration: {name}")
            self._f[name] = fn
            return fn
        return deco

    def get(self, name: str) -> Callable[..., T]:
        if name not in self._f:
            raise KeyError(f"Plugin '{name}' not found in registry")
        return self._f[name]

    def list_plugins(self) -> Sequence[str]:
        return list(self._f.keys())

    def is_available(self, key: str) -> bool:
        return key in self._f