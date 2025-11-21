# pathbench/utils/registry.py
from pathbench.core.base import RegistryBase

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
        ...

    def list_plugins(self) -> Sequence[str]:
        ...

    def is_available(self, key: str) -> bool:
        ...