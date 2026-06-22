from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType


@dataclass(frozen=True)
class MILLabModules:
    """Lazy references to installed MIL-Lab modules."""

    builder: ModuleType


def _is_module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def is_mil_lab_available() -> bool:
    """Return whether MIL-Lab can be imported without importing it eagerly."""

    return _is_module_available("src.builder") or _is_module_available("builder")


def require_mil_lab(feature: str) -> None:
    """Raise an install hint when a MIL-Lab-only feature is requested."""

    if not is_mil_lab_available():
        raise RuntimeError(
            f"{feature} selected, but 'MIL-Lab' is not installed. "
            "Install MIL-Lab following its upstream README "
            "(git clone https://github.com/mahmoodlab/MIL-Lab.git && pip install -e .) "
            "or choose another MIL backend."
        )


@lru_cache(maxsize=1)
def load_mil_lab_modules() -> MILLabModules:
    """Import MIL-Lab builder modules lazily and return stable references."""

    require_mil_lab("MIL backend 'mil-lab'")
    import_errors: list[Exception] = []
    for module_name in ("src.builder", "builder"):
        try:
            return MILLabModules(builder=import_module(module_name))
        except Exception as exc:  # pragma: no cover - exercised via fallback path
            import_errors.append(exc)

    raise RuntimeError(
        "MIL-Lab is installed but its expected builder module could not be imported. "
        "Verify the installed MIL-Lab version exposes 'src.builder' or 'builder'."
    ) from import_errors[-1]
