from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType


@dataclass(frozen=True)
class TorchMILModules:
    """Lazy references to installed TorchMIL modules.

    Attributes:
        root: Imported ``torchmil`` package module.
        models: Imported ``torchmil.models`` module. Expected to expose model classes
            such as ``ABMIL``.
        data: Imported ``torchmil.data`` module. Expected to expose ``collate_fn``
            accepting a list of bag dictionaries and returning a padded batch with
            ``X`` shaped ``[B, N_max, D]`` and ``mask`` shaped ``[B, N_max]``.
        datasets: Imported ``torchmil.datasets`` module.

    Example:
        ```python
        from pathbench.utils.optional.torchmil import load_torchmil_modules

        modules = load_torchmil_modules()
        batch = modules.data.collate_fn([{"X": x0, "Y": y0}, {"X": x1, "Y": y1}])
        model_cls = getattr(modules.models, "ABMIL")
        ```

    Raises:
        RuntimeError: If TorchMIL is not installed.
    """

    root: ModuleType
    models: ModuleType
    data: ModuleType
    datasets: ModuleType


def _is_module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def is_torchmil_available() -> bool:
    """Return whether ``torchmil`` can be imported without importing it eagerly."""

    return _is_module_available("torchmil")


def is_torchmetrics_available() -> bool:
    """Return whether ``torchmetrics`` can be imported without importing it eagerly."""

    return _is_module_available("torchmetrics")


def is_torchsurv_available() -> bool:
    """Return whether ``torchsurv`` can be imported without importing it eagerly."""

    return _is_module_available("torchsurv")


def require_torchmil(feature: str) -> None:
    """Raise an install hint when a TorchMIL-only feature is requested.

    Args:
        feature: Human-readable feature name, for example ``"MIL backend
            'torchmil'"`` or ``"TorchMIL heatmap explainer"``.

    Raises:
        RuntimeError: If ``torchmil`` is not installed.
    """

    if not is_torchmil_available():
        raise RuntimeError(
            f"{feature} selected, but 'torchmil' is not installed. "
            "Install torchmil or set mil.backend='native'."
        )


def require_torchmetrics(feature: str = "Classification metrics backend") -> None:
    """Raise an install hint when TorchMetrics-backed classification metrics are used."""

    if not is_torchmetrics_available():
        raise RuntimeError(
            f"{feature} requires 'torchmetrics'. "
            "Install torchmetrics or choose another classification metrics backend."
        )


def require_torchsurv(feature: str = "Continuous survival backend") -> None:
    """Raise an install hint when TorchSurv-backed survival functionality is used."""

    if not is_torchsurv_available():
        raise RuntimeError(
            f"{feature} requires 'torchsurv'. "
            "Install torchsurv or choose another survival backend."
        )


@lru_cache(maxsize=1)
def load_torchmil_modules() -> TorchMILModules:
    """Import TorchMIL modules lazily and return stable module references.

    Returns:
        TorchMILModules: Imported root, models, data, and datasets modules.

    Raises:
        RuntimeError: If ``torchmil`` or an expected TorchMIL submodule is missing.
    """

    require_torchmil("MIL backend 'torchmil'")
    try:
        root = import_module("torchmil")
        models = import_module("torchmil.models")
        data = import_module("torchmil.data")
        datasets = import_module("torchmil.datasets")
    except Exception as exc:
        raise RuntimeError(
            "TorchMIL is installed but its expected modules could not be imported. "
            "Verify the installed torchmil version exposes torchmil.models, "
            "torchmil.data, and torchmil.datasets."
        ) from exc
    return TorchMILModules(root=root, models=models, data=data, datasets=datasets)
