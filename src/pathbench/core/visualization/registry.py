from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pathbench.core.experiments.base import Experiment
    from pathbench.core.visualization.base import TaskVisualizationAdapterBase


_TASK_VISUALIZATION_ADAPTER_REGISTRY: dict[str, type["TaskVisualizationAdapterBase"]] = {}


def _normalize_name(name: str, *, kind: str) -> str:
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError(f"{kind} name must be a non-empty string.")
    return normalized_name


def task_visualization_adapter(
    task_name: str,
) -> Callable[[type["TaskVisualizationAdapterBase"]], type["TaskVisualizationAdapterBase"]]:
    """Register one task-specific visualization adapter."""

    normalized_task_name = _normalize_name(task_name, kind="Task visualization adapter")

    def decorator(
        adapter_cls: type["TaskVisualizationAdapterBase"],
    ) -> type["TaskVisualizationAdapterBase"]:
        if normalized_task_name in _TASK_VISUALIZATION_ADAPTER_REGISTRY:
            raise ValueError(
                f"Task visualization adapter '{normalized_task_name}' is already registered."
            )
        _TASK_VISUALIZATION_ADAPTER_REGISTRY[normalized_task_name] = adapter_cls
        return adapter_cls

    return decorator


def build_task_visualization_adapter(
    name: str,
    experiment: "Experiment",
) -> "TaskVisualizationAdapterBase":
    """Instantiate one registered task visualization adapter."""
    normalized_name = _normalize_name(name, kind="Task visualization adapter")
    if normalized_name not in _TASK_VISUALIZATION_ADAPTER_REGISTRY:
        available = ", ".join(sorted(_TASK_VISUALIZATION_ADAPTER_REGISTRY)) or "none"
        raise ValueError(
            f"Task visualization adapter '{normalized_name}' is not registered. "
            f"Available adapters: {available}"
        )
    return _TASK_VISUALIZATION_ADAPTER_REGISTRY[normalized_name](experiment)


def import_task_visualization_adapter_modules(
    package_name: str = "pathbench.core.visualization.tasks",
) -> None:
    """Import all task-visualization adapter modules so registration side effects run."""

    package = import_module(package_name)
    if not hasattr(package, "__path__"):
        raise ValueError(f"Package '{package_name}' does not have a __path__ attribute.")

    for module_info in iter_modules(package.__path__, package_name + "."):
        module_name = module_info.name.rsplit(".", 1)[-1]
        if module_name.startswith("_"):
            continue
        import_module(module_info.name)

