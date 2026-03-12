from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from pathbench.benchmarking.tasks.base import TaskBase
    from pathbench.core.experiments.base import Experiment


_TASK_REGISTRY: dict[str, type["TaskBase"]] = {}


def _normalize_task_name(name: str) -> str:
    task_name = str(name).strip().lower()
    if not task_name:
        raise ValueError("Task name must be a non-empty string.")
    return task_name


def register_task(name: str) -> Callable[[type["TaskBase"]], type["TaskBase"]]:
    """
    Decorator to register a benchmarking task class.

    Example:
        @register_task("classification")
        class ClassificationTask(TaskBase):
            ...
    """
    normalized_name = _normalize_task_name(name)

    def decorator(task_cls: type["TaskBase"]) -> type["TaskBase"]:
        if normalized_name in _TASK_REGISTRY:
            raise ValueError(
                f"Task '{normalized_name}' is already registered with "
                f"class {_TASK_REGISTRY[normalized_name].__name__}."
            )

        _TASK_REGISTRY[normalized_name] = task_cls
        return task_cls

    return decorator

def build_task(name: str, experiment: "Experiment") -> "TaskBase":
    """
    Resolve and instantiate a registered task.

    Args:
        name: Task name.
        experiment: Experiment instance passed to the task constructor.

    Returns:
        Instantiated task object.
    """
    task_cls = get_task(name)
    return task_cls(experiment)


def get_task(name: str) -> type["TaskBase"]:
    """
    Get a registered task class by name.
    """
    normalized_name = _normalize_task_name(name)

    if normalized_name not in _TASK_REGISTRY:
        available = ", ".join(list_tasks()) or "none"
        raise ValueError(
            f"Task '{normalized_name}' is not registered. "
            f"Available tasks: {available}"
        )

    return _TASK_REGISTRY[normalized_name]


def is_task_available(name: str) -> bool:
    """
    Check whether a task is registered.
    """
    normalized_name = _normalize_task_name(name)
    return normalized_name in _TASK_REGISTRY


def list_tasks() -> list[str]:
    """
    Return all registered task names in sorted order.
    """
    return sorted(_TASK_REGISTRY.keys())


def import_task_modules(package_name: str = "pathbench.benchmarking.tasks") -> None:
    """
    Import all modules inside the benchmarking.tasks package so that
    decorator-based registration is executed.

    Call this once before get_task(...).
    """
    package = import_module(package_name)

    if not hasattr(package, "__path__"):
        raise ValueError(f"Package '{package_name}' does not have a __path__ attribute.")

    for module_info in iter_modules(package.__path__, package_name + "."):
        module_name = module_info.name.rsplit(".", 1)[-1]

        if module_name.startswith("_"):
            continue

        import_module(module_info.name)