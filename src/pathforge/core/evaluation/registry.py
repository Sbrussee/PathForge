from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pkgutil import iter_modules
import re
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Pattern

from pathforge.core.evaluation.types import MetricRequest

if TYPE_CHECKING:
    from pathforge.core.evaluation.base import TaskEvaluationAdapterBase
    from pathforge.core.experiments.base import Experiment


ParseMetricFn = Callable[[str], MetricRequest | None]
MetricComputeFn = Callable[..., dict[str, Any]]

# ---------------------------------------------------------------------------
# Task evaluation adapter registry
# ---------------------------------------------------------------------------

_TASK_EVALUATION_ADAPTER_REGISTRY: dict[str, type["TaskEvaluationAdapterBase"]] = {}


@dataclass(frozen=True, slots=True)
class EvaluationMetricSpec:
    """Registered evaluation-metric specification."""

    canonical_name: str
    supported_tasks: frozenset[str]
    parser: ParseMetricFn
    compute_fn: MetricComputeFn


_METRIC_SPECS: list[EvaluationMetricSpec] = []


def _normalize_name(name: str, *, kind: str) -> str:
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError(f"{kind} name must be a non-empty string.")
    return normalized_name


def evaluation_task_adapter(
    task_name: str,
) -> Callable[[type["TaskEvaluationAdapterBase"]], type["TaskEvaluationAdapterBase"]]:
    """Register one task-specific evaluation adapter."""

    normalized_task_name = _normalize_name(task_name, kind="Task evaluation adapter")

    def decorator(
        adapter_cls: type["TaskEvaluationAdapterBase"],
    ) -> type["TaskEvaluationAdapterBase"]:
        if normalized_task_name in _TASK_EVALUATION_ADAPTER_REGISTRY:
            raise ValueError(
                f"Task evaluation adapter '{normalized_task_name}' is already registered."
            )
        _TASK_EVALUATION_ADAPTER_REGISTRY[normalized_task_name] = adapter_cls
        return adapter_cls

    return decorator


def get_task_evaluation_adapter(name: str) -> type["TaskEvaluationAdapterBase"]:
    """Return one registered task evaluation adapter."""

    normalized_name = _normalize_name(name, kind="Task evaluation adapter")
    if normalized_name not in _TASK_EVALUATION_ADAPTER_REGISTRY:
        available = ", ".join(list_task_evaluation_adapters()) or "none"
        raise ValueError(
            f"Task evaluation adapter '{normalized_name}' is not registered. "
            f"Available adapters: {available}"
        )
    return _TASK_EVALUATION_ADAPTER_REGISTRY[normalized_name]


def build_task_evaluation_adapter(
    name: str,
    experiment: "Experiment",
) -> "TaskEvaluationAdapterBase":
    """Instantiate one registered task evaluation adapter."""

    adapter_cls = get_task_evaluation_adapter(name)
    return adapter_cls(experiment)


def list_task_evaluation_adapters() -> list[str]:
    """Return registered task evaluation adapters in sorted order."""

    return sorted(_TASK_EVALUATION_ADAPTER_REGISTRY)


def _iter_non_private_modules(package: ModuleType, prefix: str) -> list[str]:
    """List direct non-private modules for one package."""

    if not hasattr(package, "__path__"):
        raise ValueError(f"Package '{package.__name__}' does not have a __path__ attribute.")

    module_names: list[str] = []
    for module_info in iter_modules(package.__path__, prefix):
        module_name = module_info.name.rsplit(".", 1)[-1]
        if module_name.startswith("_"):
            continue
        module_names.append(module_info.name)
    return module_names


def import_task_evaluation_adapter_modules(
    package_name: str = "pathforge.core.evaluation",
) -> None:
    """Import all task-evaluation adapter modules so registration side effects run."""

    package = import_module(package_name)
    if not hasattr(package, "__path__"):
        raise ValueError(f"Package '{package_name}' does not have a __path__ attribute.")

    tasks_package_name = package_name + ".tasks"
    try:
        tasks_package = import_module(tasks_package_name)
    except ModuleNotFoundError:
        tasks_package = None
    if tasks_package is not None:
        for module_name in _iter_non_private_modules(
            tasks_package,
            tasks_package_name + ".",
        ):
            import_module(module_name)

    for module_info in iter_modules(package.__path__, package_name + "."):
        module_name = module_info.name.rsplit(".", 1)[-1]
        if module_name.startswith("_") or not module_info.ispkg:
            continue
        if module_name in {"tasks", "metrics"}:
            continue
        adapter_module_name = module_info.name + ".adapter"
        try:
            import_module(adapter_module_name)
        except ModuleNotFoundError as exc:
            if exc.name != adapter_module_name:
                raise


# ---------------------------------------------------------------------------
# Evaluation metric registry
# ---------------------------------------------------------------------------


def evaluation_metric(
    name: str,
    *,
    tasks: tuple[str, ...] | list[str],
    pattern: str | Pattern[str] | None = None,
    param_builder: Callable[[re.Match[str]], dict[str, Any]] | None = None,
) -> Callable[[MetricComputeFn], MetricComputeFn]:
    """
    Register one evaluation metric.

    Inputs:
    - `name`: canonical metric name or family name.
    - `tasks`: tasks supported by this metric.
    - `pattern`: optional regex used to parse parameterized metric requests.
    - `param_builder`: optional builder that converts a regex match into metric params.

    Returns:
    - Decorator that registers the metric compute function.
    """

    canonical_name = _normalize_name(name, kind="Metric")
    supported_tasks = frozenset(
        _normalize_name(task_name, kind="Task") for task_name in tasks
    )

    if pattern is None:
        if param_builder is not None:
            raise ValueError(
                "evaluation_metric received param_builder without pattern."
            )

        def parser(raw_name: str) -> MetricRequest | None:
            normalized_raw_name = _normalize_name(raw_name, kind="Metric request")
            if normalized_raw_name != canonical_name:
                return None
            return MetricRequest(
                raw_name=normalized_raw_name,
                canonical_name=canonical_name,
                params={},
            )

    else:
        if param_builder is None:
            raise ValueError(
                "evaluation_metric received pattern without param_builder."
            )

        compiled_pattern = (
            re.compile(pattern)
            if isinstance(pattern, str)
            else pattern
        )

        def parser(raw_name: str) -> MetricRequest | None:
            normalized_raw_name = _normalize_name(raw_name, kind="Metric request")
            match = compiled_pattern.fullmatch(normalized_raw_name)
            if match is None:
                return None
            return MetricRequest(
                raw_name=normalized_raw_name,
                canonical_name=canonical_name,
                params=param_builder(match),
            )

    return _register_metric_spec(
        canonical_name=canonical_name,
        supported_tasks=supported_tasks,
        parser=parser,
    )


def _register_metric_spec(
    *,
    canonical_name: str,
    supported_tasks: frozenset[str],
    parser: ParseMetricFn,
) -> Callable[[MetricComputeFn], MetricComputeFn]:
    def decorator(metric_fn: MetricComputeFn) -> MetricComputeFn:
        for spec in _METRIC_SPECS:
            if spec.canonical_name == canonical_name:
                raise ValueError(
                    f"Evaluation metric '{canonical_name}' is already registered."
                )
        _METRIC_SPECS.append(
            EvaluationMetricSpec(
                canonical_name=canonical_name,
                supported_tasks=supported_tasks,
                parser=parser,
                compute_fn=metric_fn,
            )
        )
        return metric_fn

    return decorator


def import_evaluation_metric_modules(
    package_name: str = "pathforge.core.evaluation",
) -> None:
    """Import all evaluation-metric modules so registration side effects run."""

    package = import_module(package_name)
    if not hasattr(package, "__path__"):
        raise ValueError(f"Package '{package_name}' does not have a __path__ attribute.")

    metrics_package_name = package_name + ".metrics"
    try:
        metrics_package = import_module(metrics_package_name)
    except ModuleNotFoundError:
        metrics_package = None
    if metrics_package is not None:
        for module_name in _iter_non_private_modules(
            metrics_package,
            metrics_package_name + ".",
        ):
            import_module(module_name)

    for module_info in iter_modules(package.__path__, package_name + "."):
        module_name = module_info.name.rsplit(".", 1)[-1]
        if module_name.startswith("_") or not module_info.ispkg:
            continue
        if module_name in {"tasks", "metrics"}:
            continue
        task_metrics_package_name = module_info.name + ".metrics"
        try:
            task_metrics_package = import_module(task_metrics_package_name)
        except ModuleNotFoundError as exc:
            if exc.name != task_metrics_package_name:
                raise
            continue

        for submodule_name in _iter_non_private_modules(
            task_metrics_package,
            task_metrics_package_name + ".",
        ):
            import_module(submodule_name)


def resolve_metric_request(
    *,
    task_name: str,
    raw_name: str,
) -> tuple[EvaluationMetricSpec, MetricRequest]:
    """Resolve one raw metric request to a registered metric specification."""

    normalized_task_name = _normalize_name(task_name, kind="Task")
    normalized_raw_name = _normalize_name(raw_name, kind="Metric request")

    matching_specs = [
        spec
        for spec in _METRIC_SPECS
        if normalized_task_name in spec.supported_tasks
    ]

    for spec in matching_specs:
        request = spec.parser(normalized_raw_name)
        if request is not None:
            return spec, request

    available = ", ".join(
        sorted(
            spec.canonical_name
            for spec in matching_specs
        )
    ) or "none"
    raise ValueError(
        f"Evaluation metric '{normalized_raw_name}' is not registered for task "
        f"'{normalized_task_name}'. Available metrics: {available}"
    )


def list_evaluation_metrics(task_name: str | None = None) -> list[str]:
    """Return canonical registered metric names, optionally filtered by task."""

    if task_name is None:
        return sorted(spec.canonical_name for spec in _METRIC_SPECS)

    normalized_task_name = _normalize_name(task_name, kind="Task")
    return sorted(
        spec.canonical_name
        for spec in _METRIC_SPECS
        if normalized_task_name in spec.supported_tasks
    )
