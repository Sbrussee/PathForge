from __future__ import annotations

from importlib import import_module
import re
from typing import Any, Callable, Sequence

from pathbench.slide_retrieval.validation.types import ValidationMetricRequest

ValidationMetricFn = Callable[..., dict[str, Any]]

_VALIDATION_METRIC_REGISTRY: dict[str, ValidationMetricFn] = {}
_METRIC_NAME_PATTERN = re.compile(
    r"^(?P<metric>[a-z][a-z0-9_]*)_at_(?P<k>[1-9]\d*)$"
)


def register_validation_metric(name: str) -> Callable[[ValidationMetricFn], ValidationMetricFn]:
    """
    Register one slide-retrieval validation metric function.

    Inputs:
    - `name`: `str` canonical registry key, expected shape `<metric>_at_k`.

    Returns:
    - Decorator that stores the metric function under the normalized key.

    Example:
        ```python
        @register_validation_metric("hit_at_k")
        def compute_hit_at_k(...):
            ...
        ```
    """

    normalized_name = _normalize_metric_registry_key(name)

    def decorator(metric_fn: ValidationMetricFn) -> ValidationMetricFn:
        if normalized_name in _VALIDATION_METRIC_REGISTRY:
            raise ValueError(
                f"Validation metric '{normalized_name}' is already registered."
            )
        _VALIDATION_METRIC_REGISTRY[normalized_name] = metric_fn
        return metric_fn

    return decorator


def import_validation_metric_modules(
    package_name: str = "pathbench.slide_retrieval.validation",
) -> None:
    """
    Import the canonical slide-retrieval validation package.

    Inputs:
    - `package_name`: `str` package root that exposes the `metrics` module.

    Returns:
    - `None`. Import side effects register validation metrics.
    """

    package = import_module(package_name)
    if not hasattr(package, "__path__"):
        raise ValueError(f"Package '{package_name}' does not have a __path__ attribute.")
    import_module(f"{package_name}.metrics")


def parse_validation_metric_name(name: str) -> ValidationMetricRequest:
    """
    Parse one validation request from `experiment.evaluation`.

    Inputs:
    - `name`: `str` metric request expected to match `<metric>_at_<k>`.

    Returns:
    - `ValidationMetricRequest` with the parsed metric family and integer `k`.

    Example:
        ```python
        request = parse_validation_metric_name("hit_at_5")
        ```
    """

    normalized_name = str(name).strip().lower()
    match = _METRIC_NAME_PATTERN.fullmatch(normalized_name)
    if match is None:
        raise ValueError(
            f"Invalid slide-retrieval evaluation metric '{name}'. Expected format "
            "'<metric>_at_<k>', for example 'hit_at_5'."
        )

    metric_name = match.group("metric")
    k = int(match.group("k"))
    return ValidationMetricRequest(
        raw_name=normalized_name,
        metric_name=metric_name,
        registry_key=f"{metric_name}_at_k",
        k=k,
    )


def get_validation_metric(name: str) -> ValidationMetricFn:
    """
    Resolve one registered slide-retrieval validation metric.

    Inputs:
    - `name`: `str` canonical registry key, expected shape `<metric>_at_k`.

    Returns:
    - Registered validation metric function.
    """

    normalized_name = _normalize_metric_registry_key(name)
    if normalized_name not in _VALIDATION_METRIC_REGISTRY:
        available = ", ".join(list_validation_metrics()) or "none"
        raise ValueError(
            f"Validation metric '{normalized_name}' is not registered. "
            f"Available metrics: {available}"
        )
    return _VALIDATION_METRIC_REGISTRY[normalized_name]


def is_validation_metric_available(name: str) -> bool:
    """
    Check whether one slide-retrieval validation metric is registered.

    Inputs:
    - `name`: `str` canonical registry key.

    Returns:
    - `bool` indicating registry availability.
    """

    normalized_name = _normalize_metric_registry_key(name)
    return normalized_name in _VALIDATION_METRIC_REGISTRY


def list_validation_metrics() -> Sequence[str]:
    """
    Return all registered slide-retrieval validation metrics.

    Returns:
    - `Sequence[str]` of canonical registry keys in sorted order.
    """

    return sorted(_VALIDATION_METRIC_REGISTRY)


def _normalize_metric_registry_key(name: str) -> str:
    normalized_name = str(name).strip().lower()
    if not normalized_name:
        raise ValueError("Validation metric name must be a non-empty string.")
    return normalized_name

