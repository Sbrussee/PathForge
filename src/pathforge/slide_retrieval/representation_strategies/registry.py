from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pathforge.slide_retrieval.representation_strategies.base import (
        BaseRetrievalRepresentationStrategy,
    )


_RETRIEVAL_REPRESENTATION_REGISTRY: dict[
    str,
    type["BaseRetrievalRepresentationStrategy"],
] = {}
_REPRESENTATION_STRATEGIES_IMPORTED = False


def _normalize_representation_strategy_name(name: str) -> str:
    """Normalize a retrieval representation name for registry lookup."""
    retrieval_representation_name = str(name).strip().lower()
    if not retrieval_representation_name:
        raise ValueError("Retrieval representation name must be a non-empty string.")
    return retrieval_representation_name


def register_representation_strategy(
    name: str,
) -> Callable[
    [type["BaseRetrievalRepresentationStrategy"]],
    type["BaseRetrievalRepresentationStrategy"],
]:
    """Decorator to register a retrieval representation strategy class."""
    normalized_name = _normalize_representation_strategy_name(name)

    def decorator(
        strategy_cls: type["BaseRetrievalRepresentationStrategy"],
    ) -> type["BaseRetrievalRepresentationStrategy"]:
        if normalized_name in _RETRIEVAL_REPRESENTATION_REGISTRY:
            raise ValueError(
                f"Retrieval representation '{normalized_name}' is already registered with "
                f"class {_RETRIEVAL_REPRESENTATION_REGISTRY[normalized_name].__name__}."
            )

        _RETRIEVAL_REPRESENTATION_REGISTRY[normalized_name] = strategy_cls
        return strategy_cls

    return decorator


def build_representation_strategy(
    name: str,
    *args,
    **kwargs,
) -> "BaseRetrievalRepresentationStrategy":
    """Resolve and instantiate a registered retrieval representation strategy."""
    strategy_cls = get_representation_strategy(name)
    return strategy_cls(*args, **kwargs)


def get_representation_strategy(
    name: str,
) -> type["BaseRetrievalRepresentationStrategy"]:
    """Get a registered retrieval representation strategy class by name."""
    normalized_name = _normalize_representation_strategy_name(name)

    if normalized_name not in _RETRIEVAL_REPRESENTATION_REGISTRY:
        import_representation_strategy_modules()

    if normalized_name not in _RETRIEVAL_REPRESENTATION_REGISTRY:
        available = ", ".join(list_representation_strategies()) or "none"
        raise ValueError(
            f"Retrieval representation '{normalized_name}' is not registered. "
            f"Available retrieval representations: {available}"
        )

    return _RETRIEVAL_REPRESENTATION_REGISTRY[normalized_name]


def get_representation_strategy_output_kind(name: str) -> str:
    """Return the output representation kind declared by a strategy class."""
    strategy_cls = get_representation_strategy(name)
    return str(strategy_cls.output_representation_kind)


def get_representation_strategy_supported_feature_levels(
    name: str,
) -> frozenset[str]:
    """Return feature levels supported by a representation strategy class."""
    strategy_cls = get_representation_strategy(name)
    return frozenset(str(item) for item in strategy_cls.supported_feature_levels)


def is_representation_strategy_available(name: str) -> bool:
    """Check whether a retrieval representation is registered."""
    normalized_name = _normalize_representation_strategy_name(name)
    return normalized_name in _RETRIEVAL_REPRESENTATION_REGISTRY


def list_representation_strategies() -> list[str]:
    """Return all registered retrieval representation names in sorted order."""
    return sorted(_RETRIEVAL_REPRESENTATION_REGISTRY.keys())


def import_representation_strategy_modules(
    package_name: str = "pathforge.slide_retrieval.representation_strategies",
) -> None:
    """
    Import the canonical retrieval representation strategy package.

    Inputs:
    - package_name: `str` package root containing the `strategies` subpackage.

    Returns:
    - `None`. Import side effects register all retrieval representation strategies.

    Example:
    ```python
    import_representation_strategy_modules()
    ```
    """
    global _REPRESENTATION_STRATEGIES_IMPORTED

    if _REPRESENTATION_STRATEGIES_IMPORTED:
        return

    package = import_module(package_name)

    if not hasattr(package, "__path__"):
        raise ValueError(f"Package '{package_name}' does not have a __path__ attribute.")

    # Import the explicit strategy package so registration is driven by one
    # authoritative module list instead of filesystem discovery.
    import_module(f"{package_name}.strategies")
    _REPRESENTATION_STRATEGIES_IMPORTED = True


def get_representation_strategy_hyperparams(name: str) -> dict[str, Any]:
    """Return the hyperparameter schema for a registered retrieval representation."""
    strategy_cls = get_representation_strategy(name)
    return strategy_cls.hyperparam_spec()


def get_representation_strategy_defaults(name: str) -> dict[str, Any]:
    """Return the default hyperparameter values for a registered retrieval representation."""
    spec = get_representation_strategy_hyperparams(name)
    return {key: meta.get("default") for key, meta in spec.items()}


def get_representation_strategy_values(
    name: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return effective hyperparameter values without instantiating the strategy."""
    spec = get_representation_strategy_hyperparams(name)
    params = params or {}

    values: dict[str, Any] = {}
    for key, meta in spec.items():
        values[key] = params.get(key, meta.get("default"))

    return values


def describe_all_representations_strategies() -> dict[str, dict[str, dict[str, Any]]]:
    """Return the hyperparameter schema for all registered retrieval representations."""
    descriptions: dict[str, dict[str, dict[str, Any]]] = {}

    for name, strategy_cls in _RETRIEVAL_REPRESENTATION_REGISTRY.items():
        descriptions[name] = strategy_cls.hyperparam_spec()

    return descriptions
