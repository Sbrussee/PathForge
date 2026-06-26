from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pathforge.slide_retrieval.search_strategies.base import BaseSearchStrategy


_SEARCH_STRATEGY_REGISTRY: dict[str, type["BaseSearchStrategy"]] = {}
_SEARCH_STRATEGIES_IMPORTED = False


def _normalize_search_strategy_name(name: str) -> str:
    """Normalize a search strategy name for registry lookup."""
    search_strategy_name = str(name).strip().lower()
    if not search_strategy_name:
        raise ValueError("Search strategy name must be a non-empty string.")
    return search_strategy_name


def register_search_strategy(
    name: str,
) -> Callable[[type["BaseSearchStrategy"]], type["BaseSearchStrategy"]]:
    """Decorator to register a search strategy class."""
    normalized_name = _normalize_search_strategy_name(name)

    def decorator(
        strategy_cls: type["BaseSearchStrategy"],
    ) -> type["BaseSearchStrategy"]:
        if normalized_name in _SEARCH_STRATEGY_REGISTRY:
            raise ValueError(
                f"Search strategy '{normalized_name}' is already registered with "
                f"class {_SEARCH_STRATEGY_REGISTRY[normalized_name].__name__}."
            )

        _SEARCH_STRATEGY_REGISTRY[normalized_name] = strategy_cls
        return strategy_cls

    return decorator


def build_search_strategy(
    name: str,
    *args,
    **kwargs,
) -> "BaseSearchStrategy":
    """Resolve and instantiate a registered search strategy."""
    strategy_cls = get_search_strategy(name)
    return strategy_cls(*args, **kwargs)


def get_search_strategy(name: str) -> type["BaseSearchStrategy"]:
    """Get a registered search strategy class by name."""
    normalized_name = _normalize_search_strategy_name(name)

    if normalized_name not in _SEARCH_STRATEGY_REGISTRY:
        import_search_strategy_modules()

    if normalized_name not in _SEARCH_STRATEGY_REGISTRY:
        available = ", ".join(list_search_strategies()) or "none"
        raise ValueError(
            f"Search strategy '{normalized_name}' is not registered. "
            f"Available search strategies: {available}"
        )

    return _SEARCH_STRATEGY_REGISTRY[normalized_name]


def get_search_strategy_supported_representation_kinds(
    name: str,
) -> frozenset[str]:
    """Return representation kinds supported by a search strategy class."""
    strategy_cls = get_search_strategy(name)
    return frozenset(str(item) for item in strategy_cls.supported_representation_kinds)


def is_search_strategy_available(name: str) -> bool:
    """Check whether a search strategy is registered."""
    normalized_name = _normalize_search_strategy_name(name)
    return normalized_name in _SEARCH_STRATEGY_REGISTRY


def list_search_strategies() -> list[str]:
    """Return all registered search strategy names in sorted order."""
    return sorted(_SEARCH_STRATEGY_REGISTRY.keys())


def import_search_strategy_modules(
    package_name: str = "pathforge.slide_retrieval.search_strategies",
) -> None:
    """
    Import the canonical search strategy package.

    Inputs:
    - `package_name`: `str` package root containing the `strategies`
      subpackage.

    Returns:
    - `None`. Import side effects register all search strategies.
    """
    global _SEARCH_STRATEGIES_IMPORTED

    if _SEARCH_STRATEGIES_IMPORTED:
        return

    package = import_module(package_name)

    if not hasattr(package, "__path__"):
        raise ValueError(f"Package '{package_name}' does not have a __path__ attribute.")

    import_module(f"{package_name}.strategies")
    _SEARCH_STRATEGIES_IMPORTED = True


def get_search_strategy_supports(name: str) -> set[str]:
    """Return the supported representation types for a search strategy."""
    strategy_cls = get_search_strategy(name)
    supports = getattr(strategy_cls, "supports", set()) or set()
    return {str(item).strip().lower() for item in supports}


def get_search_strategy_hyperparams(name: str) -> dict[str, dict[str, Any]]:
    """Return the hyperparameter schema for a registered search strategy."""
    strategy_cls = get_search_strategy(name)
    return strategy_cls.hyperparam_spec()


def get_search_strategy_defaults(name: str) -> dict[str, Any]:
    """Return the default hyperparameter values for a registered search strategy."""
    spec = get_search_strategy_hyperparams(name)
    return {key: meta.get("default") for key, meta in spec.items()}


def get_search_strategy_values(
    name: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return effective hyperparameter values without instantiating the strategy."""
    spec = get_search_strategy_hyperparams(name)
    params = params or {}

    values: dict[str, Any] = {}
    for key, meta in spec.items():
        values[key] = params.get(key, meta.get("default"))

    return values


def describe_all_search_strategies() -> dict[str, dict[str, dict[str, Any]]]:
    """Return the hyperparameter schema for all registered search strategies."""
    return {
        name: get_search_strategy_hyperparams(name)
        for name in list_search_strategies()
    }
