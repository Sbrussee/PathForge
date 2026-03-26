from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HyperParam:
    """
    Declarative hyperparameter marker used by slide-retrieval strategies.

    Inputs:
    - `hp_type`: expected Python type used for casting and validation.
    - `default`: default value applied when no override is provided.
    - `min`: optional numeric lower bound.
    - `max`: optional numeric upper bound.
    - `choices`: optional finite set of allowed values.
    - `help`: human-readable description for registries / docs.

    Returns:
    - Immutable metadata object collected by strategy base classes.

    Example:
    ```python
    class ExampleStrategy:
        k = HyperParam(int, default=5, min=1, help="Retrieval depth.")
    ```
    """

    hp_type: type[Any] | None = None
    default: Any = None
    min: int | float | None = None
    max: int | float | None = None
    choices: tuple[Any, ...] | None = None
    help: str = ""

    def to_spec(self) -> dict[str, Any]:
        """
        Convert the declaration to the legacy registry-friendly metadata shape.

        Outputs:
        - `dict[str, Any]` with only populated metadata fields.
        """
        spec: dict[str, Any] = {}
        if self.hp_type is not None:
            spec["type"] = self.hp_type
        if self.default is not None or self.default is None:
            spec["default"] = self.default
        if self.min is not None:
            spec["min"] = self.min
        if self.max is not None:
            spec["max"] = self.max
        if self.choices is not None:
            spec["choices"] = list(self.choices)
        if self.help:
            spec["help"] = self.help
        return spec


def collect_hyperparams(strategy_cls: type[Any]) -> dict[str, HyperParam]:
    """
    Collect `HyperParam` declarations from a strategy class hierarchy.

    Inputs:
    - `strategy_cls`: strategy class whose MRO should be inspected.

    Returns:
    - `dict[str, HyperParam]` keyed by attribute name. Child classes override
      parent declarations with the same name.
    """
    collected: dict[str, HyperParam] = {}
    for base_cls in reversed(strategy_cls.__mro__):
        for name, value in vars(base_cls).items():
            if isinstance(value, HyperParam):
                collected[name] = value
    return collected


def resolve_hyperparam(
    *,
    name: str,
    declaration: HyperParam,
    params: dict[str, Any],
) -> Any:
    """
    Resolve one effective hyperparameter value from user params and defaults.

    Inputs:
    - `name`: hyperparameter field name.
    - `declaration`: declarative metadata for this hyperparameter.
    - `params`: runtime override mapping.

    Returns:
    - Validated and normalized effective value.
    """
    value = params.get(name, declaration.default)
    hp_type = declaration.hp_type

    if hp_type is not None and value is not None and not isinstance(value, hp_type):
        try:
            value = hp_type(value)
        except Exception as exc:
            raise ValueError(
                f"Failed to cast hyperparam '{name}' to {hp_type.__name__}: {value!r}"
            ) from exc

    if declaration.choices is not None and value is not None:
        if value not in declaration.choices:
            raise ValueError(
                f"Hyperparam '{name}' must be one of {list(declaration.choices)}, "
                f"got {value!r}"
            )

    if isinstance(value, (int, float)):
        if declaration.min is not None:
            value = max(declaration.min, value)
        if declaration.max is not None:
            value = min(declaration.max, value)

    return value
