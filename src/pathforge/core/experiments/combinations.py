from __future__ import annotations

from itertools import product
from typing import Any

from pathforge.config.config import BenchmarkParamEntry, Config


class ComboConfig:
    """
    Generic, dynamically-populated benchmark combination configuration.

    Inputs:
        keyword arguments (object):
            Benchmark parameter values keyed by parameter name. Each key becomes
            an attribute on the created object.

    Outputs:
        ComboConfig:
            Configuration object exposing combo values and optional
            ``<key>_params`` hyperparameter dictionaries.

    Semantic goal:
        Represent one fully-materialized benchmark parameter combination in a
        lightweight attribute-based object.

    Example:
        .. code-block:: python

            combo_cfg = ComboConfig.from_keys_values(
                keys=["feature_extraction", "tile_px"],
                values=[
                    BenchmarkParamEntry(value="uni", hyperparams={"family": "foundation"}),
                    256,
                ],
            )
            assert combo_cfg.feature_extraction == "uni"
            assert combo_cfg.get_hyperparams("feature_extraction") == {
                "family": "foundation",
            }

    """

    def __init__(self, **kwargs: object) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_keys_values(
        cls,
        keys: list[str],
        values: list[object],
    ) -> "ComboConfig":
        """
        Build one combo config from aligned key/value lists.

        Inputs:
            keys (list[str]):
                Parameter names. Shape: ``(n_keys,)``.
            values (list[object]):
                Parameter values aligned with ``keys``. Shape: ``(n_keys,)``.

        Outputs:
            ComboConfig:
                Combo object containing one attribute per key and one
                ``<key>_params`` attribute per key.
        """
        data: dict[str, object] = {}

        for key, value in zip(keys, values):
            if isinstance(value, BenchmarkParamEntry):
                data[key] = value.value
                data[f"{key}_params"] = dict(value.hyperparams)
            else:
                data[key] = value
                data[f"{key}_params"] = {}

        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        """Return a shallow dictionary representation of the combo."""
        return dict(self.__dict__)

    def get(self, key: str, default: object = None) -> object:
        """Return one combo value by key with a dict-like fallback default."""
        return getattr(self, key, default)

    def get_hyperparams(
        self,
        key: str,
        default: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the hyperparameters attached to one combo value."""
        params = getattr(self, f"{key}_params", None)
        if params is None:
            return {} if default is None else dict(default)
        return dict(params)


def build_combinations(cfg: Config, keys: list[str]) -> list[ComboConfig]:
    """
    Materialize the benchmark search space for the requested parameter keys.

    Inputs:
        cfg (Config):
            Validated application configuration containing
            ``benchmark_parameters``.
        keys (list[str]):
            Parameter names to include in the grid. Shape: ``(n_keys,)``.

    Outputs:
        list[ComboConfig]:
            List of materialized benchmark combinations. Shape:
            ``(n_combos,)``.

    Semantic goal:
        Convert the declarative benchmark parameter config into concrete combo
        objects that policies and tasks can execute.

    Example:
        .. code-block:: python

            combos = build_combinations(
                cfg=cfg,
                keys=["feature_extraction", "tile_px", "tile_mpp"],
            )

    """
    benchmark_params = cfg.benchmark_parameters
    value_lists: list[list[Any]] = []

    for key in keys:
        if not hasattr(benchmark_params, key):
            raise AttributeError(f"benchmark_parameters has no field '{key}'")

        values = benchmark_params.get_entries(key)
        if key == "color_norm" and not values:
            value_lists.append([None])
            continue

        if not values:
            raise ValueError(f"benchmark_parameters.{key} is empty; cannot build grid.")

        value_lists.append(values)

    combos: list[ComboConfig] = []

    for values in product(*value_lists):
        combos.append(ComboConfig.from_keys_values(keys, list(values)))

    return combos
