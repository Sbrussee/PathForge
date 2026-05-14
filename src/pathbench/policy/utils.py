from __future__ import annotations

import inspect
from itertools import product
from pathlib import Path
from typing import Any

from pathbench.config.config import Config, SearchSpaceParameter
from pathbench.core.datasets.bag_dataset import BagDataset
from pathbench.core.experiments.base import ComboConfig
from pathbench.utils.registries import MODELS


def calculate_combinations(params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Calculate parameter combinations from a mapping of names to candidate values.

    This wrapper remains for compatibility, but ``ComboConfig`` is canonically
    implemented in ``pathbench.core.experiments.base`` to avoid duplicate
    implementations across PathBench.
    """

    keys = list(params)
    values = [params[key] for key in keys]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def benchmark_search_space(config: Config) -> dict[str, list[Any]]:
    """Return the non-empty benchmark grid declared in the config."""

    payload = config.benchmark_parameters.model_dump()
    return {
        key: value
        for key, value in payload.items()
        if isinstance(value, list) and len(value) > 0
    }


def apply_search_params(config: Config, params: dict[str, Any]) -> Config:
    """Apply one benchmark/optimization parameter set to a config copy."""

    if "mil" in params:
        setattr(config, "_active_model_name", str(params["mil"]))
    if "loss" in params:
        setattr(config, "_active_loss_name", str(params["loss"]))

    for key, value in params.items():
        if key == "optimizer":
            config.mil.optimizer = str(value)
        elif key == "batch_size":
            config.mil.batch_size = int(value)
        elif key == "lr":
            config.mil.lr = float(value)
        elif key == "dropout_p":
            config.mil.dropout_p = float(value)
        elif key == "mil":
            continue
        elif key == "loss":
            continue

    setattr(config, "_active_search_params", dict(params))
    return config


def build_bag_dataset_for_task(
    config: Config,
    *,
    feature_dir: str | Path,
    name: str,
) -> BagDataset:
    """Construct a task-aware bag dataset from config and a feature directory."""

    task = str(config.experiment.task or "classification")
    return BagDataset(
        name,
        str(feature_dir),
        str(config.experiment.annotation_file),
        config.experiment.label_column,
        task=task,
        slide_column=config.experiment.slide_column,
        time_column=config.experiment.survival_time_column,
        event_column=config.experiment.survival_event_column,
        bag_size=config.mil.bag_size,
    )


def resolve_dataset_feature_dir(dataset_entry: Any) -> Path:
    """Resolve the directory containing bag feature files for one dataset."""

    feature_dir = (
        getattr(dataset_entry, "features_dir", None) or dataset_entry.artifacts_dir
    )
    return Path(feature_dir)


def infer_model_dimensions(dataset: BagDataset) -> tuple[int, int]:
    """Infer ``(input_dim, output_dim)`` from one prepared bag dataset."""

    return dataset.feature_dim, dataset.output_dim()


def _filter_constructor_kwargs(factory: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return constructor kwargs accepted by one model factory.

    Args:
        factory: Callable model constructor or registered factory.
        kwargs: Candidate keyword arguments assembled by PathBench.

    Returns:
        dict[str, Any]: Accepted keyword arguments. If the callable accepts
        ``**kwargs`` or cannot be introspected, the input mapping is returned.
    """

    try:
        parameters = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return kwargs

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return kwargs

    accepted_names = {
        param.name
        for param in parameters
        if param.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {name: value for name, value in kwargs.items() if name in accepted_names}


def build_mil_model_for_config(
    config: Config,
    *,
    model_name: str,
    input_dim: int,
    output_dim: int,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build one MIL model while respecting backend-specific user config.

    Args:
        config: Active PathBench configuration.
        model_name: Registry key selected for this run.
        input_dim: Bag feature dimensionality inferred from ``BagDataset``.
        output_dim: Task output dimensionality inferred from annotations.
        extra_kwargs: Optional extra constructor kwargs supplied by callers.

    Returns:
        Any: Instantiated model object from the ``MODELS`` registry.
    """

    task = str(config.experiment.task or "classification")
    model_factory = MODELS.get(model_name)
    caller_kwargs = dict(extra_kwargs or {})

    if config.mil.backend == "torchmil":
        backend_kwargs = dict(config.mil.torchmil_model_kwargs)
        backend_kwargs.setdefault("in_shape", (int(input_dim),))
        backend_kwargs.setdefault("out_shape", int(output_dim))
        ctor_kwargs = {
            "torchmil_model": str(config.mil.torchmil_model),
            "task": task,
            "torchmil_model_kwargs": backend_kwargs,
        }
    elif config.mil.backend == "mil-lab":
        backend_kwargs = dict(config.mil.mil_lab_model_kwargs)
        backend_kwargs.setdefault("input_dim", int(input_dim))
        backend_kwargs.setdefault("output_dim", int(output_dim))
        ctor_kwargs = {
            "mil_lab_model": str(config.mil.mil_lab_model),
            "task": task,
            "mil_lab_model_kwargs": backend_kwargs,
            "mil_lab_from_pretrained": bool(config.mil.mil_lab_from_pretrained),
        }
    else:
        ctor_kwargs = {
            "input_dim": int(input_dim),
            "output_dim": int(output_dim),
            **caller_kwargs,
        }

    filtered_kwargs = _filter_constructor_kwargs(model_factory, ctor_kwargs)
    return model_factory(**filtered_kwargs)


def suggest_parameter(
    trial: Any,
    *,
    name: str,
    spec: SearchSpaceParameter,
) -> Any:
    """Suggest one optimization parameter from a validated search-space spec."""

    if spec.kind == "categorical":
        return trial.suggest_categorical(name, spec.choices)
    if spec.kind == "int":
        return trial.suggest_int(
            name,
            int(spec.low),
            int(spec.high),
            step=int(spec.step) if spec.step is not None else 1,
            log=bool(spec.log),
        )
    return trial.suggest_float(
        name,
        float(spec.low),
        float(spec.high),
        step=float(spec.step) if spec.step is not None else None,
        log=bool(spec.log),
    )


__all__ = [
    "ComboConfig",
    "apply_search_params",
    "benchmark_search_space",
    "build_mil_model_for_config",
    "build_bag_dataset_for_task",
    "calculate_combinations",
    "infer_model_dimensions",
    "resolve_dataset_feature_dir",
    "suggest_parameter",
]
