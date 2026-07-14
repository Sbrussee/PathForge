from __future__ import annotations

import importlib
import json
import logging
import inspect
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from pathforge.config.config import Config, SearchSpaceParameter
from pathforge.core.datasets.bag_dataset import BagDataset
from pathforge.core.experiments.base import ComboConfig
from pathforge.utils.constants import DATASET_COL
from pathforge.utils.registries import MODELS


def calculate_combinations(params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """
    Calculate parameter combinations from a mapping of names to candidate values.

    This wrapper remains for compatibility, but ``ComboConfig`` is canonically
    implemented in ``pathforge.core.experiments.base`` to avoid duplicate
    implementations across PathForge.
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


def optimization_search_space(config: Config) -> dict[str, SearchSpaceParameter]:
    """Return the merged optimization search space for one config.

    The optimization policy samples hyperparameter ranges from
    ``optimization.search_space`` and automatically exposes every non-empty
    benchmark grid list as a categorical Optuna search dimension. This keeps
    pipeline-component selection and training hyperparameter search aligned
    between benchmarking and optimization modes.
    """

    merged_space = {
        name: (
            spec
            if isinstance(spec, SearchSpaceParameter)
            else SearchSpaceParameter.model_validate(spec)
        )
        for name, spec in config.optimization.search_space.items()
    }
    for name, values in benchmark_search_space(config).items():
        if name == "seeds" or name in merged_space or len(values) <= 1:
            continue
        merged_space[name] = SearchSpaceParameter(kind="categorical", choices=values)
    return merged_space


def apply_search_params(config: Config, params: dict[str, Any]) -> Config:
    """Apply one benchmark/optimization parameter set to a config copy."""

    _apply_benchmark_parameter_overrides(config, params)
    _apply_mil_parameter_overrides(config, params)
    _apply_active_component_overrides(config, params)

    setattr(config, "_active_search_params", dict(params))
    return config


def _apply_active_component_overrides(config: Config, params: dict[str, Any]) -> None:
    """Store active pipeline-component choices on the runtime config."""

    if "mil" in params:
        model_name = str(params["mil"])
        setattr(config, "_active_model_name", model_name)
        from pathforge.utils.registries import resolve_mil_model_backend

        backend = resolve_mil_model_backend(model_name)
        if backend == "torchmil" and model_name != "torchmil":
            config.mil.backend = "torchmil"
            config.mil.torchmil_model = model_name
        elif backend == "mil-lab" and model_name != "mil-lab":
            config.mil.backend = "mil-lab"
            config.mil.mil_lab_model = model_name
        elif backend == "native":
            config.mil.backend = "native"
    if "loss" in params:
        setattr(config, "_active_loss_name", str(params["loss"]))
    if "feature_extraction" in params:
        setattr(config, "_active_feature_extractor_name", str(params["feature_extraction"]))


def _apply_benchmark_parameter_overrides(
    config: Config,
    params: dict[str, Any],
) -> None:
    """Reduce benchmark grid lists to the active sampled value where applicable."""

    benchmark_fields = set(type(config.benchmark_parameters).model_fields)
    for key, value in params.items():
        if key not in benchmark_fields:
            continue
        current_value = getattr(config.benchmark_parameters, key)
        if isinstance(current_value, list):
            setattr(config.benchmark_parameters, key, [value])


def _apply_mil_parameter_overrides(config: Config, params: dict[str, Any]) -> None:
    """Map sampled parameter values onto the MIL config used by trainers/models."""

    for key, value in params.items():
        if key == "optimizer":
            config.mil.optimizer = str(value)
        elif key == "scheduler":
            config.mil.scheduler = str(value)
        elif key == "batch_size":
            config.mil.batch_size = int(value)
        elif key == "epochs":
            config.mil.epochs = int(value)
        elif key == "lr":
            config.mil.lr = float(value)
        elif key == "weight_decay":
            config.mil.weight_decay = float(value)
        elif key == "dropout_p":
            config.mil.dropout_p = float(value)
        elif key == "bag_size":
            config.mil.bag_size = int(value)
        elif key == "z_dim":
            config.mil.z_dim = int(value)
        elif key == "encoder_layers":
            config.mil.encoder_layers = int(value)
        elif key == "k":
            config.mil.k = int(value)


def build_bag_dataset_for_task(
    config: Config,
    *,
    feature_dir: str | Path,
    name: str,
    dataset_entry: Any | None = None,
) -> BagDataset:
    """Construct a task-aware bag dataset from config and a feature directory."""

    task = str(config.experiment.task or "classification")
    annotations_df = pd.read_csv(config.experiment.annotation_file)
    dataset_name = None
    if dataset_entry is not None and DATASET_COL in annotations_df.columns:
        dataset_name = str(dataset_entry.name)
    slide_column = config.experiment.slide_column
    if slide_column not in annotations_df.columns and "slide_id" in annotations_df.columns:
        slide_column = None
    return BagDataset(
        name,
        str(feature_dir),
        str(config.experiment.annotation_file),
        config.experiment.label_column,
        annotations_df=annotations_df,
        dataset_name=dataset_name,
        task=task,
        slide_column=slide_column,
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
        kwargs: Candidate keyword arguments assembled by PathForge.

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
        config: Active PathForge configuration.
        model_name: Registry key selected for this run.
        input_dim: Bag feature dimensionality inferred from ``BagDataset``.
        output_dim: Task output dimensionality inferred from annotations.
        extra_kwargs: Optional extra constructor kwargs supplied by callers.

    Returns:
        Any: Instantiated model object from the ``MODELS`` registry.
    """

    task = str(config.experiment.task or "classification")
    from pathforge.utils.registries import resolve_mil_model_backend

    backend = resolve_mil_model_backend(model_name)
    registry_name = backend if backend in {"torchmil", "mil-lab"} else model_name
    model_factory = MODELS.get(registry_name)
    caller_kwargs = dict(extra_kwargs or {})

    if backend == "torchmil":
        backend_kwargs = dict(config.mil.torchmil_model_kwargs)
        backend_kwargs.setdefault("in_shape", (int(input_dim),))
        backend_kwargs.setdefault("out_shape", int(output_dim))
        ctor_kwargs = {
            "torchmil_model": str(
                config.mil.torchmil_model if model_name == "torchmil" else model_name
            ),
            "task": task,
            "torchmil_model_kwargs": backend_kwargs,
        }
    elif backend == "mil-lab":
        backend_kwargs = dict(config.mil.mil_lab_model_kwargs)
        backend_kwargs.setdefault("input_dim", int(input_dim))
        backend_kwargs.setdefault("output_dim", int(output_dim))
        ctor_kwargs = {
            "mil_lab_model": str(
                config.mil.mil_lab_model if model_name == "mil-lab" else model_name
            ),
            "task": task,
            "mil_lab_model_kwargs": backend_kwargs,
            "mil_lab_from_pretrained": bool(config.mil.mil_lab_from_pretrained),
        }
    else:
        native_kwargs = _native_model_config_kwargs(config)
        ctor_kwargs = {
            "input_dim": int(input_dim),
            "output_dim": int(output_dim),
            **native_kwargs,
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


def _native_model_config_kwargs(config: Config) -> dict[str, Any]:
    """Return architecture kwargs commonly consumed by native MIL models.

    Args:
        config: Active PathForge config whose ``mil`` section defines sampled
            architecture and regularization settings.

    Returns:
        dict[str, Any]: Candidate kwargs for native model constructors. The
        final constructor call is still filtered by
        :func:`_filter_constructor_kwargs`.
    """

    z_dim = int(config.mil.z_dim)
    dropout = float(config.mil.dropout_p)
    layers = int(config.mil.encoder_layers)
    k_value = int(config.mil.k)
    return {
        "z_dim": z_dim,
        "hidden_dim": z_dim,
        "latent_dim": z_dim,
        "embed_dim": z_dim,
        "dropout": dropout,
        "dropout_p": dropout,
        "encoder_layers": layers,
        "num_layers": layers,
        "depth": layers,
        "k": k_value,
        "num_prototypes": k_value,
    }


def experiment_output_root(config: Config) -> Path:
    """Return the experiment-level output directory for policy artifacts.

    Args:
        config: Active PathForge configuration.

    Returns:
        Path: Absolute output directory used for experiment-wide reports.
    """

    root = Path(config.experiment.project_root or ".").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def metric_should_minimize(metric_name: str) -> bool:
    """Return whether lower values should rank higher for one metric name."""

    normalized = metric_name.lower()
    return any(
        token in normalized for token in ("loss", "error", "mae", "mse", "rmse")
    )


def collect_run_summary_row(
    config: Config,
    *,
    run_index: int,
    status: str,
    objective_metric: str,
    objective_value: float | None = None,
    checkpoint_path: str | None = None,
    error: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one flat experiment-summary row for a benchmark or optimization run.

    Args:
        config: Active run configuration.
        run_index: Zero-based run index within the experiment.
        status: Run status such as ``"success"`` or ``"failed"``.
        objective_metric: Metric used for run ranking.
        objective_value: Metric value for the completed run.
        checkpoint_path: Optional best-checkpoint path.
        error: Optional error message for failed runs.
        extra_fields: Optional extra scalar fields to append.

    Returns:
        dict[str, Any]: Flat summary row safe to store in CSV.
    """

    row: dict[str, Any] = {
        "run_index": int(run_index),
        "project_name": config.experiment.project_name,
        "mode": config.experiment.mode,
        "task": config.experiment.task,
        "status": status,
        "objective_metric": objective_metric,
        "objective_value": objective_value,
        "checkpoint_path": checkpoint_path,
        "error": error,
        "mil_backend": config.mil.backend,
        "model": getattr(config, "_active_model_name", None),
        "loss": getattr(config, "_active_loss_name", None),
        "feature_extraction": getattr(config, "_active_feature_extractor_name", None),
        "tile_px": _first_or_none(config.benchmark_parameters.tile_px),
        "tile_mpp": _first_or_none(config.benchmark_parameters.tile_mpp),
        "batch_size": config.mil.batch_size,
        "epochs": config.mil.epochs,
        "optimizer": config.mil.optimizer,
        "scheduler": config.mil.scheduler,
        "lr": config.mil.lr,
        "weight_decay": config.mil.weight_decay,
        "dropout_p": config.mil.dropout_p,
        "bag_size": config.mil.bag_size,
        "z_dim": config.mil.z_dim,
        "encoder_layers": config.mil.encoder_layers,
        "k": config.mil.k,
    }
    for key, value in getattr(config, "_active_search_params", {}).items():
        row[key] = _csv_safe_value(value)
    if extra_fields:
        for key, value in extra_fields.items():
            row[key] = _csv_safe_value(value)
    return row


def write_experiment_summary_csv(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    objective_metric: str,
    minimize: bool,
) -> pd.DataFrame:
    """Write one sorted experiment-wide summary CSV and return its dataframe."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        df = pd.DataFrame(
            columns=[
                "run_index",
                "status",
                "objective_metric",
                "objective_value",
            ]
        )
        df.to_csv(output_path, index=False)
        return df

    df = pd.DataFrame(rows)
    if "objective_value" in df.columns:
        df["objective_value"] = pd.to_numeric(df["objective_value"], errors="coerce")
    status_series = (
        df["status"].astype(str).str.lower()
        if "status" in df.columns
        else pd.Series("", index=df.index, dtype="object")
    )
    success_mask = status_series.isin({"success", "complete", "completed"}) & df[
        "objective_value"
    ].notna()
    success_df = df.loc[success_mask].sort_values(
        by=["objective_value", "run_index"],
        ascending=[minimize, True],
        kind="stable",
    )
    if not success_df.empty:
        success_df = success_df.copy()
        success_df["rank"] = range(1, len(success_df) + 1)
    failed_df = df.loc[~success_mask].copy()
    failed_df["rank"] = pd.NA
    if not failed_df.empty:
        failed_df = failed_df.sort_values(by=["run_index"], kind="stable")
    ordered = pd.concat([success_df, failed_df], ignore_index=True)
    ordered.to_csv(output_path, index=False)
    return ordered


def save_benchmark_visualizations(
    summary_csv_path: Path,
    *,
    output_dir: Path,
    objective_metric: str,
    minimize: bool,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Build ranked benchmark-wide visualizations from one summary CSV."""

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv_path)
    if df.empty or "objective_value" not in df.columns:
        return []
    status_series = (
        df["status"].astype(str).str.lower()
        if "status" in df.columns
        else pd.Series("", index=df.index, dtype="object")
    )
    success_df = df[
        status_series.isin({"success", "complete", "completed"})
        & pd.to_numeric(df["objective_value"], errors="coerce").notna()
    ].copy()
    if success_df.empty:
        return []
    success_df["objective_value"] = pd.to_numeric(
        success_df["objective_value"], errors="coerce"
    )
    success_df = success_df.sort_values(
        by=["objective_value", "run_index"],
        ascending=[minimize, True],
        kind="stable",
    )
    success_df["run_label"] = success_df.apply(
        lambda row: f"#{int(row['rank'])} {row.get('model') or 'run'}", axis=1
    )
    figures = _load_plotly_modules()
    if figures is None:
        _log_optional_skip(
            logger,
            "Skipping benchmark visualizations because plotly is not available.",
        )
        return []
    px, _ = figures
    hover_columns = [
        column
        for column in (
            "loss",
            "feature_extraction",
            "tile_px",
            "tile_mpp",
            "batch_size",
            "epochs",
            "lr",
            "dropout_p",
        )
        if column in success_df.columns and success_df[column].notna().any()
    ]
    exported: list[Path] = []
    ranked_bar = px.bar(
        success_df,
        x="run_label",
        y="objective_value",
        color="model" if "model" in success_df.columns and success_df["model"].notna().any() else None,
        hover_data=hover_columns,
        title=f"Benchmark Performance Ranked by {objective_metric}",
    )
    ranked_bar.update_layout(
        xaxis_title="Configuration Run (Best to Worst)",
        yaxis_title=objective_metric,
    )
    bar_path = output_dir / "benchmark_performance_ranked.html"
    ranked_bar.write_html(str(bar_path))
    exported.append(bar_path)

    ranked_scatter = px.scatter(
        success_df,
        x="rank",
        y="objective_value",
        color="model" if "model" in success_df.columns and success_df["model"].notna().any() else None,
        hover_name="run_label",
        hover_data=hover_columns,
        title=f"Benchmark Rank vs {objective_metric}",
    )
    ranked_scatter.update_layout(
        xaxis_title="Rank",
        yaxis_title=objective_metric,
    )
    scatter_path = output_dir / "benchmark_rank_scatter.html"
    ranked_scatter.write_html(str(scatter_path))
    exported.append(scatter_path)
    return exported


def save_optuna_visualizations(
    study: Any,
    *,
    output_dir: Path,
    logger: logging.Logger | None = None,
) -> list[Path]:
    """Export supported Optuna visualization figures as HTML and PNG files.

    The current Optuna documentation exposes Plotly-backed visualization
    helpers including ``plot_param_importances``, ``plot_optimization_history``,
    ``plot_rank``, ``plot_timeline``, and ``plot_hypervolume_history``.
    Hypervolume history is exported only for multi-objective studies because
    Optuna requires at least two objectives and a caller-provided reference
    point.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        visualization = importlib.import_module("optuna.visualization")
    except Exception:
        _log_optional_skip(
            logger,
            "Skipping Optuna visualizations because optuna.visualization is unavailable.",
        )
        return []

    exported: list[Path] = []
    figure_builders: list[tuple[str, Any, dict[str, Any]]] = [
        ("plot_optimization_history", getattr(visualization, "plot_optimization_history", None), {}),
        ("plot_param_importances", getattr(visualization, "plot_param_importances", None), {}),
        ("plot_rank", getattr(visualization, "plot_rank", None), {}),
        ("plot_timeline", getattr(visualization, "plot_timeline", None), {}),
    ]

    for name, builder, kwargs in figure_builders:
        if builder is None:
            continue
        try:
            fig = builder(study, **kwargs)
        except Exception as exc:
            _log_optional_skip(logger, f"Skipping Optuna visualization {name}: {exc}")
            continue
        output_path = output_dir / f"{name}.html"
        fig.write_html(str(output_path))
        exported.append(output_path)
        png_path = output_dir / f"{name}.png"
        if _write_optuna_png(
            fig,
            name=name,
            study=study,
            output_path=png_path,
            logger=logger,
        ):
            exported.append(png_path)

    hypervolume_builder = getattr(visualization, "plot_hypervolume_history", None)
    reference_point = _hypervolume_reference_point(study)
    if hypervolume_builder is not None and reference_point is not None:
        try:
            fig = hypervolume_builder(study, reference_point)
        except Exception as exc:
            _log_optional_skip(
                logger,
                f"Skipping Optuna visualization plot_hypervolume_history: {exc}",
            )
        else:
            output_path = output_dir / "plot_hypervolume_history.html"
            fig.write_html(str(output_path))
            exported.append(output_path)
            png_path = output_dir / "plot_hypervolume_history.png"
            if _write_optuna_png(
                fig,
                name="plot_hypervolume_history",
                study=study,
                output_path=png_path,
                logger=logger,
                reference_point=reference_point,
            ):
                exported.append(png_path)
    return exported


def _write_optuna_png(
    fig: Any,
    *,
    name: str,
    study: Any,
    output_path: Path,
    logger: logging.Logger | None = None,
    reference_point: Any | None = None,
) -> bool:
    try:
        fig.write_image(str(output_path), format="png")
        return output_path.exists()
    except Exception as exc:
        _log_optional_skip(
            logger,
            f"Plotly static export unavailable for Optuna visualization {name}: {exc}",
        )

    return _write_optuna_matplotlib_png(
        name=name,
        study=study,
        output_path=output_path,
        logger=logger,
        reference_point=reference_point,
    )


def _write_optuna_matplotlib_png(
    *,
    name: str,
    study: Any,
    output_path: Path,
    logger: logging.Logger | None = None,
    reference_point: Any | None = None,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        mpl_visualization = importlib.import_module("optuna.visualization.matplotlib")
        builder = getattr(mpl_visualization, name, None)
        if builder is None:
            return False
        if name == "plot_hypervolume_history":
            axis = builder(study, reference_point)
        else:
            axis = builder(study)
        figure = axis.figure
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path.exists()
    except Exception as exc:
        _log_optional_skip(
            logger,
            f"Skipping PNG export for Optuna visualization {name}: {exc}",
        )
        return False


def _first_or_none(values: list[Any]) -> Any:
    return values[0] if values else None


def _csv_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return json.dumps(value, sort_keys=True, default=str)


def _load_plotly_modules() -> tuple[Any, Any] | None:
    try:
        px = importlib.import_module("plotly.express")
        graph_objects = importlib.import_module("plotly.graph_objects")
    except Exception:
        return None
    return px, graph_objects


def _log_optional_skip(logger: logging.Logger | None, message: str) -> None:
    if logger is not None:
        logger.warning(message)


def _hypervolume_reference_point(study: Any) -> list[float] | None:
    """Infer a simple reference point for Optuna hypervolume plots."""

    directions = getattr(study, "directions", None)
    if not directions or len(directions) < 2:
        return None
    complete_trials = [
        trial
        for trial in getattr(study, "trials", [])
        if getattr(trial, "values", None) is not None
    ]
    if not complete_trials:
        return None
    objective_count = len(directions)
    columns = list(zip(*(trial.values for trial in complete_trials)))
    if len(columns) != objective_count:
        return None
    reference_point: list[float] = []
    for direction, values in zip(directions, columns):
        numeric_values = [float(value) for value in values]
        span = max(numeric_values) - min(numeric_values)
        margin = max(abs(span) * 0.1, 1.0)
        if str(direction).lower().endswith("minimize"):
            reference_point.append(max(numeric_values) + margin)
        else:
            reference_point.append(min(numeric_values) - margin)
    return reference_point


__all__ = [
    "ComboConfig",
    "apply_search_params",
    "benchmark_search_space",
    "build_mil_model_for_config",
    "build_bag_dataset_for_task",
    "calculate_combinations",
    "infer_model_dimensions",
    "collect_run_summary_row",
    "experiment_output_root",
    "metric_should_minimize",
    "optimization_search_space",
    "resolve_dataset_feature_dir",
    "save_benchmark_visualizations",
    "save_optuna_visualizations",
    "suggest_parameter",
    "write_experiment_summary_csv",
]
