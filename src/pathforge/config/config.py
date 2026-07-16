from __future__ import annotations

import csv
import inspect
import os
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional

import torch
import yaml

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from pathforge.utils.constants import TASK_TYPES, MODE_TYPES, AGGREGATION_LEVELS
from pathforge.utils.optional.torchmil import (
    is_torchmetrics_available,
    is_torchmil_available,
    is_torchsurv_available,
)
from pathforge.utils.optional.mil_lab import is_mil_lab_available
from pathforge.utils.registries import (
    MODELS,
    LAZYSLIDE_MODEL_NAMES,
    is_feature_extractor_available,
    all_feature_extractor_names,
    populate_dynamic_registries,
)
from pathforge.core.models.mil_base import MILModelBase
from pathforge.adapters.tcga_tools import resolve_external_dataset_sources

TaskType = Literal[tuple(TASK_TYPES)]
ModeType = Literal[tuple(MODE_TYPES)]
AggregationLevel = Literal[tuple(AGGREGATION_LEVELS)]

BenchmarkScalar = int | float | str | None
BenchmarkParamMapping = dict[BenchmarkScalar, dict[str, Any]]
BenchmarkParamInput = BenchmarkScalar | BenchmarkParamMapping


@dataclass(frozen=True, slots=True)
class BenchmarkParamEntry:
    """
    Normalized benchmark-parameter entry used for combo expansion.

    Inputs:
    - `value`: base benchmark value consumed by existing code paths.
    - `hyperparams`: optional hyperparameter mapping attached to that value.

    Returns:
    - Immutable normalized entry for one benchmark-parameter option.
    """

    value: BenchmarkScalar
    hyperparams: dict[str, Any] = dataclass_field(default_factory=dict)


CLASSIFICATION_METRIC_NAMES = {
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auroc",
    "pr_auc",
    "brier_score",
}
SURVIVAL_METRIC_NAMES = {
    "brier_score",
    "c_index",
    "td_auc",
    "num_eval_times",
}
REGRESSION_METRIC_NAMES = {
    "mae",
    "mse",
    "rmse",
    "r2",
}
MONITOR_FALLBACK_NAMES = {"val_loss", "train_loss", "loss"}

# ---------------------------------------------------------------------------
# Config Sections
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Universal experiment settings."""

    project_name: str
    annotation_file: str
    project_root: str | None = None

    num_workers: int = Field(0, ge=0)

    # Task + Mode
    task: Optional[TaskType] = None
    mode: ModeType = "benchmark"
    prediction_level: Literal["mil", "slide"] = "mil"
    aggregation_level: AggregationLevel = "slide"
    label_column: str = "category"
    slide_column: str = "slide"
    survival_time_column: Optional[str] = None
    survival_event_column: Optional[str] = None

    # Global behavior
    report: bool = False
    thumbnail: bool = False
    mixed_precision: bool = False

    # Optional reporting/evaluation selectors preserved for config compatibility.
    visualization: List[str] = Field(default_factory=list)
    evaluation: List[str] = Field(default_factory=list)
    custom_metrics: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_task_for_mode(self) -> "ExperimentConfig":
        if self.mode != "feature_extraction" and self.task is None:
            raise ValueError(
                "experiment.task is required unless mode == 'feature_extraction'."
            )
        return self

    @field_validator("num_workers")
    @classmethod
    def validate_num_workers(cls, value: int) -> int:
        available_cpus = os.cpu_count() or 1
        if value > available_cpus:
            raise ValueError(
                "experiment.num_workers exceeds available CPUs. "
                f"Got {value}, available {available_cpus}."
            )
        return value


class ClassificationConfig(BaseModel):
    """Classification task settings."""

    split_technique: Literal["k-fold", "k-fold-stratified", "fixed"] = "k-fold"
    val_fraction: float = Field(0.1, gt=0, lt=1)

    # Training Loop
    epochs: int = Field(20, gt=0)
    batch_size: int = Field(1, gt=0)
    best_epoch_based_on: str = "val_loss"
    patience: int = Field(10, ge=1)
    accumulate_grad_batches: int = Field(1, ge=1)
    gradient_clip_val: float = Field(0.0, ge=0.0)

    # Optimization
    lr: float = Field(1e-4, gt=0)
    weight_decay: float = Field(1e-5, ge=0)
    scheduler: Literal["none", "reduce_on_plateau", "cosine"] = "none"
    scheduler_monitor: str = "val_loss"

    # Data
    balancing: Optional[str] = None
    class_weighting: bool = False

    # Architecture (General)
    bag_size: int = 512
    encoder_layers: int = 1
    z_dim: int = 256
    dropout_p: float = Field(0.1, ge=0.0, le=1.0)
    k: int = 2

    skip_extracted: bool = True
    skip_feature_extraction: bool = True


class MILConfig(BaseModel):
    """MIL backend selection and training settings."""

    backend: Literal["native", "torchmil", "mil-lab"] = "native"
    torchmil_model: Optional[str] = None
    torchmil_model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    use_torchmil_collate: bool = True
    mil_lab_model: Optional[str] = None
    mil_lab_model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    mil_lab_from_pretrained: bool = False

    # Training Loop
    epochs: int = Field(20, gt=0)
    batch_size: int = Field(1, gt=0)
    best_epoch_based_on: str = "val_loss"
    patience: int = Field(10, ge=1)
    accumulate_grad_batches: int = Field(1, ge=1)
    gradient_clip_val: float = Field(0.0, ge=0.0)

    # Optimization
    optimizer: str = "Adam"
    lr: float = Field(1e-4, gt=0)
    weight_decay: float = Field(1e-5, ge=0)
    scheduler: Literal["none", "reduce_on_plateau", "cosine"] = "none"
    scheduler_monitor: str = "val_loss"

    # Data
    balancing: Optional[str] = None
    class_weighting: bool = False

    # Architecture (General)
    bag_size: int = 512
    encoder_layers: int = 1
    z_dim: int = 256
    dropout_p: float = Field(0.1, ge=0.0, le=1.0)
    k: int = 2

    skip_extracted: bool = True
    skip_feature_extraction: bool = True


class SlideRetrievalConfig(BaseModel):
    """Slide retrieval task settings."""

    exclusion_level: Literal["none", "slide", "case", "patient"] = "patient"
    search_workers: int = Field(1, ge=1)
    representation_loader_workers: int | None = Field(default=None, ge=0)
    representation_workers: int | None = Field(default=None, ge=1)
    visualization_metadata_columns: List[str] = Field(default_factory=list)
    visualization_top_k: int = Field(10, ge=1)
    prototypes_root: str | None = None

    @field_validator("visualization_metadata_columns")
    @classmethod
    def validate_visualization_metadata_columns(
        cls,
        value: List[str],
    ) -> List[str]:
        normalized_columns: list[str] = []
        for column_name in value:
            normalized_name = str(column_name).strip()
            if not normalized_name:
                raise ValueError(
                    "slide_retrieval.visualization_metadata_columns may not contain empty values."
                )
            normalized_columns.append(normalized_name)
        return normalized_columns


class ExplainabilityConfig(BaseModel):
    """Explainability backend selection."""

    heatmap_backend: Literal["native", "torchmil"] = "native"
    heatmap_colormap: str = "inferno"
    heatmap_tile_alpha: float = Field(0.65, ge=0.0, le=1.0)
    heatmap_smoothed_alpha: float = Field(0.8, ge=0.0, le=1.0)
    heatmap_smoothing_sigma_scale: float = Field(0.75, gt=0.0)
    heatmap_top_k_tiles: int = Field(10, ge=1)


class MetricsConfig(BaseModel):
    """Metric and loss backend selection.

    Examples:
        .. code-block:: yaml

            metrics:
              classification_backend: torchmetrics
              survival_continuous_backend: torchsurv

    """

    classification_backend: str = "torchmetrics"
    survival_continuous_backend: str = "torchsurv"
    registry_namespace: Optional[str] = None
    classification_metrics: list[str] = Field(
        default_factory=lambda: [
            "accuracy",
            "balanced_accuracy",
            "f1",
            "auroc",
            "pr_auc",
        ]
    )
    survival_metrics: list[str] = Field(
        default_factory=lambda: ["c_index", "td_auc", "brier_score", "num_eval_times"]
    )
    regression_metrics: list[str] = Field(default_factory=lambda: ["mae", "mse"])

    @field_validator("classification_metrics")
    @classmethod
    def validate_classification_metrics(cls, value: list[str]) -> list[str]:
        invalid = sorted(set(value) - CLASSIFICATION_METRIC_NAMES)
        if invalid:
            raise ValueError(
                "Unsupported classification metrics: "
                f"{invalid}. Allowed metrics: {sorted(CLASSIFICATION_METRIC_NAMES)}."
            )
        return value

    @field_validator("survival_metrics")
    @classmethod
    def validate_survival_metrics(cls, value: list[str]) -> list[str]:
        invalid = sorted(set(value) - SURVIVAL_METRIC_NAMES)
        if invalid:
            raise ValueError(
                "Unsupported survival metrics: "
                f"{invalid}. Allowed metrics: {sorted(SURVIVAL_METRIC_NAMES)}."
            )
        return value

    @field_validator("regression_metrics")
    @classmethod
    def validate_regression_metrics(cls, value: list[str]) -> list[str]:
        invalid = sorted(set(value) - REGRESSION_METRIC_NAMES)
        if invalid:
            raise ValueError(
                "Unsupported regression metrics: "
                f"{invalid}. Allowed metrics: {sorted(REGRESSION_METRIC_NAMES)}."
            )
        return value

    def metrics_for_task(self, task: str | None) -> list[str]:
        """Return the configured metric names compatible with one task."""
        if task == "classification":
            return list(self.classification_metrics)
        if task in {"survival", "survival_discrete"}:
            return list(self.survival_metrics)
        if task == "regression":
            return list(self.regression_metrics)
        return []


class SearchSpaceParameter(BaseModel):
    """One optimization search-space parameter specification."""

    kind: Literal["categorical", "float", "int"] = Field(
        validation_alias=AliasChoices("kind", "type")
    )
    choices: list[Any] = Field(default_factory=list)
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False

    @model_validator(mode="after")
    def validate_shape(self) -> "SearchSpaceParameter":
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(
                    "categorical search-space parameters require non-empty choices."
                )
            return self
        if self.low is None or self.high is None:
            raise ValueError(
                "float/int search-space parameters require both low and high values."
            )
        if self.high < self.low:
            raise ValueError("search-space parameter high must be >= low.")
        return self


class FeatureExtractionRuntimeConfig(BaseModel):
    """Runtime controls forwarded to slide-backend feature extraction."""

    batch_size: int = Field(32, gt=0)
    num_workers: int = Field(0, ge=0)
    amp: bool = False


class SlideProcessingConfig(BaseModel):
    """Settings for slide processing backends."""

    backend: Literal["lazyslide", "openslide", "cucim"] = "lazyslide"
    save_tiles: bool = False
    segmentation_method: Optional[str] = None
    feature_extraction: FeatureExtractionRuntimeConfig = Field(
        default_factory=FeatureExtractionRuntimeConfig
    )

    qc_filters: List[Dict[str, Any]] = Field(default_factory=list)


class OptimizationConfig(BaseModel):
    """Optuna Optimization settings."""

    study_name: str = "study"
    load_study: bool = False
    objective_metric: str = "balanced_accuracy"
    objective_mode: Literal["max", "min"] = "max"
    objective_dataset: Literal["val", "test"] = "val"

    sampler: str = "TPESampler"
    trials: int = Field(100, gt=0)
    pruner: Optional[str] = "HyperbandPruner"
    storage: Optional[str] = None
    trials_per_worker: int = Field(1, gt=0)
    heartbeat_interval: Optional[int] = Field(default=None, gt=0)
    stale_trial_timeout: Optional[int] = Field(default=None, gt=0)
    search_space: dict[str, SearchSpaceParameter] = Field(
        default_factory=lambda: {
            "lr": SearchSpaceParameter(kind="float", low=1e-5, high=1e-3, log=True),
            "epochs": SearchSpaceParameter(kind="int", low=10, high=50, step=5),
            "z_dim": SearchSpaceParameter(
                kind="categorical",
                choices=[128, 256, 512],
            ),
            "dropout_p": SearchSpaceParameter(kind="float", low=0.1, high=0.5),
            "weight_decay": SearchSpaceParameter(
                kind="float",
                low=1e-6,
                high=1e-3,
                log=True,
            ),
        }
    )


class StageResourceConfig(BaseModel):
    """Resource request used when rendering one distributed execution stage."""

    cpus: int = Field(1, gt=0)
    gpus: int = Field(0, ge=0)
    memory_gb: int = Field(8, gt=0)
    time: str = "01:00:00"


class ExecutionResourcesConfig(BaseModel):
    """Per-stage resource requests for generated scheduler jobs."""

    feature_extraction: StageResourceConfig = Field(
        default_factory=lambda: StageResourceConfig(
            cpus=4, gpus=1, memory_gb=32, time="04:00:00"
        )
    )
    benchmarking: StageResourceConfig = Field(
        default_factory=lambda: StageResourceConfig(
            cpus=4, gpus=1, memory_gb=24, time="08:00:00"
        )
    )
    optimization: StageResourceConfig = Field(
        default_factory=lambda: StageResourceConfig(
            cpus=4, gpus=1, memory_gb=24, time="08:00:00"
        )
    )
    aggregation: StageResourceConfig = Field(default_factory=StageResourceConfig)


class SlurmExecutionConfig(BaseModel):
    """SLURM-specific rendering controls for distributed execution plans."""

    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    constraint: Optional[str] = None
    max_concurrent: int = Field(20, gt=0)
    extra_directives: list[str] = Field(default_factory=list)


class ExecutionConfig(BaseModel):
    """Global scheduler-neutral settings for distributed PathForge work."""

    backend: Literal["local", "slurm", "dask"] = "local"
    work_dir: Optional[str] = None
    resume: bool = True
    max_workers: int = Field(1, gt=0)
    resources: ExecutionResourcesConfig = Field(default_factory=ExecutionResourcesConfig)
    slurm: SlurmExecutionConfig = Field(default_factory=SlurmExecutionConfig)


class EvaluationConfig(BaseModel):
    """Post-run evaluation and visualization settings."""

    label_column: str | None = None
    metrics: List[str] = Field(default_factory=list)
    visualization: List[str] = Field(default_factory=list)
    visualization_subset_file: str | None = None

    @field_validator("label_column")
    @classmethod
    def validate_label_column(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized_value = str(value).strip()
        if not normalized_value:
            raise ValueError("evaluation.label_column must be a non-empty string.")
        return normalized_value

    @field_validator("visualization_subset_file")
    @classmethod
    def validate_visualization_subset_file(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        normalized_value = str(value).strip()
        if not normalized_value:
            raise ValueError(
                "evaluation.visualization_subset_file must be a non-empty string when provided."
            )
        return normalized_value


class DatasetEntry(BaseModel):
    """Definition of a dataset source."""

    name: str
    slides_dir: str
    artifacts_dir: str
    features_dir: Optional[str] = None
    tissue_annotations_dir: Optional[str] = None
    used_for: str

    @field_validator("used_for")
    @classmethod
    def validate_used_for(cls, value: str) -> str:
        normalized_value = str(value).strip().lower()
        if not normalized_value:
            raise ValueError("datasets[].used_for must be a non-empty string.")
        return normalized_value


class BenchmarkParameters(BaseModel):
    """
    Grid search parameters.
    Pydantic validators enforce logic previously implemented manually.
    """

    # --- Slide processing ---
    tile_px: List[BenchmarkParamInput] = Field(default_factory=lambda: [256])
    tile_mpp: List[BenchmarkParamInput] = Field(default_factory=lambda: [0.5])
    color_norm: Optional[List[BenchmarkParamInput | None]] = None

    # --- Feature extraction ---
    feature_extraction: List[BenchmarkParamInput] = Field(default_factory=list)

    # --- MIL / slide-level models ---
    mil: List[BenchmarkParamInput] = Field(default_factory=list)
    slide_level_models: List[str] = Field(default_factory=list)
    slide_aggregation: List[Literal["mean", "max", "mean_max"]] = Field(
        default_factory=lambda: ["mean"]
    )

    # --- Training components ---
    loss: List[BenchmarkParamInput] = Field(default_factory=list)
    activation_function: List[BenchmarkParamInput] = Field(default_factory=list)
    optimizer: List[BenchmarkParamInput] = Field(default_factory=list)

    # --- Slide retrieval ---
    search_strategy: Optional[List[BenchmarkParamInput]] = None
    retrieval_representation: Optional[List[BenchmarkParamInput]] = None

    # --- Hyperparameter grid (plain typed lists for grid sweep) ---
    scheduler: List[Literal["none", "reduce_on_plateau", "cosine"]] = Field(
        default_factory=list
    )
    batch_size: List[int] = Field(default_factory=lambda: [16])
    epochs: List[int] = Field(default_factory=list)
    lr: List[float] = Field(default_factory=list)
    weight_decay: List[float] = Field(default_factory=list)
    dropout_p: List[float] = Field(default_factory=list)
    bag_size: List[int] = Field(default_factory=list)
    z_dim: List[int] = Field(default_factory=list)
    encoder_layers: List[int] = Field(default_factory=list)
    k: List[int] = Field(default_factory=list)
    seeds: List[int] = Field(default_factory=lambda: [1, 2, 3])

    _FIELD_TYPES: ClassVar[dict[str, type[int] | type[float] | type[str]]] = {
        "tile_px": int,
        "tile_mpp": float,
        "feature_extraction": str,
        "color_norm": str,
        "mil": str,
        "loss": str,
        "activation_function": str,
        "optimizer": str,
        "search_strategy": str,
        "retrieval_representation": str,
    }

    @classmethod
    def _normalize_entry(
        cls,
        entry: BenchmarkParamInput,
        *,
        field_name: str,
    ) -> BenchmarkParamEntry:
        expected_type = cls._FIELD_TYPES[field_name]

        if isinstance(entry, dict):
            if len(entry) != 1:
                raise ValueError(
                    f"benchmark_parameters.{field_name} mapping entries must contain "
                    "exactly one value->params pair."
                )
            value, params = next(iter(entry.items()))
            if not isinstance(params, dict):
                raise ValueError(
                    f"benchmark_parameters.{field_name} params must be a mapping."
                )
        else:
            value = entry
            params = {}

        if field_name == "color_norm" and value is None:
            return BenchmarkParamEntry(value=None, hyperparams=dict(params))

        if expected_type is int and isinstance(value, bool):
            raise ValueError(
                f"benchmark_parameters.{field_name} values must be {expected_type.__name__}."
            )
        if expected_type is float and isinstance(value, bool):
            raise ValueError(
                f"benchmark_parameters.{field_name} values must be float."
            )
        if expected_type is int and not isinstance(value, int):
            raise ValueError(
                f"benchmark_parameters.{field_name} values must be int. Got {value!r}."
            )
        elif expected_type is float and not isinstance(value, (int, float)):
            raise ValueError(
                f"benchmark_parameters.{field_name} values must be float. Got {value!r}."
            )
        elif expected_type is str and not isinstance(value, str):
            raise ValueError(
                f"benchmark_parameters.{field_name} values must be str. Got {value!r}."
            )

        normalized_value = float(value) if expected_type is float else value
        return BenchmarkParamEntry(value=normalized_value, hyperparams=dict(params))

    @classmethod
    def _normalize_entries(
        cls,
        value: list[BenchmarkParamInput] | None,
        *,
        field_name: str,
    ) -> list[BenchmarkParamEntry]:
        if value is None:
            return []
        return [cls._normalize_entry(entry, field_name=field_name) for entry in value]

    def get_entries(self, field_name: str) -> list[BenchmarkParamEntry]:
        """Return normalized benchmark entries for one benchmark-parameter field."""
        if field_name not in self._FIELD_TYPES:
            raise AttributeError(f"benchmark_parameters has no field '{field_name}'")
        return self._normalize_entries(getattr(self, field_name, None), field_name=field_name)

    def get_values(self, field_name: str) -> list[BenchmarkScalar]:
        """Return only the base values for one benchmark-parameter field."""
        return [entry.value for entry in self.get_entries(field_name)]

    @model_validator(mode="after")
    def validate_strategy_hyperparams(self) -> "BenchmarkParameters":
        """Validate slide-retrieval hyperparameter names when provided."""
        self._validate_search_strategy_hyperparams()
        self._validate_retrieval_representation_hyperparams()
        return self

    def _validate_search_strategy_hyperparams(self) -> None:
        if not self.search_strategy:
            return
        from pathforge.slide_retrieval.search_strategies.registry import (
            get_search_strategy_hyperparams,
            import_search_strategy_modules,
            is_search_strategy_available,
        )
        import_search_strategy_modules()
        for entry in self.get_entries("search_strategy"):
            strategy_name = str(entry.value)
            if not is_search_strategy_available(strategy_name):
                raise ValueError(f"Search strategy '{strategy_name}' is not registered.")
            declared_params = get_search_strategy_hyperparams(strategy_name)
            self._validate_declared_hyperparams(
                field_name="search_strategy",
                value_name=strategy_name,
                provided_params=entry.hyperparams,
                declared_params=declared_params,
            )

    def _validate_retrieval_representation_hyperparams(self) -> None:
        if not self.retrieval_representation:
            return
        from pathforge.slide_retrieval.representation_strategies.registry import (
            get_representation_strategy_hyperparams,
            import_representation_strategy_modules,
            is_representation_strategy_available,
        )
        import_representation_strategy_modules()
        for entry in self.get_entries("retrieval_representation"):
            strategy_name = str(entry.value)
            if not is_representation_strategy_available(strategy_name):
                raise ValueError(
                    f"Retrieval representation '{strategy_name}' is not registered."
                )
            declared_params = get_representation_strategy_hyperparams(strategy_name)
            self._validate_declared_hyperparams(
                field_name="retrieval_representation",
                value_name=strategy_name,
                provided_params=entry.hyperparams,
                declared_params=declared_params,
            )

    @staticmethod
    def _validate_declared_hyperparams(
        *,
        field_name: str,
        value_name: str,
        provided_params: dict[str, Any],
        declared_params: dict[str, dict[str, Any]],
    ) -> None:
        unknown_params = sorted(set(provided_params) - set(declared_params))
        if unknown_params:
            raise ValueError(
                f"benchmark_parameters.{field_name} entry '{value_name}' defines "
                f"unknown hyperparams {unknown_params}. Allowed hyperparams: "
                f"{sorted(declared_params)}"
            )

    @field_validator("tile_px")
    @classmethod
    def validate_tile_px(cls, v: List[BenchmarkParamInput]) -> List[BenchmarkParamInput]:
        for entry in cls._normalize_entries(v, field_name="tile_px"):
            px = int(entry.value)
            if px % 2 != 0:
                raise ValueError(f"Invalid tile_px: {px}. Must be divisible by 2.")
        return v

    @field_validator("tile_mpp")
    @classmethod
    def validate_tile_mpp(cls, v: List[BenchmarkParamInput]) -> List[BenchmarkParamInput]:
        for entry in cls._normalize_entries(v, field_name="tile_mpp"):
            mpp = float(entry.value)
            if mpp <= 0:
                raise ValueError(f"Invalid tile_mpp: {mpp}. Must be > 0.")
        return v

    @field_validator("feature_extraction")
    @classmethod
    def validate_feature_extractors(
        cls,
        v: list[BenchmarkParamInput],
    ) -> list[BenchmarkParamInput]:
        populate_dynamic_registries()
        for entry in cls._normalize_entries(v, field_name="feature_extraction"):
            fe = str(entry.value)
            if not is_feature_extractor_available(fe):
                raise ValueError(
                    f"Feature extractor '{fe}' is not registered. "
                    f"Available feature extractors: {sorted(all_feature_extractor_names())}"
                )
        return v

    @field_validator("color_norm")
    @classmethod
    def validate_color_norm(
        cls,
        v: list[BenchmarkParamInput | None] | None,
    ) -> list[BenchmarkParamInput | None] | None:
        if v is None:
            return None
        allowed = {"reinhard", "macenko"}
        for entry in cls._normalize_entries(v, field_name="color_norm"):
            if entry.value is None:
                continue
            color_norm = str(entry.value).strip().lower()
            if color_norm not in allowed:
                raise ValueError(
                    f"Invalid color_norm '{entry.value}'. Allowed values: {sorted(allowed)}"
                )
        return v

    @field_validator("mil")
    @classmethod
    def validate_mil_models(cls, v: List[BenchmarkParamInput]) -> List[BenchmarkParamInput]:
        populate_dynamic_registries()
        for entry in cls._normalize_entries(v, field_name="mil"):
            model_name = str(entry.value)
            if not MODELS.is_available(model_name):
                raise ValueError(f"MIL model '{model_name}' not found in registry.")
            model_cls = MODELS.get(model_name)
            if isinstance(model_cls, type) and not issubclass(model_cls, MILModelBase):
                raise ValueError(f"Model '{model_name}' does not inherit from MILModelBase.")
        return v

    @field_validator("slide_level_models")
    @classmethod
    def validate_slide_level_models(cls, v: List[str]) -> List[str]:
        from pathforge.core.models.sklearn_slide import SLIDE_LEVEL_MODEL_NAMES

        populate_dynamic_registries()
        for model_name in v:
            in_torch_registry = MODELS.is_available(model_name)
            in_sklearn_names = model_name in SLIDE_LEVEL_MODEL_NAMES
            if not in_torch_registry and not in_sklearn_names:
                raise ValueError(
                    f"Slide-level model '{model_name}' not found. "
                    f"Known slide-level models: {sorted(SLIDE_LEVEL_MODEL_NAMES)}."
                )
        return v

    @field_validator("activation_function")
    @classmethod
    def validate_activations(
        cls,
        v: List[BenchmarkParamInput],
    ) -> List[BenchmarkParamInput]:
        valid_activations = {
            name
            for name, _ in inspect.getmembers(torch.nn.modules.activation, inspect.isclass)
        }
        for entry in cls._normalize_entries(v, field_name="activation_function"):
            act = str(entry.value)
            if act not in valid_activations and not hasattr(torch.nn, act):
                raise ValueError(f"Activation '{act}' not found in torch.nn.")
        return v

    @field_validator("optimizer")
    @classmethod
    def validate_optimizers(
        cls,
        v: List[BenchmarkParamInput],
    ) -> List[BenchmarkParamInput]:
        valid_optimizers = {
            name
            for name, _ in inspect.getmembers(torch.optim, inspect.isclass)
            if name != "Optimizer"
        }
        for entry in cls._normalize_entries(v, field_name="optimizer"):
            opt = str(entry.value)
            if opt not in valid_optimizers:
                raise ValueError(f"Optimizer '{opt}' not found in torch.optim.")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_sizes(cls, value: list[int]) -> list[int]:
        for batch_size in value:
            if batch_size <= 0:
                raise ValueError(f"Invalid batch_size: {batch_size}. Must be > 0.")
        return value

    @field_validator("epochs")
    @classmethod
    def validate_epochs(cls, value: list[int]) -> list[int]:
        for epochs in value:
            if epochs <= 0:
                raise ValueError(f"Invalid epochs: {epochs}. Must be > 0.")
        return value

    @field_validator("lr")
    @classmethod
    def validate_learning_rates(cls, value: list[float]) -> list[float]:
        for lr in value:
            if lr <= 0:
                raise ValueError(f"Invalid lr: {lr}. Must be > 0.")
        return value

    @field_validator("weight_decay")
    @classmethod
    def validate_weight_decay(cls, value: list[float]) -> list[float]:
        for weight_decay in value:
            if weight_decay < 0:
                raise ValueError(
                    f"Invalid weight_decay: {weight_decay}. Must be >= 0."
                )
        return value

    @field_validator("dropout_p")
    @classmethod
    def validate_dropouts(cls, value: list[float]) -> list[float]:
        for dropout in value:
            if dropout < 0.0 or dropout > 1.0:
                raise ValueError(
                    f"Invalid dropout_p: {dropout}. Must be inside [0, 1]."
                )
        return value

    @field_validator("bag_size", "z_dim", "encoder_layers", "k")
    @classmethod
    def validate_positive_int_lists(cls, value: list[int], info: Any) -> list[int]:
        for item in value:
            if item <= 0:
                raise ValueError(
                    f"Invalid {info.field_name}: {item}. Must be > 0."
                )
        return value


# ---------------------------------------------------------------------------
# Top-Level Config
# ---------------------------------------------------------------------------


class Config(BaseModel):
    """Top-level configuration object."""

    experiment: ExperimentConfig
    classification: ClassificationConfig | None = None
    slide_retrieval: SlideRetrievalConfig | None = None
    mil: MILConfig = Field(default_factory=MILConfig)
    slide_processing: SlideProcessingConfig = Field(default_factory=SlideProcessingConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    datasets: List[DatasetEntry] = Field(default_factory=list)
    benchmark_parameters: BenchmarkParameters = Field(default_factory=BenchmarkParameters)

    weights_dir: str = "./pretrained_weights"
    hf_key: Optional[str] = None

    @model_validator(mode="after")
    def validate_backend_constraints(self) -> "Config":
        """Ensures selected optional backends are installed before runtime uses them."""
        backend = self.slide_processing.backend
        fe_list = self.benchmark_parameters.get_values("feature_extraction")

        for fe in fe_list:
            if fe in LAZYSLIDE_MODEL_NAMES and backend != "lazyslide":
                raise ValueError(
                    f"Feature extractor '{fe}' requires 'lazyslide' backend. "
                    f"Current backend: '{backend}'."
                )

        if (
            self.experiment.mode != "feature_extraction"
            and self.mil.backend == "torchmil"
        ):
            if not is_torchmil_available():
                raise RuntimeError(
                    "MIL backend 'torchmil' selected, but 'torchmil' is not installed. "
                    "Install torchmil or set mil.backend='native'."
                )
            if not self.mil.torchmil_model:
                raise ValueError(
                    "mil.torchmil_model is required when mil.backend='torchmil'."
                )

        if (
            self.experiment.mode != "feature_extraction"
            and self.mil.backend == "mil-lab"
        ):
            if not is_mil_lab_available():
                raise RuntimeError(
                    "MIL backend 'mil-lab' selected, but 'MIL-Lab' is not installed. "
                    "Install MIL-Lab following its upstream README or set mil.backend to "
                    "'native' or 'torchmil'."
                )
            if not self.mil.mil_lab_model:
                raise ValueError(
                    "mil.mil_lab_model is required when mil.backend='mil-lab'."
                )

        if (
            self.explainability.heatmap_backend == "torchmil"
            and not is_torchmil_available()
        ):
            raise RuntimeError(
                "Explainability heatmap backend 'torchmil' selected, but 'torchmil' is not installed. "
                "Install torchmil or set explainability.heatmap_backend='native'."
            )

        task = self.experiment.task
        if (
            task == "classification"
            and self.metrics.classification_backend == "torchmetrics"
            and not is_torchmetrics_available()
        ):
            raise RuntimeError(
                "Classification metrics backend requires 'torchmetrics'. "
                "Install torchmetrics or choose another classification metrics backend."
            )

        if (
            task == "survival"
            and self.metrics.survival_continuous_backend == "torchsurv"
            and not is_torchsurv_available()
        ):
            raise RuntimeError(
                "Continuous survival backend requires 'torchsurv'. "
                "Install torchsurv or choose another survival backend."
            )

        if self.experiment.mode == "benchmark":
            prediction_level = self.experiment.prediction_level
            if prediction_level == "mil":
                if self.benchmark_parameters.slide_level_models:
                    raise ValueError(
                        "prediction_level='mil' is incompatible with "
                        "benchmark_parameters.slide_level_models. "
                        "Use prediction_level='slide' or remove slide_level_models."
                    )
                if (
                    self.experiment.task != "slide_retrieval"
                    and not self.benchmark_parameters.mil
                ):
                    raise ValueError(
                        "prediction_level='mil' requires at least one model in "
                        "benchmark_parameters.mil."
                    )
            elif prediction_level == "slide":
                if self.benchmark_parameters.mil:
                    raise ValueError(
                        "prediction_level='slide' is incompatible with "
                        "benchmark_parameters.mil. "
                        "Use prediction_level='mil' or remove MIL models."
                    )
                if not self.benchmark_parameters.slide_level_models:
                    self.benchmark_parameters.slide_level_models = ["SlideVectorMLP"]

        task_metrics = set(self.metrics.metrics_for_task(task))
        allowed_monitor_names = task_metrics | MONITOR_FALLBACK_NAMES
        if self.mil.best_epoch_based_on not in allowed_monitor_names:
            raise ValueError(
                "mil.best_epoch_based_on must be compatible with the configured task. "
                f"Got {self.mil.best_epoch_based_on!r}; allowed values: {sorted(allowed_monitor_names)}."
            )
        if (
            self.experiment.mode == "optimization"
            and self.optimization.objective_metric not in allowed_monitor_names
        ):
            raise ValueError(
                "optimization.objective_metric must be compatible with the configured task. "
                f"Got {self.optimization.objective_metric!r}; allowed values: {sorted(allowed_monitor_names)}."
            )

        self._validate_dataset_use_semantics()
        self._validate_task_evaluation_metrics()
        self._validate_task_specific_config()

        return self

    def _validate_task_specific_config(self) -> None:
        task_name = self.experiment.task
        if task_name is None:
            return

        task_config_fields: dict[str, tuple[str, type[BaseModel]]] = {
            "classification": ("classification", ClassificationConfig),
            "slide_retrieval": ("slide_retrieval", SlideRetrievalConfig),
        }

        registration = task_config_fields.get(task_name)
        if registration is None:
            return

        field_name, model_cls = registration
        block_value = getattr(self, field_name)
        has_required_fields = any(
            field_info.is_required()
            for field_info in model_cls.model_fields.values()
        )

        if block_value is None:
            if has_required_fields:
                raise ValueError(
                    f"experiment.task='{task_name}' requires a top-level "
                    f"'{field_name}' config block."
                )
            setattr(self, field_name, model_cls())

    def _validate_dataset_use_semantics(self) -> None:
        if self.experiment.mode != "benchmark" or self.experiment.task is None:
            return

        from pathforge.core.tasks.registry import (
            get_task_allowed_dataset_uses,
            import_task_modules,
            is_task_available,
        )

        import_task_modules()
        if not is_task_available(self.experiment.task):
            return

        allowed_task_uses = get_task_allowed_dataset_uses(self.experiment.task)
        if allowed_task_uses is None:
            return

        allowed_uses = set(allowed_task_uses) | {"ignore"}
        invalid_uses = sorted(
            {
                ds_cfg.used_for
                for ds_cfg in self.datasets
                if ds_cfg.used_for not in allowed_uses
            }
        )
        if invalid_uses:
            raise ValueError(
                f"Invalid dataset used_for values for task '{self.experiment.task}': "
                f"{invalid_uses}. Allowed values: {sorted(allowed_uses)}"
            )

    def _validate_task_evaluation_metrics(self) -> None:
        if self.experiment.task is None:
            return
        if not self.evaluation.metrics:
            return
        if self.evaluation.label_column is None:
            raise ValueError(
                "evaluation.label_column is required when evaluation.metrics are set."
            )

        from pathforge.core.evaluation.registry import (
            import_evaluation_metric_modules,
            resolve_metric_request,
        )

        import_evaluation_metric_modules()
        for metric_name in self.evaluation.metrics:
            resolve_metric_request(
                task_name=str(self.experiment.task),
                raw_name=metric_name,
            )

        annotation_path = Path(self.experiment.annotation_file)
        if not annotation_path.is_file():
            return

        with annotation_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])

        if self.evaluation.label_column not in header:
            raise ValueError(
                f"evaluation.label_column '{self.evaluation.label_column}' was not "
                f"found in annotations header: {header}"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r") as f:
            data = yaml.safe_load(f) or {}

        data = resolve_external_dataset_sources(
            data,
            config_path=path.resolve(),
        )
        return cls.model_validate(data)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)
