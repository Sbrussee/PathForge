from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, List, Optional, Dict
import yaml
import torch
import inspect

# Pydantic Imports
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

# Internal Imports
from pathbench.utils.constants import TASK_TYPES, MODE_TYPES
from pathbench.utils.optional.torchmil import (
    is_torchmetrics_available,
    is_torchmil_available,
    is_torchsurv_available,
)
from pathbench.utils.optional.mil_lab import is_mil_lab_available
from pathbench.utils.registries import (
    MODELS,
    LAZYSLIDE_MODEL_NAMES,
    is_feature_extractor_available,
    all_feature_extractor_names,
    populate_dynamic_registries,
)
from pathbench.core.models.mil_base import MILModelBase
from pathbench.adapters.tcga_tools import resolve_external_dataset_sources

TaskType = Literal[tuple(TASK_TYPES)]
ModeType = Literal[tuple(MODE_TYPES)]

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

    # Execution settings consumed directly by trainers and pipeline policies.
    num_workers: int = Field(0, ge=0)
    split_technique: Literal["k-fold", "k-fold-stratified", "fixed"] = "k-fold"
    val_fraction: float = Field(0.1, gt=0, lt=1)

    # Task + Mode
    task: Optional[TaskType] = None
    mode: ModeType = "benchmark"
    prediction_level: Literal["mil", "slide"] = "mil"
    aggregation_level: Literal["slide", "patient"] = "slide"
    label_column: str = "category"
    slide_column: str = "slide"
    survival_time_column: Optional[str] = None
    survival_event_column: Optional[str] = None

    # Global behavior
    report: bool = False
    mixed_precision: bool = False

    # Optional reporting/evaluation selectors preserved for config compatibility.
    visualization: List[str] = Field(default_factory=list)
    evaluation: List[str] = Field(default_factory=list)
    custom_metrics: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_task_for_mode(self) -> "ExperimentConfig":
        """
        `task` may be omitted ONLY when mode == 'feature_extraction'.
        For all other modes, `task` is required.
        """
        if self.mode != "feature_extraction" and self.task is None:
            raise ValueError(
                "experiment.task is required unless mode == 'feature_extraction'."
            )
        return self


class MILConfig(BaseModel):
    """MIL Model specific settings."""

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
        ```yaml
        metrics:
          classification_backend: torchmetrics
          survival_continuous_backend: torchsurv
        ```
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


class SlideProcessingConfig(BaseModel):
    """Settings for slide processing backends."""

    backend: Literal["lazyslide", "openslide", "cucim"] = "lazyslide"
    save_tiles: bool = False
    segmentation_method: Optional[str] = None

    # Optional backend-specific slide quality-control filter payloads.
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


class DatasetEntry(BaseModel):
    """Definition of a dataset source."""

    name: str
    slides_dir: str
    artifacts_dir: str
    features_dir: Optional[str] = None
    tissue_annotations_dir: Optional[str] = None
    used_for: Literal["training", "testing", "validation", "ignore", "all"]


class BenchmarkParameters(BaseModel):
    """
    Grid search parameters.
    Pydantic validators enforce logic previously implemented manually.
    """

    tile_px: List[int] = Field(default_factory=lambda: [224])
    tile_mpp: List[float] = Field(default_factory=lambda: [0.5])
    feature_extraction: List[str] = Field(default_factory=list)
    mil: List[str] = Field(default_factory=list)
    slide_level_models: List[str] = Field(default_factory=list)
    slide_aggregation: List[Literal["mean", "max", "mean_max"]] = Field(
        default_factory=lambda: ["mean"]
    )
    loss: List[str] = Field(default_factory=list)
    activation_function: List[str] = Field(default_factory=list)
    optimizer: List[str] = Field(default_factory=list)
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

    @field_validator("tile_px")
    @classmethod
    def validate_tile_px(cls, v: List[int]) -> List[int]:
        for px in v:
            if px % 2 != 0:
                raise ValueError(f"Invalid tile_px: {px}. Must be divisible by 2.")
        return v

    @field_validator("tile_mpp")
    @classmethod
    def validate_tile_mpp(cls, v: List[float]) -> List[float]:
        for mpp in v:
            if mpp <= 0:
                raise ValueError(f"Invalid tile_mpp: {mpp}. Must be > 0.")
        return v

    @field_validator("feature_extraction")
    @classmethod
    def validate_feature_extractors(cls, v: list[str]) -> list[str]:
        populate_dynamic_registries()
        for fe in v:
            if not is_feature_extractor_available(fe):
                raise ValueError(
                    f"Feature extractor '{fe}' is not registered. "
                    f"Available feature extractors: {sorted(all_feature_extractor_names())}"
                )

        return v

    @field_validator("mil")
    @classmethod
    def validate_mil_models(cls, v: List[str]) -> List[str]:
        populate_dynamic_registries()
        for model_name in v:
            if not MODELS.is_available(model_name):
                raise ValueError(f"MIL model '{model_name}' not found in registry.")

            # Check Inheritance
            model_cls = MODELS.get(model_name)
            if isinstance(model_cls, type):
                if not issubclass(model_cls, MILModelBase):
                    raise ValueError(
                        f"Model '{model_name}' does not inherit from MILModelBase."
                    )
        return v

    @field_validator("slide_level_models")
    @classmethod
    def validate_slide_level_models(cls, v: List[str]) -> List[str]:
        from pathbench.core.models.sklearn_slide import SLIDE_LEVEL_MODEL_NAMES

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
    def validate_activations(cls, v: List[str]) -> List[str]:
        valid_activations = {
            name
            for name, _ in inspect.getmembers(
                torch.nn.modules.activation, inspect.isclass
            )
        }
        for act in v:
            if act not in valid_activations and not hasattr(torch.nn, act):
                raise ValueError(f"Activation '{act}' not found in torch.nn.")
        return v

    @field_validator("optimizer")
    @classmethod
    def validate_optimizers(cls, v: List[str]) -> List[str]:
        valid_optimizers = {
            name
            for name, _ in inspect.getmembers(torch.optim, inspect.isclass)
            if name != "Optimizer"
        }
        for opt in v:
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
    def validate_positive_int_lists(
        cls, value: list[int], info: Any
    ) -> list[int]:
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
    """
    Top-level configuration object.
    """

    experiment: ExperimentConfig
    mil: MILConfig = Field(default_factory=MILConfig)
    slide_processing: SlideProcessingConfig = Field(
        default_factory=SlideProcessingConfig
    )
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    datasets: List[DatasetEntry] = Field(default_factory=list)
    benchmark_parameters: BenchmarkParameters = Field(
        default_factory=BenchmarkParameters
    )

    # Backward-compatible global paths/tokens used by a few feature backends.
    weights_dir: str = "./pretrained_weights"
    hf_key: Optional[str] = None

    @model_validator(mode="after")
    def validate_backend_constraints(self) -> "Config":
        """
        Ensures selected optional backends are installed before runtime paths use them.
        """
        backend = self.slide_processing.backend
        fe_list = self.benchmark_parameters.feature_extraction

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
        ):
            if not is_torchmetrics_available():
                raise RuntimeError(
                    "Classification metrics backend requires 'torchmetrics'. "
                    "Install torchmetrics or choose another classification metrics backend."
                )

        if (
            task == "survival"
            and self.metrics.survival_continuous_backend == "torchsurv"
        ):
            if not is_torchsurv_available():
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
                if not self.benchmark_parameters.mil:
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

        return self

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
            # model_dump is Pydantic v2 for to_dict
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)
