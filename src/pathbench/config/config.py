from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, List, Optional, Dict
import yaml
import torch
import inspect

# Pydantic Imports
from pydantic import BaseModel, Field, field_validator, model_validator

# Internal Imports
from pathbench.utils.constants import TASK_TYPES, MODE_TYPES
from pathbench.utils.optional.torchmil import (
    is_torchmetrics_available,
    is_torchmil_available,
    is_torchsurv_available,
)
from pathbench.utils.registries import (
    MODELS,
    LAZYSLIDE_MODEL_NAMES,
    is_feature_extractor_available,
    all_feature_extractor_names,
)
from pathbench.core.models.mil_base import MILModelBase
from pathbench.adapters.tcga_tools import resolve_external_dataset_sources

TaskType = Literal[tuple(TASK_TYPES)]
ModeType = Literal[tuple(MODE_TYPES)]

# ---------------------------------------------------------------------------
# Config Sections
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Universal experiment settings."""

    project_name: str
    annotation_file: str
    project_root: str | None = None

    # Execution #TODO: Is this part of the experiment config, we do not use it in the experiment class
    num_workers: int = Field(0, ge=0)
    split_technique: Literal["k-fold", "k-fold-stratified", "fixed"] = "k-fold"
    val_fraction: float = Field(0.1, gt=0, lt=1)

    # Task + Mode
    task: Optional[TaskType] = None
    mode: ModeType = "benchmark"
    aggregation_level: Literal["slide", "patient"] = "slide"

    # Global behavior
    report: bool = False
    mixed_precision: bool = False

    # TODO: Does this make sense to be here?
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

    backend: Literal["native", "torchmil"] = "native"
    torchmil_model: Optional[str] = None
    torchmil_model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    use_torchmil_collate: bool = True

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


class SlideProcessingConfig(BaseModel):
    """Settings for slide processing backends."""

    backend: Literal["lazyslide", "openslide", "cucim"] = "lazyslide"
    save_tiles: bool = False
    segmentation_method: Optional[str] = None

    # TODO: Do we need this option below?
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


class DatasetEntry(BaseModel):
    """Definition of a dataset source."""

    name: str
    slides_dir: str
    artifacts_dir: str
    tissue_annotations_dir: Optional[str] = None
    used_for: Literal["training", "testing", "validation", "ignore", "all"]


class BenchmarkParameters(BaseModel):
    """
    Grid search parameters.
    Pydantic validators enforce logic previously implemented manually.
    """

    tile_px: List[int] = Field(default_factory=lambda: [256])
    tile_mpp: List[float] = Field(default_factory=lambda: [0.5])
    feature_extraction: List[str] = Field(default_factory=list)
    normalization: Optional[List[str]] = (
        None  # TODO: Do we want to keep this and is it available in lazyslide?
    )
    mil: List[str] = Field(default_factory=list)
    loss: List[str] = Field(default_factory=list)
    activation_function: List[str] = Field(default_factory=list)
    optimizer: List[str] = Field(default_factory=list)

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

    # TODO: Do we still need these?
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

        if self.experiment.mode == "benchmark" and not self.benchmark_parameters.mil:
            raise ValueError(
                "Mode is 'benchmark' but no MIL models specified in benchmark_parameters."
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
