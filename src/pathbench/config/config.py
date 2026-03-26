from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, ClassVar, Literal, List, Optional, Dict
import yaml
import torch
import inspect

# Pydantic Imports
from pydantic import BaseModel, Field, field_validator, model_validator

# Internal Imports
from pathbench.utils.constants import TASK_TYPES, MODE_TYPES, EXPERIMENTS_DIR, AGGREGATION_LEVELS
from pathbench.utils.registries import MODELS, FEATURE_EXTRACTORS, LAZYSLIDE_MODEL_NAMES, is_feature_extractor_available, all_feature_extractor_names
from pathbench.core.models.mil_base import MILModelBase

TaskType = Literal[tuple(TASK_TYPES)]
ModeType = Literal[tuple(MODE_TYPES)]
AggregationLevel = Literal[tuple(AGGREGATION_LEVELS)]

BenchmarkScalar = int | float | str
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
    aggregation_level: AggregationLevel = "slide"

    # Global behavior
    report: bool = False
    mixed_precision: bool = False
    
    #TODO: Does this make sense to be here? 
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


class SlideProcessingConfig(BaseModel):
    """Settings for slide processing backends."""
    backend: Literal["lazyslide", "openslide", "cucim"] = "lazyslide"
    save_tiles: bool = False
    segmentation_method: Optional[str] = None

    #TODO: Do we need this option below? 
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
    tile_px: List[BenchmarkParamInput] = Field(default_factory=lambda: [256])
    tile_mpp: List[BenchmarkParamInput] = Field(default_factory=lambda: [0.5])
    feature_extraction: List[BenchmarkParamInput] = Field(default_factory=list)
    normalization: Optional[List[BenchmarkParamInput]] = None  # TODO: Do we want to keep this and is it available in lazyslide?
    mil: List[BenchmarkParamInput] = Field(default_factory=list)
    loss: List[BenchmarkParamInput] = Field(default_factory=list)
    activation_function: List[BenchmarkParamInput] = Field(default_factory=list)
    optimizer: List[BenchmarkParamInput] = Field(default_factory=list)
    search_strategy: Optional[List[BenchmarkParamInput]] = None
    retrieval_representation: Optional[List[BenchmarkParamInput]] = None

    _FIELD_TYPES: ClassVar[dict[str, type[int] | type[float] | type[str]]] = {
        "tile_px": int,
        "tile_mpp": float,
        "feature_extraction": str,
        "normalization": str,
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
        return BenchmarkParamEntry(
            value=normalized_value,
            hyperparams=dict(params),
        )

    @classmethod
    def _normalize_entries(
        cls,
        value: list[BenchmarkParamInput] | None,
        *,
        field_name: str,
    ) -> list[BenchmarkParamEntry]:
        if value is None:
            return []
        return [
            cls._normalize_entry(entry, field_name=field_name)
            for entry in value
        ]

    def get_entries(self, field_name: str) -> list[BenchmarkParamEntry]:
        """Return normalized benchmark entries for one benchmark-parameter field."""
        if field_name not in self._FIELD_TYPES:
            raise AttributeError(f"benchmark_parameters has no field '{field_name}'")

        return self._normalize_entries(
            getattr(self, field_name, None),
            field_name=field_name,
        )

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

        from pathbench.slide_retrieval.search_strategies.registry import (
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

        from pathbench.slide_retrieval.representation_strategies.registry import (
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

    @field_validator('tile_px')
    @classmethod
    def validate_tile_px(cls, v: List[BenchmarkParamInput]) -> List[BenchmarkParamInput]:
        for entry in cls._normalize_entries(v, field_name="tile_px"):
            px = int(entry.value)
            if px % 2 != 0:
                raise ValueError(f"Invalid tile_px: {px}. Must be divisible by 2.")
        return v

    @field_validator('tile_mpp')
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
        for entry in cls._normalize_entries(v, field_name="feature_extraction"):
            fe = str(entry.value)
            if not is_feature_extractor_available(fe):
                raise ValueError(
                    f"Feature extractor '{fe}' is not registered. "
                    f"Available feature extractors: {sorted(all_feature_extractor_names())}"
                )
        return v

    @field_validator('mil')
    @classmethod
    def validate_mil_models(cls, v: List[BenchmarkParamInput]) -> List[BenchmarkParamInput]:
        for entry in cls._normalize_entries(v, field_name="mil"):
            model_name = str(entry.value)
            if not MODELS.is_available(model_name):
                raise ValueError(f"MIL model '{model_name}' not found in registry.")

            model_cls = MODELS.get(model_name)
            if isinstance(model_cls, type) and not issubclass(model_cls, MILModelBase):
                raise ValueError(f"Model '{model_name}' does not inherit from MILModelBase.")
        return v

    @field_validator('activation_function')
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

    @field_validator('optimizer')
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


# ---------------------------------------------------------------------------
# Top-Level Config
# ---------------------------------------------------------------------------

class Config(BaseModel):
    """
    Top-level configuration object. 
    """
    experiment: ExperimentConfig
    mil: MILConfig = Field(default_factory=MILConfig)
    slide_processing: SlideProcessingConfig = Field(default_factory=SlideProcessingConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    datasets: List[DatasetEntry] = Field(default_factory=list)
    benchmark_parameters: BenchmarkParameters = Field(default_factory=BenchmarkParameters)
    
    #TODO: Do we still need these? 
    weights_dir: str = "./pretrained_weights"
    hf_key: Optional[str] = None

    @model_validator(mode='after')
    def validate_backend_constraints(self) -> "Config":
        """
        Ensures that if a Lazyslide-specific model is selected, 
        the backend is set to 'lazyslide'.
        """
        backend = self.slide_processing.backend
        fe_list = self.benchmark_parameters.get_values("feature_extraction")
        
        for fe in fe_list:
            if fe in LAZYSLIDE_MODEL_NAMES and backend != "lazyslide":
                raise ValueError(
                    f"Feature extractor '{fe}' requires 'lazyslide' backend. "
                    f"Current backend: '{backend}'."
                )
        
        if (
            self.experiment.mode == "benchmark"
            and self.experiment.task != "slide_retrieval"
            and not self.benchmark_parameters.mil
        ):
            raise ValueError(
                "Mode is 'benchmark' but no MIL models specified in benchmark_parameters."
            )

        self._validate_dataset_use_semantics()
        self._validate_task_evaluation_metrics()
             
        return self

    def _validate_dataset_use_semantics(self) -> None:
        if self.experiment.mode != "benchmark" or self.experiment.task is None:
            return

        from pathbench.benchmarking.registry import (
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
        if self.experiment.mode != "benchmark":
            return

        if self.experiment.task != "slide_retrieval":
            return

        if not self.experiment.evaluation:
            return

        from pathbench.slide_retrieval.validation.registry import (
            import_validation_metric_modules,
            is_validation_metric_available,
            parse_validation_metric_name,
        )

        import_validation_metric_modules()
        for metric_name in self.experiment.evaluation:
            request = parse_validation_metric_name(metric_name)
            if not is_validation_metric_available(request.registry_key):
                raise ValueError(
                    f"Slide-retrieval evaluation metric '{metric_name}' is not "
                    f"registered. Expected one of the registered metric families "
                    f"matching '<metric>_at_<k>'."
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
            
        return cls.model_validate(data)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            # model_dump is Pydantic v2 for to_dict
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)
