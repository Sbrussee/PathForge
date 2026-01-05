from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, List, Optional, Dict, Union
import yaml
import torch
import inspect

# Pydantic Imports
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo

# Internal Imports
from pathbench.utils.constants import TASK_TYPES, MODE_TYPES, EXPERIMENTS_DIR
from pathbench.utils.registries import MODELS, FEATURE_EXTRACTORS, LAZYSLIDE_MODEL_NAMES
from pathbench.core.models.mil_base import MILModelBase

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
    
    # Execution
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

class BagDatasetConfig(BaseModel):
    """Configuration for MIL bag datasets."""
    id_column: str = "slide_id"
    label_column: str = "label"
    dataset_column: str = "dataset"
    feature_path_column: Optional[str] = None
    feature_extension: str = ".pt"
    allow_missing_features: bool = False
    drop_missing_labels: bool = True
    label_dtype: Literal["int", "float", "str"] = "int"
    max_instances: Optional[int] = Field(None, gt=0)
    sampling_strategy: Literal["random", "first"] = "random"
    random_seed: int = 0
    return_slide_id: bool = False


class EvaluationConfig(BaseModel):
    """Evaluation settings for MIL workflows."""
    metrics: List[str] = Field(default_factory=lambda: ["accuracy"])
    average: Literal["macro", "micro", "weighted"] = "macro"
    positive_label: int = 1


class SlideProcessingConfig(BaseModel):
    """Settings for slide processing backends."""
    backend: Literal["lazyslide", "openslide", "cucim"] = "lazyslide"
    save_tiles: bool = False
    segmentation_method: str = "otsu"
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
    slide_path: str
    tiles_path: Optional[str] = None
    roi_path: Optional[str] = None
    features_path: Optional[str] = None
    used_for: Literal["training", "testing", "validation", "ignore", "all"]


class BenchmarkParameters(BaseModel):
    """
    Grid search parameters. 
    Pydantic validators enforce logic previously implemented manually.
    """
    tile_px: List[int] = Field(default_factory=lambda: [256])
    tile_mpp: List[float] = Field(default_factory=lambda: [0.5])
    feature_extraction: List[str] = Field(default_factory=list)
    mil: List[str] = Field(default_factory=list)
    loss: List[str] = Field(default_factory=list)
    activation_function: List[str] = Field(default_factory=list)
    optimizer: List[str] = Field(default_factory=list)

    @field_validator('tile_px')
    @classmethod
    def validate_tile_px(cls, v: List[int]) -> List[int]:
        for px in v:
            if px % 2 != 0:
                raise ValueError(f"Invalid tile_px: {px}. Must be divisible by 2.")
        return v

    @field_validator('tile_mpp')
    @classmethod
    def validate_tile_mpp(cls, v: List[float]) -> List[float]:
        for mpp in v:
            if mpp <= 0:
                raise ValueError(f"Invalid tile_mpp: {mpp}. Must be > 0.")
        return v

    @field_validator('feature_extraction')
    @classmethod
    def validate_feature_extractors(cls, v: List[str]) -> List[str]:
        for fe in v:
            if not FEATURE_EXTRACTORS.is_available(fe):
                raise ValueError(f"Feature extractor '{fe}' is not registered in timm/lazyslide.")
        return v

    @field_validator('mil')
    @classmethod
    def validate_mil_models(cls, v: List[str]) -> List[str]:
        for model_name in v:
            if not MODELS.is_available(model_name):
                raise ValueError(f"MIL model '{model_name}' not found in registry.")
            
            # Check Inheritance
            model_cls = MODELS.get(model_name)
            # If the registry returns a class, we check subclass
            if isinstance(model_cls, type):
                if not issubclass(model_cls, MILModelBase):
                    raise ValueError(f"Model '{model_name}' does not inherit from MILModelBase.")
        return v

    @field_validator('activation_function')
    @classmethod
    def validate_activations(cls, v: List[str]) -> List[str]:
        valid_activations = {name for name, _ in inspect.getmembers(torch.nn.modules.activation, inspect.isclass)}
        for act in v:
            if act not in valid_activations and not hasattr(torch.nn, act):
                raise ValueError(f"Activation '{act}' not found in torch.nn.")
        return v

    @field_validator('optimizer')
    @classmethod
    def validate_optimizers(cls, v: List[str]) -> List[str]:
        valid_optimizers = {name for name, _ in inspect.getmembers(torch.optim, inspect.isclass) if name != "Optimizer"}
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

    #TODO: Based on specified mode, only load the relevant sections to save memory.
    experiment: ExperimentConfig
    mil: MILConfig = Field(default_factory=MILConfig)
    bag_dataset = Field(default_factory=BagDatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    slide_processing: SlideProcessingConfig = Field(default_factory=SlideProcessingConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    datasets: List[DatasetEntry] = Field(default_factory=list)
    benchmark_parameters: BenchmarkParameters = Field(default_factory=BenchmarkParameters)
    
    weights_dir: str = "./pretrained_weights"
    hf_key: Optional[str] = None

    @model_validator(mode='after')
    def validate_backend_constraints(self) -> "Config":
        """
        Ensures that if a Lazyslide-specific model is selected, 
        the backend is set to 'lazyslide'.
        """
        backend = self.slide_processing.backend
        fe_list = self.benchmark_parameters.feature_extraction
        
        for fe in fe_list:
            if fe in LAZYSLIDE_MODEL_NAMES and backend != "lazyslide":
                raise ValueError(
                    f"Feature extractor '{fe}' requires 'lazyslide' backend. "
                    f"Current backend: '{backend}'."
                )
        
        if self.experiment.mode == "benchmark" and not self.benchmark_parameters.mil:
             raise ValueError("Mode is 'benchmark' but no MIL models specified in benchmark_parameters.")
             
        return self

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