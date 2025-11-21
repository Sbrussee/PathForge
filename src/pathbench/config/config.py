from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Sequence

import yaml

from pathbench.config.base import ConfigBase
from pathbench.utils.constants import TASK_TYPES, MODE_TYPES


TaskType = Literal[tuple(TASK_TYPES)]
ModeType = Literal[tuple(MODE_TYPES)]

# ---------------------------------------------------------------------------
# 1) Split configs
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExperimentConfig:
    """
    General, universal experiment settings (independent of MIL details).
    """

    project_name: str
    annotation_file: str

    # execution / data split
    num_workers: int = 0
    split_technique: Literal["k-fold", "k-fold-stratified", "fixed"] = "k-fold"
    val_fraction: float = 0.1

    # task + mode
    task: TaskType = "classification"
    mode: ModeType = "benchmark"
    aggregation_level: Literal["slide", "patient"] = "slide"

    # global behaviour
    report: bool = False
    mixed_precision: bool = False

    # optional evaluation / logging knobs
    visualization: list[str] = field(default_factory=list)
    evaluation: list[str] = field(default_factory=list)
    custom_metrics: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MILConfig:
    """
    Settings related to MIL models and training.
    """

    # data balancing / weighting
    balancing: str | None = None
    class_weighting: bool = False

    # training loop
    epochs: int = 5
    best_epoch_based_on: str = "val_loss"
    batch_size: int = 32

    # MIL architecture hyperparameters
    bag_size: int = 512
    encoder_layers: int = 1
    z_dim: int = 256
    dropout_p: float = 0.1
    k: int = 2  # for top-k / attention pooling etc.

    # pipeline flags
    skip_extracted: bool = True
    skip_feature_extraction: bool = True


@dataclass(slots=True)
class SlideProcessingConfig:
    """
    Settings for slide/tile processing before MIL training.
    """
    
    # Backend framework
    backend: Literal["lazyslide"] = "lazyslide"

    # whether to persist generated tiles on disk
    save_tiles: bool = False

    # QC / filtering
    qc: list[str] = field(default_factory=list)
    qc_filters: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 2) Existing “split” configs
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class OptimizationConfig:
    study_name: str = "study"
    load_study: bool = False
    objective_metric: str = "balanced_accuracy"
    objective_mode: Literal["max", "min"] = "max"
    objective_dataset: Literal["val", "test"] = "val"
    sampler: str = "TPESampler"
    trials: int = 100
    pruner: str | None = "HyperbandPruner"


@dataclass(slots=True)
class DatasetEntry:
    name: str
    slide_path: str
    tfrecord_path: str
    tile_path: str
    used_for: Literal["training", "testing", "validation"]


@dataclass(slots=True)
class BenchmarkParameters:
    tile_px: list[int] = field(default_factory=lambda: [256])
    tile_um: list[str] = field(default_factory=lambda: ["20x"])
    feature_extraction: list[str] = field(default_factory=list)
    mil: list[str] = field(default_factory=list)
    loss: list[str] = field(default_factory=list)
    activation_function: list[str] = field(default_factory=list)
    optimizer: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 3) Top-level Config implementing ConfigBase
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Config(ConfigBase):
    """
    Top-level configuration object used by the rest of the framework.

    Layout in YAML (new style) is expected to look roughly like:

    experiment:
      project_name: ...
      annotation_file: ...
      ...
    mil:
      epochs: 10
      ...
    slide_processing:
      save_tiles: false
      ...
    optimization:
      ...
    datasets:
      - name: cohort1
        slide_path: ...
        ...
    benchmark_parameters:
      ...

    weights_dir: ./pretrained_weights
    hf_key: null
    """

    experiment: ExperimentConfig
    mil: MILConfig = field(default_factory=MILConfig)
    slide_processing: SlideProcessingConfig = field(default_factory=SlideProcessingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    datasets: list[DatasetEntry] = field(default_factory=list)
    benchmark_parameters: BenchmarkParameters = field(default_factory=BenchmarkParameters)
    weights_dir: str = "./pretrained_weights"
    hf_key: str | None = None

    # ---- ConfigBase implementation --------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        """Create a Config from a nested dict (e.g. YAML-loaded)."""
        experiment = ExperimentConfig(**data["experiment"])

        mil = MILConfig(**data.get("mil", {}))
        slide_processing = SlideProcessingConfig(**data.get("slide_processing", {}))
        optimization = OptimizationConfig(**data.get("optimization", {}))

        datasets = [DatasetEntry(**d) for d in data.get("datasets", [])]

        benchmark_parameters = BenchmarkParameters(
            **data.get("benchmark_parameters", {})
        )

        return cls(
            experiment=experiment,
            mil=mil,
            slide_processing=slide_processing,
            optimization=optimization,
            datasets=datasets,
            benchmark_parameters=benchmark_parameters,
            weights_dir=data.get("weights_dir", "./pretrained_weights"),
            hf_key=data.get("hf_key", None),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config (and sub-configs) to a YAML-friendly dict."""
        return {
            "experiment": asdict(self.experiment),
            "mil": asdict(self.mil),
            "slide_processing": asdict(self.slide_processing),
            "optimization": asdict(self.optimization),
            "datasets": [asdict(d) for d in self.datasets],
            "benchmark_parameters": asdict(self.benchmark_parameters),
            "weights_dir": self.weights_dir,
            "hf_key": self.hf_key,
        }

    # Convenience aliases if you prefer the old name
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Shortcut alias around ConfigBase.load_yaml with correct type."""
        from pathlib import Path as _Path
        path = _Path(path)
        with path.open("r") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls.from_dict(data)

        assert cfg.experiment.mode in {"benchmark", "optimization", "feature_extraction"}
        return cfg
