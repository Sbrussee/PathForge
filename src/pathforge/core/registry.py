"""Backward-compatible registry imports for legacy core modules."""

from __future__ import annotations

from pathforge.utils.registries import (
    AUGMENTATION_METHODS,
    DATASETS,
    EXPLAINERS,
    FEATURE_EXTRACTORS,
    LOSSES,
    MODELS,
    NORMALIZERS,
    REGISTRIES,
    SLIDE_PROCESSORS,
    SURVIVAL_LOSSES,
    SURVIVAL_METRICS,
    TASKS,
    TRAINERS,
)

__all__ = [
    "AUGMENTATION_METHODS",
    "DATASETS",
    "EXPLAINERS",
    "FEATURE_EXTRACTORS",
    "LOSSES",
    "MODELS",
    "NORMALIZERS",
    "REGISTRIES",
    "SLIDE_PROCESSORS",
    "SURVIVAL_LOSSES",
    "SURVIVAL_METRICS",
    "TASKS",
    "TRAINERS",
]
