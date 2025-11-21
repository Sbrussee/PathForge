from pathbench.utils.registry import Registry
from pathbench.core.base import CoreRegistries

REGISTRIES = CoreRegistries(
    datasets=Registry(),
    models=Registry(),
    losses=Registry(),
    tasks=Registry(),
    explainers=Registry(),
    feature_extractors=Registry(),
    normalizers=Registry(),
    augmentation_methods=Registry(),
)

DATASETS = REGISTRIES.datasets
MODELS = REGISTRIES.models
LOSSES = REGISTRIES.losses
TASKS = REGISTRIES.tasks
EXPLAINERS = REGISTRIES.explainers
FEATURE_EXTRACTORS = REGISTRIES.feature_extractors
NORMALIZERS = REGISTRIES.normalizers
AUGMENTATION_METHODS = REGISTRIES.augmentation_methods