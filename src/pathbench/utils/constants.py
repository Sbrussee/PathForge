"""Shared constants used across PathBench."""

DEFAULT_WEIGHTS_DIR = "./pretrained_weights"

TASK_TYPES = ["classification", "regression", "survival", "survival_discrete"]
MODE_TYPES = ["benchmark", "optimization", "feature_extraction"]

REGISTRY_DATASETS = "datasets"
REGISTRY_MODELS = "models"
REGISTRY_LOSSES = "losses"
REGISTRY_TASKS = "tasks"
REGISTRY_EXPLAINERS = "explainers"
REGISTRY_FEATURE_EXTRACTORS = "feature_extractors"
REGISTRY_NORMALIZERS = "normalizers"
REGISTRY_AUGMENTATION_METHODS = "augmentation_methods"

EXPERIMENTS_DIR = "experiments"
LOGS_DIR = "logs"

# ``WSIDataset`` discovers slide files before a concrete slide backend is
# instantiated, so it accepts the union of suffixes currently supported by the
# Lazyslide/WSIData ingestion stack used by PathBench.
SLIDE_FILE_FORMATS = (".svs", ".ndpi", ".tiff", ".tif", ".mrxs")
