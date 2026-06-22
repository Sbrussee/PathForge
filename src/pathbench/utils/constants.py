"""Shared constants used across PathBench."""

TASK_TYPES = [
    "classification",
    "regression",
    "survival",
    "survival_discrete",
    "slide_retrieval",
]
MODE_TYPES = ["benchmark", "optimization", "feature_extraction"]

AGGREGATION_LEVELS = ["slide", "case", "patient"]

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

CATEGORY_COL = "category"
SLIDE_ID_COL = "slide"
CASE_ID_COL = "case"
PATIENT_ID_COL = "patient"
DATASET_COL = "dataset"
CENTER_COL = "center"

LZS_ABS_MPP_TOL = 1e-3  # absolute tolerance in um/px
LZS_REL_MPP_TOL = 1e-2  # relative tolerance as sanity check (1%)
# ``WSIDataset`` discovers slide files before a concrete slide backend is
# instantiated, so it accepts the union of suffixes currently supported by the
# Lazyslide/WSIData ingestion stack used by PathBench.
SLIDE_FILE_FORMATS = (".svs", ".ndpi", ".tiff", ".tif", ".mrxs")
