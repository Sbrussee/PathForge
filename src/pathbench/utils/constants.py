# pathbench/utils/constants.py
# This file contains constant values used throughout the PathBench project.

DEFAULT_WEIGHTS_DIR = "./pretrained_weights"

TASK_TYPES = ["classification", "regression", "survival", "survival_discrete"]
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

SLIDE_FILE_FORMATS = [".svs", ".ndpi", ".tiff", ".tif", ".mrxs"] #TODO: Base on backends? 
