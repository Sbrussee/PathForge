# pathbench/utils/constants.py
# This file contains constant values used throughout the PathBench project.

#------------ Default Directories ------------#
DEFAULT_WEIGHTS_DIR = "./pretrained_weights"

#------------ Task and Mode Types ------------#
TASK_TYPES = ["classification", "regression", "survival", "survival_discrete"]
MODE_TYPES = ["benchmark", "optimization", "feature_extraction"]


#------------ Registry Keys ------------#
REGISTRY_DATASETS = "datasets"
REGISTRY_MODELS = "models"
REGISTRY_LOSSES = "losses"
REGISTRY_TASKS = "tasks"
REGISTRY_EXPLAINERS = "explainers"
REGISTRY_FEATURE_EXTRACTORS = "feature_extractors"
REGISTRY_NORMALIZERS = "normalizers"
REGISTRY_AUGMENTATION_METHODS = "augmentation_methods"

#------------ Directory Names ------------#
EXPERIMENTS_DIR = "experiments"
LOGS_DIR = "logs"


#------------ Slide Formats ------------#
OPENSLIDE_SLIDE_FORMATS = [".svs", ".tif", ".dcm", ".ndpi", ".vms", ".vmu",
                           ".scn", ".mrxs", ".tiff", ".svslide", ".bif", ".czi"] # for openslide, pyvips backends
TIFFSLIDE_SLIDE_FORMATS = [".tiff", ".tif", ".svs", ".ndpi", ".scn"] # for tiffslide backend

#NOTE: Preferably use tiffslide for .tiff/.tif, .svs files instead of openslide for better performance

CUCIM_SLIDE_FORMATS = [".svs", ".tiff", ".tif"] # for cucim backend

# Unified list for generic slide discovery.
SLIDE_FILE_FORMATS = sorted({*OPENSLIDE_SLIDE_FORMATS, *TIFFSLIDE_SLIDE_FORMATS, *CUCIM_SLIDE_FORMATS})