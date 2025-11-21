# src/pathbench/utils/registries.py
from __future__ import annotations
import torch
import timm
try:
    import lazyslide as zs
except ImportError:
    zs = None

from pathbench.utils.registry import Registry
from pathbench.core.base import CoreRegistries

# Define Registries
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

# Shortcuts
DATASETS = REGISTRIES.datasets
MODELS = REGISTRIES.models
LOSSES = REGISTRIES.losses
TASKS = REGISTRIES.tasks
EXPLAINERS = REGISTRIES.explainers
FEATURE_EXTRACTORS = REGISTRIES.feature_extractors
NORMALIZERS = REGISTRIES.normalizers
AUGMENTATION_METHODS = REGISTRIES.augmentation_methods

# Infrastructure
SLIDE_PROCESSORS = Registry()
TRAINERS = Registry()

# Track Lazyslide-specific models for validation
LAZYSLIDE_MODEL_NAMES = set()

def populate_dynamic_registries():
    """
    Populates registries with dynamic entries from external libraries.
    """
    # 1. Register timm models
    for model_name in timm.list_models():
        if not FEATURE_EXTRACTORS.is_available(model_name):
            @FEATURE_EXTRACTORS.register(model_name)
            def _timm_factory(name=model_name, pretrained=True, **kwargs):
                return timm.create_model(name, pretrained=pretrained, **kwargs)

    # 2. Register lazyslide models (only if installed)
    if zs is not None:
        for model_name in zs.models.list_models():
            LAZYSLIDE_MODEL_NAMES.add(model_name)
            
            # If name conflict with timm (e.g. resnet50), we rely on user preference 
            # or registration order. Here we skip if already registered (timm priority)
            # OR we can force overwrite if lazyslide implementation is preferred.
            if not FEATURE_EXTRACTORS.is_available(model_name):
                @FEATURE_EXTRACTORS.register(model_name)
                def _zs_factory(name=model_name, **kwargs):
                    return name # Lazyslide extractors might just return the string ID for the backend

# Auto-populate
populate_dynamic_registries()