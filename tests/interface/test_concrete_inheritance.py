"""Interface tests for PathForge base-to-concrete inheritance contracts."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from typing import Any

import pytest

from pathforge.core.annotations.base import AnnotationsBase
from pathforge.core.datasets.base import BagDatasetBase, DatasetBase
from pathforge.core.models.base import ModelBase, ScikitBase
from pathforge.core.models.mil_base import MILModelBase
from pathforge.core.models.slide_base import SlideLevelModel
from pathforge.policy.base import PolicyBase
from pathforge.training.base import TrainerBase


OPTIONAL_MODULE_DEPS = {
    "pathforge.core.models.mamba_mil": ("mamba",),
}

CONCRETE_CLASS_BASES: dict[str, dict[str, tuple[type[Any], ...]]] = {
    "pathforge.core.annotations.csv": {
        "CSVAnnotations": (AnnotationsBase,),
    },
    "pathforge.core.datasets.bag_dataset": {
        "BagDataset": (BagDatasetBase,),
    },
    "pathforge.core.datasets.wsi_dataset": {
        "WSIDataset": (DatasetBase,),
    },
    "pathforge.training.lightning": {
        "LightningTrainer": (TrainerBase,),
    },
    "pathforge.training.sklearn_trainer": {
        "SklearnSlideTrainer": (TrainerBase,),
    },
}

DISCOVERED_MODULE_BASES: dict[str, tuple[type[Any], ...]] = {
    "pathforge.policy.benchmarking": (PolicyBase,),
    "pathforge.policy.feature_extraction": (PolicyBase,),
    "pathforge.policy.optimization": (PolicyBase,),
    "pathforge.core.models.mamba_mil": (MILModelBase, ModelBase),
    "pathforge.core.models.mil_ens": (MILModelBase, ModelBase),
    "pathforge.core.models.mil_graph": (MILModelBase, ModelBase),
    "pathforge.core.models.perceiver_mil": (MILModelBase, ModelBase),
    "pathforge.core.models.prototype_mil": (MILModelBase, ModelBase),
    "pathforge.core.models.slide_mlp": (SlideLevelModel, MILModelBase, ModelBase),
    "pathforge.core.models.sklearn_slide": (ScikitBase, ModelBase),
    "pathforge.core.models.var_mil": (MILModelBase, ModelBase),
}


def _skip_if_optional_dependency_missing(module_name: str) -> None:
    for dependency in OPTIONAL_MODULE_DEPS.get(module_name, ()):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"Optional dependency '{dependency}' is not installed.")


def _classes_defined_in_module(module: Any) -> list[type[Any]]:
    return [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
    ]


@pytest.mark.parametrize(
    ("module_name", "class_bases"),
    sorted(CONCRETE_CLASS_BASES.items()),
)
def test_named_concrete_classes_subclass_expected_bases(
    module_name: str,
    class_bases: dict[str, tuple[type[Any], ...]],
) -> None:
    _skip_if_optional_dependency_missing(module_name)
    module = importlib.import_module(module_name)

    for class_name, expected_bases in class_bases.items():
        cls = getattr(module, class_name)
        assert issubclass(cls, expected_bases)
        assert not inspect.isabstract(cls)


@pytest.mark.parametrize(
    ("module_name", "expected_bases"),
    sorted(DISCOVERED_MODULE_BASES.items()),
)
def test_discovered_concrete_classes_subclass_expected_bases(
    module_name: str,
    expected_bases: tuple[type[Any], ...],
) -> None:
    _skip_if_optional_dependency_missing(module_name)
    module = importlib.import_module(module_name)
    classes = _classes_defined_in_module(module)
    concrete_classes = [
        cls
        for cls in classes
        if issubclass(cls, expected_bases) and not inspect.isabstract(cls)
    ]

    assert concrete_classes, f"No concrete classes discovered in {module_name}"
    assert all(issubclass(cls, expected_bases) for cls in concrete_classes)
