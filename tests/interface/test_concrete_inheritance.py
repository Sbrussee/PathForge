"""Interface tests for PathBench base-to-concrete inheritance contracts."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from typing import Any

import pytest

from pathbench.core.annotations.base import AnnotationsBase
from pathbench.core.datasets.base import BagDatasetBase, DatasetBase
from pathbench.core.models.base import ModelBase, ScikitBase
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.models.slide_base import SlideLevelModel
from pathbench.policy.base import PolicyBase
from pathbench.training.base import TrainerBase


OPTIONAL_MODULE_DEPS = {
    "pathbench.core.models.mamba_mil": ("mamba",),
}

CONCRETE_CLASS_BASES: dict[str, dict[str, tuple[type[Any], ...]]] = {
    "pathbench.core.annotations.csv": {
        "CSVAnnotations": (AnnotationsBase,),
    },
    "pathbench.core.datasets.bag_dataset": {
        "BagDataset": (BagDatasetBase,),
    },
    "pathbench.core.datasets.wsi_dataset": {
        "WSIDataset": (DatasetBase,),
    },
    "pathbench.training.lightning": {
        "LightningTrainer": (TrainerBase,),
    },
    "pathbench.training.sklearn_trainer": {
        "SklearnSlideTrainer": (TrainerBase,),
    },
}

DISCOVERED_MODULE_BASES: dict[str, tuple[type[Any], ...]] = {
    "pathbench.policy.benchmarking": (PolicyBase,),
    "pathbench.policy.feature_extraction": (PolicyBase,),
    "pathbench.policy.optimization": (PolicyBase,),
    "pathbench.core.models.mamba_mil": (MILModelBase, ModelBase),
    "pathbench.core.models.mil_ens": (MILModelBase, ModelBase),
    "pathbench.core.models.mil_graph": (MILModelBase, ModelBase),
    "pathbench.core.models.mil_mm": (MILModelBase, ModelBase),
    "pathbench.core.models.perceiver_mil": (MILModelBase, ModelBase),
    "pathbench.core.models.prototype_mil": (MILModelBase, ModelBase),
    "pathbench.core.models.slide_mlp": (SlideLevelModel, MILModelBase, ModelBase),
    "pathbench.core.models.sklearn_slide": (ScikitBase, ModelBase),
    "pathbench.core.models.var_mil": (MILModelBase, ModelBase),
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
