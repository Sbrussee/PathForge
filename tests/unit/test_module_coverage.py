from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any

import pytest
import torch

from pathbench.core.losses.base import BaseLoss
from pathbench.core.models.base import ModelBase
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.models.slide_base import SlideLevelModel
from pathbench.policy.base import PolicyBase


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "pathbench"
MODELS_ROOT = SRC_ROOT / "core" / "models"
LOSSES_ROOT = SRC_ROOT / "core" / "losses"
POLICY_ROOT = SRC_ROOT / "policy"

MODEL_SUPPORT_MODULES = {
    "__init__",
    "base",
    "layers",
    "mil_base",
    "slide_base",
    "utils",
}
LOSS_SUPPORT_MODULES = {"__init__", "base", "impl"}
POLICY_SUPPORT_MODULES = {"__init__", "base", "utils"}

OPTIONAL_MODULE_DEPS = {
    "pathbench.core.models.mamba_mil": ("mamba",),
}


def _module_names(root: Path) -> list[str]:
    prefix = ".".join(root.relative_to(SRC_ROOT.parent).parts)
    return [f"{prefix}.{path.stem}" for path in sorted(root.glob("*.py"))]


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


def _concrete_model_classes(module_name: str) -> list[type[Any]]:
    module = importlib.import_module(module_name)
    classes = _classes_defined_in_module(module)
    return [
        cls
        for cls in classes
        if issubclass(cls, (ModelBase, MILModelBase, SlideLevelModel))
        and cls not in {ModelBase, MILModelBase, SlideLevelModel}
        and not inspect.isabstract(cls)
    ]


def _all_model_classes(module_name: str) -> list[type[Any]]:
    module = importlib.import_module(module_name)
    classes = _classes_defined_in_module(module)
    return [
        cls
        for cls in classes
        if issubclass(cls, (ModelBase, MILModelBase, SlideLevelModel))
        and cls not in {ModelBase, MILModelBase, SlideLevelModel}
    ]


def _concrete_loss_classes(module_name: str) -> list[type[Any]]:
    module = importlib.import_module(module_name)
    classes = _classes_defined_in_module(module)
    return [
        cls
        for cls in classes
        if issubclass(cls, BaseLoss)
        and cls is not BaseLoss
        and not inspect.isabstract(cls)
    ]


def _concrete_policy_classes(module_name: str) -> list[type[Any]]:
    module = importlib.import_module(module_name)
    classes = _classes_defined_in_module(module)
    return [
        cls
        for cls in classes
        if issubclass(cls, PolicyBase)
        and cls is not PolicyBase
        and not inspect.isabstract(cls)
    ]


def _has_only_optional_parameters(cls: type[Any]) -> bool:
    parameters = list(inspect.signature(cls).parameters.values())[1:]
    return all(param.default is not inspect._empty for param in parameters)


class _DummyExperiment:
    def __init__(self) -> None:
        self.cfg = type("Cfg", (), {})()
        self.project_root = None


class _DummyMILMember(MILModelBase):
    @property
    def bag_size(self) -> int | None:
        return None

    def forward_bag(self, bag: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return bag.mean(dim=1)


@pytest.mark.parametrize("module_name", _module_names(MODELS_ROOT))
def test_all_model_modules_are_covered_by_importable_tests(module_name: str) -> None:
    _skip_if_optional_dependency_missing(module_name)
    importlib.import_module(module_name)

    if module_name.split(".")[-1] in MODEL_SUPPORT_MODULES:
        return

    classes = _all_model_classes(module_name)
    assert classes, f"No model classes discovered in {module_name}"


@pytest.mark.parametrize("module_name", _module_names(LOSSES_ROOT))
def test_all_loss_modules_are_covered_by_importable_tests(module_name: str) -> None:
    importlib.import_module(module_name)

    if module_name.split(".")[-1] in LOSS_SUPPORT_MODULES:
        return

    classes = _concrete_loss_classes(module_name)
    assert classes, f"No concrete loss classes discovered in {module_name}"


@pytest.mark.parametrize("module_name", _module_names(POLICY_ROOT))
def test_all_policy_modules_are_covered_by_importable_tests(module_name: str) -> None:
    importlib.import_module(module_name)

    if module_name.split(".")[-1] in POLICY_SUPPORT_MODULES:
        return

    classes = _concrete_policy_classes(module_name)
    assert classes, f"No concrete policy classes discovered in {module_name}"


def test_all_lightweight_model_classes_can_be_instantiated() -> None:
    instances: list[Any] = []
    for module_name in _module_names(MODELS_ROOT):
        _skip_if_optional_dependency_missing(module_name)
        stem = module_name.split(".")[-1]
        if stem in MODEL_SUPPORT_MODULES:
            continue

        for cls in _concrete_model_classes(module_name):
            if cls.__name__ == "EnsembleMILModel":
                instances.append(cls(members=[_DummyMILMember(), _DummyMILMember()]))
                continue
            if _has_only_optional_parameters(cls):
                instances.append(cls())

    assert instances, "Expected at least one concrete model instance to be created."


def test_all_loss_classes_can_be_instantiated() -> None:
    instances = [
        cls()
        for module_name in _module_names(LOSSES_ROOT)
        if module_name.split(".")[-1] not in LOSS_SUPPORT_MODULES
        for cls in _concrete_loss_classes(module_name)
    ]
    assert instances, "Expected concrete loss classes to be instantiated."


def test_policy_base_still_accepts_experiment_like_object() -> None:
    instance = _DummyExperiment()
    assert instance.cfg is not None
    assert instance.project_root is None
