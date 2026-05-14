"""Interface-level contract tests for public PathBench base layers."""

from __future__ import annotations

import inspect

from pathbench.core.base import RegistryBase
from pathbench.core.datasets.base import BagDatasetBase, DatasetBase
from pathbench.core.slide_processing.base import SlideProcessorBase
from pathbench.policy.base import PolicyBase
from pathbench.training.base import TrainerBase


def test_registry_base_exposes_required_interface_methods() -> None:
    """RegistryBase must expose the stable plugin-registry interface."""
    abstract_methods = RegistryBase.__abstractmethods__
    assert {"register", "get", "list_plugins", "is_available"} <= abstract_methods


def test_dataset_base_contracts_are_present() -> None:
    """Dataset interfaces must define the expected abstract access surface."""
    assert {"name", "num_samples", "__getitem__"} <= DatasetBase.__abstractmethods__
    assert {"num_bags", "__getitem__"} <= BagDatasetBase.__abstractmethods__


def test_slide_processor_base_exposes_required_workflow_contract() -> None:
    """SlideProcessorBase must define the core extraction workflow surface."""
    assert {
        "load_wsi",
        "get_thumbnail",
        "segment_tissue",
        "extract_patches",
        "validate_tile_spec",
        "extract_features",
        "extract_cells",
        "inspect_slide",
    } <= SlideProcessorBase.__abstractmethods__


def test_policy_and_trainer_bases_expose_execution_contracts() -> None:
    """Policy and trainer abstractions must keep their stable execution methods."""
    assert "execute" in PolicyBase.__abstractmethods__
    assert {"fit", "predict"} <= TrainerBase.__abstractmethods__


def test_trainer_base_method_names_remain_stable() -> None:
    """TrainerBase should keep predictable public parameter names."""
    fit_signature = inspect.signature(TrainerBase.fit)
    predict_signature = inspect.signature(TrainerBase.predict)

    assert list(fit_signature.parameters)[:5] == [
        "self",
        "model",
        "dataset_train",
        "dataset_val",
        "loss_func",
    ]
    assert list(predict_signature.parameters)[:3] == ["self", "model", "dataset"]
