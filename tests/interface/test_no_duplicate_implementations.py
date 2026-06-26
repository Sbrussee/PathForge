"""Regression tests for canonical, non-duplicated implementations."""

from __future__ import annotations

from pathforge.core.experiments.base import ComboConfig
from pathforge.core.models.base import TorchModelBase
from pathforge.core.models.mil_base import MILModelBase
from pathforge.core.models.slide_base import SlideLevelModel
from pathforge.policy.utils import ComboConfig as PolicyComboConfig


def test_policy_combo_config_reuses_canonical_experiment_implementation() -> None:
    """Policy utilities should re-export the canonical ComboConfig implementation."""
    assert PolicyComboConfig is ComboConfig


def test_mil_and_slide_model_bases_share_one_torch_implementation() -> None:
    """Torch-backed model plumbing should live in one shared base class."""
    assert issubclass(MILModelBase, TorchModelBase)
    assert issubclass(SlideLevelModel, TorchModelBase)
