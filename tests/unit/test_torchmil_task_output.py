import pytest
import torch

from pathforge.adapters.torchmil.task_output import normalize_torchmil_output


def test_normalize_classification_logits_from_dict():
    logits = normalize_torchmil_output({"logits": torch.zeros(2, 3)}, task="classification")

    assert logits.shape == (2, 3)


def test_normalize_survival_flattens_single_risk_column():
    risk = normalize_torchmil_output(torch.zeros(2, 1), task="survival")

    assert risk.shape == (2,)


def test_normalize_survival_discrete_requires_time_bins():
    with pytest.raises(AssertionError, match="Discrete survival output"):
        normalize_torchmil_output(torch.zeros(2), task="survival_discrete")
