from __future__ import annotations

import pytest
import torch

from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.models.prototype_mil import PrototypeMIL
from pathbench.core.models.var_mil import VarMIL


def test_instance_scores_returns_attention_for_varmil() -> None:
    torch.manual_seed(7)
    model = VarMIL(input_dim=4, hidden_dim=3, output_dim=2)
    bag = torch.randn(2, 5, 4)

    attention = model.forward_bag(bag, return_attention=True)["attention"].squeeze(-1)
    scores = model.instance_scores(bag)

    assert scores.shape == (2, 5)
    assert torch.allclose(scores, attention)


def test_instance_scores_reduces_multi_channel_attention_to_per_instance_scores() -> None:
    torch.manual_seed(7)
    model = PrototypeMIL(input_dim=4, num_prototypes=3, output_dim=2)
    bag = torch.randn(1, 4, 4)

    attention = model.forward_bag(bag, return_attention=True)["attention"]
    scores = model.instance_scores(bag)

    assert attention.shape == (1, 4, 3)
    assert scores.shape == (1, 4)
    assert torch.allclose(scores, attention.mean(dim=-1))


def test_instance_scores_raises_when_model_has_no_attention_output() -> None:
    class _NoAttentionModel(MILModelBase):
        @property
        def bag_size(self) -> int | None:
            return None

        def forward_bag(self, bag, mask=None, coords=None, label=None, loss_fn=None):
            del mask, coords, label, loss_fn
            return torch.zeros(bag.shape[0], 2)

    model = _NoAttentionModel()

    with pytest.raises(AttributeError, match="instance-level attention scores"):
        model.instance_scores(torch.zeros(1, 3, 2))
