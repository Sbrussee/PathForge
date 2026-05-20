from __future__ import annotations

import torch
import torch.nn as nn

from pathbench.core.models.layers import create_mlp


def test_create_mlp_builds_hidden_stack_and_output_shape() -> None:
    mlp = create_mlp(in_dim=6, hid_dims=[4, 3], out_dim=2, dropout=0.0)
    output = mlp(torch.randn(5, 6))

    assert isinstance(mlp, nn.Sequential)
    assert output.shape == (5, 2)


def test_create_mlp_without_hidden_layers_builds_single_linear_head() -> None:
    mlp = create_mlp(in_dim=4, hid_dims=[], out_dim=1, end_with_fc=True)

    assert isinstance(mlp, nn.Sequential)
    assert len(mlp) == 1
    assert isinstance(mlp[0], nn.Linear)
