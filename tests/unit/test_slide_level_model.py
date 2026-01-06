"""Unit tests for slide-level vector models."""

from __future__ import annotations

import torch

from pathbench.core.models.slide_mlp import SlideVectorMLP


def test_slide_vector_mlp_forward_shapes():
    """
    Ensure slide-level models accept (batch, feature_dim) inputs.

    Expected logits shape: (batch_size=3, num_classes=2).
    """
    model = SlideVectorMLP(input_dim=5, hidden_dim=4, output_dim=2)
    inputs = torch.randn(3, 5)

    logits = model(inputs)
    assert logits.shape == (3, 2)

    labels = torch.tensor([0, 1, 0])
    output = model(inputs, label=labels, loss_fn=torch.nn.CrossEntropyLoss())
    assert "logits" in output and "loss" in output