"""Unit tests for MIL training utilities and batch collation."""

from __future__ import annotations

import torch

from pathbench.training.utils import mil_collate


def test_mil_collate_with_slide_ids_and_padding():
    """
    Collate variable-length bags with slide IDs and verify padding/mask shapes.

    Expected batched bag shape: (batch_size=2, max_instances=3, feature_dim=4).
    Expected mask shape: (batch_size=2, max_instances=3).
    """
    bag_a = torch.randn(3, 4)
    bag_b = torch.randn(1, 4)

    batch = [
        (bag_a, 0, "slide_a"),
        (bag_b, 1, "slide_b"),
    ]

    bags, labels, mask, slide_ids = mil_collate(batch)
    assert bags.shape == (2, 3, 4)
    assert labels.shape == (2,)
    assert mask.shape == (2, 3)
    assert slide_ids == ["slide_a", "slide_b"]
    assert mask[0].tolist() == [True, True, True]
    assert mask[1].tolist() == [True, False, False]


def test_mil_collate_survival_labels():
    """
    Collate survival labels into dict tensors with aligned shapes.

    Expected label shapes: time=(batch_size,), event=(batch_size,).
    """
    bag_a = torch.randn(2, 6)
    bag_b = torch.randn(2, 6)

    batch = [
        (bag_a, {"time": 5.0, "event": 1.0}),
        (bag_b, {"time": 3.0, "event": 0.0}),
    ]

    bags, labels, mask = mil_collate(batch)
    assert bags.shape == (2, 2, 6)
    assert mask.shape == (2, 2)
    assert set(labels.keys()) == {"time", "event"}
    assert labels["time"].shape == (2,)
    assert labels["event"].shape == (2,)