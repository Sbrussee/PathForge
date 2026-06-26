import pytest
import torch

from pathforge.adapters.torchmil.collate import pathforge_collate, torchmil_or_pathforge_collate
from pathforge.core.datasets.bag_schema import assert_bag_schema


def test_assert_bag_schema_accepts_valid_batched_bag():
    batch = {
        "X": torch.zeros(2, 3, 4),
        "Y": torch.tensor([0, 1]),
        "mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        "coords": torch.zeros(2, 3, 2),
    }

    assert_bag_schema(batch, batched=True)


def test_assert_bag_schema_rejects_nonfinite_features():
    batch = {"X": torch.tensor([[[float("nan")]]]), "Y": torch.tensor([0])}

    with pytest.raises(AssertionError, match="NaN or Inf"):
        assert_bag_schema(batch)


def test_pathforge_collate_pads_variable_length_bags_and_mask():
    batch = [
        {"X": torch.ones(2, 4), "Y": torch.tensor(0)},
        {"X": torch.full((3, 4), 2.0), "Y": torch.tensor(1)},
    ]

    out = pathforge_collate(batch)

    assert out["X"].shape == (2, 3, 4)
    assert out["X"].dtype == torch.float32
    assert out["mask"].tolist() == [[True, True, False], [True, True, True]]
    assert out["Y"].tolist() == [0, 1]
    assert out["X"][0, 2].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_pathforge_collate_pads_optional_coords_and_adjacency():
    batch = [
        {
            "X": torch.ones(2, 4),
            "Y": torch.tensor(0),
            "coords": torch.tensor([[0, 0], [1, 1]], dtype=torch.float32),
            "adj": torch.eye(2),
        },
        {
            "X": torch.ones(3, 4),
            "Y": torch.tensor(1),
            "coords": torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.float32),
            "adj": torch.eye(3),
        },
    ]

    out = pathforge_collate(batch)

    assert out["coords"].shape == (2, 3, 2)
    assert out["adj"].shape == (2, 3, 3)
    assert out["coords"][0, 2].tolist() == [0.0, 0.0]
    assert out["adj"][0, 2].tolist() == [0.0, 0.0, 0.0]


def test_pathforge_collate_rejects_partial_optional_keys():
    batch = [
        {"X": torch.ones(2, 4), "Y": torch.tensor(0), "coords": torch.zeros(2, 2)},
        {"X": torch.ones(3, 4), "Y": torch.tensor(1)},
    ]

    with pytest.raises(AssertionError, match="coords"):
        pathforge_collate(batch)

def test_torchmil_or_pathforge_collate_accepts_canonical_single_bag_dicts():
    batch = [
        {"X": torch.ones(1, 2), "Y": torch.tensor(0)},
        {"X": torch.ones(2, 2), "Y": torch.tensor(1)},
    ]

    out = torchmil_or_pathforge_collate(batch, use_torchmil=False)

    assert out["X"].shape == (2, 2, 2)
    assert out["mask"].tolist() == [[True, False], [True, True]]
