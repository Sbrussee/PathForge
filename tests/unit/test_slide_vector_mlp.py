"""Unit tests for SlideVectorMLP — forward pass and trainer compatibility."""

from __future__ import annotations

import torch


def _register_and_import():
    from pathbench.utils.registries import MODELS
    from pathbench.core.models.mil_base import MILModelBase
    from pathbench.core.models.slide_base import SlideLevelModel
    from pathbench.core.models.base import ModelBase

    if not MODELS.is_available("SlideVectorMLP"):
        import importlib
        importlib.import_module("pathbench.core.models.slide_mlp")

    from pathbench.core.models.slide_mlp import SlideVectorMLP
    return SlideVectorMLP, MILModelBase, SlideLevelModel, ModelBase


def test_slide_vector_mlp_inherits_correctly():
    SlideVectorMLP, MILModelBase, SlideLevelModel, ModelBase = _register_and_import()
    assert issubclass(SlideVectorMLP, MILModelBase)
    assert issubclass(SlideVectorMLP, SlideLevelModel)
    assert issubclass(SlideVectorMLP, ModelBase)


def test_slide_vector_mlp_forward_bag_output_shape():
    SlideVectorMLP, *_ = _register_and_import()
    model = SlideVectorMLP(input_dim=16, hidden_dim=32, output_dim=3)
    bag = torch.randn(2, 10, 16)  # (B=2, N=10, D=16)
    out = model.forward_bag(bag)
    # without loss_fn the model returns logits (B, output_dim)
    assert out.shape == (2, 3)


def test_slide_vector_mlp_forward_bag_with_loss_returns_dict():
    SlideVectorMLP, *_ = _register_and_import()
    from pathbench.utils.registries import LOSSES

    model = SlideVectorMLP(input_dim=8, hidden_dim=16, output_dim=2)
    bag = torch.randn(4, 5, 8)
    label = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    loss_fn = LOSSES.get("CrossEntropyLoss")()
    result = model.forward_bag(bag, label=label, loss_fn=loss_fn)

    assert isinstance(result, dict)
    assert "logits" in result
    assert "loss" in result
    assert result["logits"].shape == (4, 2)
    assert result["loss"].ndim == 0  # scalar


def test_slide_vector_mlp_mean_pools_tiles():
    """forward_bag must mean-pool along the tile dimension."""
    SlideVectorMLP, *_ = _register_and_import()
    model = SlideVectorMLP(input_dim=4, hidden_dim=8, output_dim=2)
    model.eval()

    # single tile bag == multi-tile bag with same tile repeated
    tile = torch.randn(1, 4)
    bag_one = tile.unsqueeze(0)          # (1, 1, 4)
    bag_multi = tile.expand(1, 5, 4)    # (1, 5, 4)

    with torch.no_grad():
        out_one = model.forward_bag(bag_one)
        out_multi = model.forward_bag(bag_multi)

    assert torch.allclose(out_one, out_multi, atol=1e-5)


def test_slide_vector_mlp_bag_size_is_none():
    SlideVectorMLP, *_ = _register_and_import()
    model = SlideVectorMLP(input_dim=16, hidden_dim=32, output_dim=2)
    assert model.bag_size is None


def test_slide_vector_mlp_registered_in_models_registry():
    from pathbench.utils.registries import MODELS
    # trigger lazy registration
    _register_and_import()
    assert MODELS.is_available("SlideVectorMLP")
