from __future__ import annotations

from typing import Any

import torch

from pathforge.core.datasets.bag_schema import BagBatch, assert_bag_schema
from pathforge.utils.optional.torchmil import is_torchmil_available, load_torchmil_modules


def torchmil_or_pathforge_collate(batch: list[Any], *, use_torchmil: bool = True) -> BagBatch:
    """Collate variable-length MIL bags with TorchMIL when available.

    Args:
        batch: List of canonical single-bag dictionaries. Each ``X`` is shaped
            ``[N_i, D]``.
        use_torchmil: If true and TorchMIL is installed, dispatch to
            ``torchmil.data.collate_fn``. Otherwise use PathForge's fallback
            implementation with the same padded ``X`` and ``mask`` semantics.

    Returns:
        BagBatch: Padded batch with ``X`` shaped ``[B, N_max, D]``, ``Y`` stacked
        along ``[B]`` where possible, and ``mask`` shaped ``[B, N_max]`` where
        real instances are true.

    Example:
        .. code-block:: python

            torch.manual_seed(7)
            batch = [
                {"X": torch.ones(2, 4), "Y": torch.tensor(0)},
                {"X": torch.ones(3, 4), "Y": torch.tensor(1)},
            ]
            out = torchmil_or_pathforge_collate(batch, use_torchmil=False)
            assert out["X"].shape == (2, 3, 4)
            assert out["mask"].tolist() == [[True, True, False], [True, True, True]]

    """

    bag_dicts = []
    for item in batch:
        assert_bag_schema(item, batched=False)
        bag_dicts.append(item)
    if use_torchmil and is_torchmil_available():
        collated = load_torchmil_modules().data.collate_fn(bag_dicts)
        result = dict(collated)
        if "mask" in result:
            result["mask"] = result["mask"].bool()
        assert_bag_schema(result, batched=True)
        return result  # type: ignore[return-value]
    return pathforge_collate(bag_dicts)


def pathforge_collate(batch: list[BagBatch]) -> BagBatch:
    """Fallback collate implementation matching TorchMIL padding semantics.

    Args:
        batch: Canonical single-bag dictionaries with ``X`` shaped ``[N_i, D]``.

    Returns:
        BagBatch: Padded batch. Padding values are zero, ``X`` is ``float32``,
        and ``mask`` is boolean.

    Raises:
        AssertionError: If feature dimensions differ or optional tensor shapes
            are incompatible.
    """

    assert batch, "Cannot collate an empty MIL batch."
    for item in batch:
        assert_bag_schema(item, batched=False)

    xs = [item["X"].float() for item in batch]
    feature_dim = xs[0].shape[-1]
    assert all(x.shape[-1] == feature_dim for x in xs), "All bags must share feature dimension D."

    batch_size = len(xs)
    max_instances = max(x.shape[0] for x in xs)
    padded_x = xs[0].new_zeros((batch_size, max_instances, feature_dim))
    mask = torch.zeros((batch_size, max_instances), dtype=torch.bool, device=xs[0].device)
    for row_idx, x in enumerate(xs):
        padded_x[row_idx, : x.shape[0]] = x
        mask[row_idx, : x.shape[0]] = True

    result: dict[str, Any] = {"X": padded_x, "Y": _collate_targets([item["Y"] for item in batch]), "mask": mask}
    _collate_optional(batch, result, "coords", trailing_shape=(2,))
    _collate_optional(batch, result, "y_inst", trailing_shape=())
    _collate_optional_adj(batch, result)
    assert_bag_schema(result, batched=True)
    return result  # type: ignore[return-value]


def _collate_targets(targets: list[Any]) -> Any:
    if all(isinstance(target, dict) for target in targets):
        keys = set(targets[0])
        assert all(set(target) == keys for target in targets), "Survival target dict keys must match."
        return {key: _stack_or_tensor([target[key] for target in targets]) for key in keys}
    return _stack_or_tensor(targets)


def _stack_or_tensor(values: list[Any]) -> torch.Tensor:
    if all(isinstance(value, torch.Tensor) for value in values):
        return torch.stack([value.reshape(()) if value.ndim == 0 else value for value in values])
    return torch.as_tensor(values)


def _collate_optional(
    batch: list[BagBatch],
    result: dict[str, Any],
    key: str,
    *,
    trailing_shape: tuple[int, ...],
) -> None:
    if not any(key in item and item[key] is not None for item in batch):
        return
    assert all(key in item and item[key] is not None for item in batch), (
        f"Optional key '{key}' must be present for all bags or none."
    )
    max_instances = result["X"].shape[1]
    first = batch[0][key]
    assert isinstance(first, torch.Tensor)
    padded_shape = (len(batch), max_instances, *trailing_shape)
    padded = first.new_zeros(padded_shape)
    for row_idx, item in enumerate(batch):
        value = item[key]
        assert isinstance(value, torch.Tensor)
        padded[row_idx, : value.shape[0]] = value
    result[key] = padded


def _collate_optional_adj(batch: list[BagBatch], result: dict[str, Any]) -> None:
    key = "adj"
    if not any(key in item and item[key] is not None for item in batch):
        return
    assert all(key in item and item[key] is not None for item in batch), (
        "Optional key 'adj' must be present for all bags or none."
    )
    max_instances = result["X"].shape[1]
    first = batch[0][key]
    assert isinstance(first, torch.Tensor)
    padded = first.new_zeros((len(batch), max_instances, max_instances))
    for row_idx, item in enumerate(batch):
        value = item[key]
        assert isinstance(value, torch.Tensor)
        padded[row_idx, : value.shape[0], : value.shape[1]] = value
    result[key] = padded
