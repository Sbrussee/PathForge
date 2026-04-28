from __future__ import annotations

from typing import Any

import torch

from pathbench.core.datasets.bag_schema import BagBatch, assert_bag_schema


def pathbench_item_to_bag_dict(item: Any) -> BagBatch:
    """Convert a legacy PathBench dataset item into the canonical bag schema.

    Args:
        item: Either a canonical bag dictionary or a legacy ``(bag, target)``
            tuple. ``bag`` must be a finite floating tensor shaped ``[N, D]`` or
            ``[B, N, D]``. ``target`` may be a scalar, tensor, or survival target
            dictionary.

    Returns:
        BagBatch: Dictionary with ``X`` and ``Y`` plus any existing optional
        tensors.

    Example:
        ```python
        bag = torch.zeros(8, 1024)
        converted = pathbench_item_to_bag_dict((bag, 1))
        assert converted["X"].shape == (8, 1024)
        ```

    Raises:
        TypeError: If the item is neither a dict nor a two-item tuple/list.
        AssertionError: If the resulting bag violates schema invariants.
    """

    if isinstance(item, dict):
        converted = dict(item)
    elif isinstance(item, (tuple, list)) and len(item) == 2:
        converted = {"X": item[0], "Y": item[1]}
    else:
        raise TypeError(
            "MIL dataset items must be canonical bag dictionaries or legacy "
            "(bag, target) tuples."
        )

    if not isinstance(converted["X"], torch.Tensor):
        converted["X"] = torch.as_tensor(converted["X"], dtype=torch.float32)
    elif converted["X"].dtype != torch.float32:
        converted["X"] = converted["X"].float()

    converted["Y"] = _target_to_tensor_if_scalar(converted["Y"])
    assert_bag_schema(converted)
    return converted  # type: ignore[return-value]


def _target_to_tensor_if_scalar(target: Any) -> Any:
    if isinstance(target, dict):
        return {
            key: value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            for key, value in target.items()
        }
    if isinstance(target, torch.Tensor):
        return target
    if isinstance(target, int):
        return torch.tensor(target, dtype=torch.long)
    if isinstance(target, float):
        return torch.tensor(target, dtype=torch.float32)
    return target
