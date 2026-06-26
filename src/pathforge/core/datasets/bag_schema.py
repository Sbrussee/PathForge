from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict, cast

import torch


class BagBatch(TypedDict):
    """Canonical PathForge MIL batch schema.

    Required keys:
        X: Floating point feature tensor shaped ``[B, N, D]`` for a padded batch
            or ``[N, D]`` for one unbatched bag. Values are finite feature
            embeddings, usually ``float32``.
        Y: Bag-level target tensor. Classification targets are usually ``[B]``
            integer class ids. Continuous survival targets may be a dict with
            ``time`` and ``event`` tensors shaped ``[B]``.

    Optional keys:
        mask: Boolean or binary tensor shaped ``[B, N]`` where ``True``/``1``
            marks real instances and ``False``/``0`` marks padding.
        coords: Coordinate tensor shaped ``[B, N, 2]`` with x/y tile positions.
        adj: Dense adjacency tensor shaped ``[B, N, N]``.
        y_inst: Instance-level labels shaped ``[B, N]``.

    Example:
        ```python
        import torch
        from pathforge.core.datasets.bag_schema import assert_bag_schema

        batch = {
            "X": torch.zeros(2, 4, 1024, dtype=torch.float32),
            "Y": torch.tensor([0, 1], dtype=torch.long),
            "mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool),
        }
        assert_bag_schema(batch, batched=True)
        ```

    Raises:
        AssertionError: Validation helpers raise when required keys, ranks,
            dtypes, shapes, or finite-value contracts are violated.
    """

    X: torch.Tensor
    Y: Any
    mask: NotRequired[torch.Tensor]
    coords: NotRequired[torch.Tensor]
    adj: NotRequired[torch.Tensor]
    y_inst: NotRequired[torch.Tensor]


def as_bag_batch(batch: dict[str, Any]) -> BagBatch:
    """Cast a plain dictionary to ``BagBatch`` after runtime validation."""

    assert_bag_schema(batch)
    return cast(BagBatch, batch)


def assert_bag_schema(
    batch: dict[str, Any],
    *,
    batched: bool | None = None,
    check_finite: bool = True,
) -> None:
    """Validate the canonical MIL bag schema at module boundaries.

    Args:
        batch: Mapping containing at least ``X`` and ``Y``.
        batched: ``True`` requires ``X`` rank 3, ``False`` requires rank 2, and
            ``None`` accepts either. Optional tensors are checked against the
            inferred rank.
        check_finite: When true, checks floating tensors for NaN/Inf.

    Raises:
        AssertionError: If the bag violates required shape, dtype, or finiteness
            invariants.
    """

    assert isinstance(batch, dict), "Bag batch must be a dictionary."
    assert "X" in batch, "Bag batch requires key 'X'."
    assert "Y" in batch, "Bag batch requires key 'Y'."

    x = batch["X"]
    assert isinstance(x, torch.Tensor), "Bag batch 'X' must be a torch.Tensor."
    assert x.is_floating_point(), "Bag batch 'X' must be floating point."
    assert x.ndim in {2, 3}, "Bag batch 'X' must have shape [N, D] or [B, N, D]."
    if batched is True:
        assert x.ndim == 3, "Batched bag 'X' must have shape [B, N, D]."
    if batched is False:
        assert x.ndim == 2, "Single bag 'X' must have shape [N, D]."
    assert x.shape[-2] > 0, "Bag must contain at least one instance."
    assert x.shape[-1] > 0, "Bag feature dimension must be positive."
    if check_finite:
        assert torch.isfinite(x).all(), "Bag batch 'X' contains NaN or Inf."

    batch_size = x.shape[0] if x.ndim == 3 else None
    bag_size = x.shape[-2]

    if "mask" in batch and batch["mask"] is not None:
        _assert_optional_tensor(
            name="mask",
            value=batch["mask"],
            expected_shape=(batch_size, bag_size) if x.ndim == 3 else (bag_size,),
            dtype_kind="mask",
            check_finite=check_finite,
        )

    if "coords" in batch and batch["coords"] is not None:
        _assert_optional_tensor(
            name="coords",
            value=batch["coords"],
            expected_shape=(batch_size, bag_size, 2) if x.ndim == 3 else (bag_size, 2),
            dtype_kind="numeric",
            check_finite=check_finite,
        )

    if "adj" in batch and batch["adj"] is not None:
        _assert_optional_tensor(
            name="adj",
            value=batch["adj"],
            expected_shape=(batch_size, bag_size, bag_size)
            if x.ndim == 3
            else (bag_size, bag_size),
            dtype_kind="numeric",
            check_finite=check_finite,
        )

    if "y_inst" in batch and batch["y_inst"] is not None:
        _assert_optional_tensor(
            name="y_inst",
            value=batch["y_inst"],
            expected_shape=(batch_size, bag_size) if x.ndim == 3 else (bag_size,),
            dtype_kind="any",
            check_finite=check_finite,
        )


def _assert_optional_tensor(
    *,
    name: str,
    value: Any,
    expected_shape: tuple[int | None, ...],
    dtype_kind: Literal["any", "mask", "numeric"],
    check_finite: bool,
) -> None:
    assert isinstance(value, torch.Tensor), f"Bag batch '{name}' must be a torch.Tensor."
    assert value.ndim == len(expected_shape), (
        f"Bag batch '{name}' rank mismatch: expected {len(expected_shape)}, "
        f"got {value.ndim}."
    )
    for dim_idx, expected in enumerate(expected_shape):
        if expected is not None:
            assert value.shape[dim_idx] == expected, (
                f"Bag batch '{name}' shape mismatch at dim {dim_idx}: "
                f"expected {expected}, got {value.shape[dim_idx]}."
            )
    if dtype_kind == "mask":
        assert value.dtype in {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}, (
            "Bag batch 'mask' must be bool or integer binary."
        )
    if check_finite and value.is_floating_point():
        assert torch.isfinite(value).all(), f"Bag batch '{name}' contains NaN or Inf."
