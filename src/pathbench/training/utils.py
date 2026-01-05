from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch


def unpack_mil_batch(
    batch: Tuple[torch.Tensor, Any] | Tuple[torch.Tensor, Any, torch.Tensor] | Tuple[Any, ...]
) -> Tuple[torch.Tensor, Any, torch.Tensor | None]:
    """Unpack a MIL batch into (bag, target, mask)."""
    if len(batch) == 2:
        bag, target = batch
        return bag, target, None
    if len(batch) == 3:
        bag, target, mask = batch
        return bag, target, mask
    if len(batch) >= 4:
        bag, target, mask = batch[:3]
        return bag, target, mask
    raise ValueError("Unexpected batch structure for MIL training.")


def resolve_loss_and_logits(
    output: torch.Tensor | Dict[str, Any],
    target: Any,
    loss_fn: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract logits from model output and compute loss."""
    if isinstance(output, dict):
        logits = output.get("logits")
        if logits is None:
            raise ValueError("Model output dict must include a 'logits' entry.")
        loss = output.get("loss", loss_fn(logits, target))
        return loss, logits

    logits = output
    loss = loss_fn(logits, target)
    return loss, logits


def mil_collate(
    batch: Sequence[
        Tuple[torch.Tensor, Any]
        | Tuple[torch.Tensor, Any, str]
        | Tuple[torch.Tensor, Any, torch.Tensor]
        | Tuple[torch.Tensor, Any, torch.Tensor, str]
    ]
) -> Tuple[torch.Tensor, Any, torch.Tensor] | Tuple[torch.Tensor, Any, torch.Tensor, List[str]]:
    """
    Collate MIL batches with padding and masks.

    Supports scalar labels or survival label dicts (time/event).
    """
    bags: List[torch.Tensor] = []
    labels: List[Any] = []
    slide_ids: List[str] = []

    for item in batch:
        if len(item) == 2:
            bag, label = item
        elif len(item) == 3:
            bag, label, extra = item
            if isinstance(extra, str):
                slide_ids.append(extra)
        else:
            bag, label, _mask, slide_id = item[:4]
            slide_ids.append(slide_id)

        bags.append(bag)
        labels.append(label)

    lengths = torch.tensor([bag.shape[0] for bag in bags], dtype=torch.long)
    max_len = int(lengths.max().item()) if lengths.numel() else 0
    feature_dim = max((bag.shape[1] for bag in bags), default=0)

    padded = []
    mask = []
    for bag in bags:
        if bag.shape[1] not in {0, feature_dim}:
            raise ValueError("Inconsistent feature dimensions in batch.")
        if bag.shape[1] == 0 and feature_dim > 0:
            bag = torch.zeros((bag.shape[0], feature_dim), dtype=bag.dtype)
        pad_size = max_len - bag.shape[0]
        if pad_size > 0:
            pad = torch.zeros((pad_size, bag.shape[1]), dtype=bag.dtype)
            padded_bag = torch.cat([bag, pad], dim=0)
        else:
            padded_bag = bag
        padded.append(padded_bag)
        mask.append(torch.arange(max_len) < bag.shape[0])

    batched_bags = torch.stack(padded, dim=0) if padded else torch.empty((0, 0, 0))
    batched_mask = torch.stack(mask, dim=0) if mask else torch.empty((0, 0), dtype=torch.bool)

    batched_labels: Any
    if labels and isinstance(labels[0], dict):
        keys = labels[0].keys()
        batched_labels = {
            key: torch.stack([torch.as_tensor(label[key]) for label in labels])
            for key in keys
        }
    else:
        batched_labels = torch.stack([torch.as_tensor(label) for label in labels]) if labels else torch.empty((0,))

    if any(slide_ids):
        return batched_bags, batched_labels, batched_mask, slide_ids
    return batched_bags, batched_labels, batched_mask