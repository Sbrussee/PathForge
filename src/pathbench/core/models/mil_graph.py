from __future__ import annotations
import torch
from torch import nn
from .mil_base import MILModel


class GraphMILModel(MILModel):
    """Stub: later, wrap torch_geometric layers (kept optional via extras)."""

    def __init__(self, embed_dim: int = 256, lr: float = 1e-3):
        super().__init__()
        self.embed_dim = embed_dim
        self.lr = lr
        self.projection = nn.Identity()

    @property
    def bag_size(self) -> int | None:
        return None

    def forward_bag(
        self,
        bag: torch.Tensor,
        mask: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        if mask is None:
            pooled = bag.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1).float()
            pooled = (bag * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)
