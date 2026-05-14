from __future__ import annotations
from typing import Any
import torch
from torch import nn
from .mil_base import MILBase


class MultimodalMILModel(MILBase):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.tab_mlp = nn.Linear(embed_dim, embed_dim)

    @property
    def bag_size(self) -> int | None:
        return None

    def forward_bag(
        self,
        bag: torch.Tensor,
        *args: Any,
        tab: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        logits = bag.mean(dim=1)
        if tab is not None:
            logits = logits + self.tab_mlp(tab)
        return logits
