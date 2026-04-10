from __future__ import annotations
from typing import Sequence
import torch
from torch import nn
from .mil_base import MILBase

class MultimodalMILModel(MILBase):
    def __init__(self, embed_dim: int = 256):
        super().__init__(embed_dim)
        self.tab_mlp = nn.Linear(embed_dim, embed_dim)
    def forward(self, bag: Sequence[torch.Tensor], tab: torch.Tensor | None = None):
        logits = super().forward(bag)
        if tab is not None:
            logits = logits + self.tab_mlp(tab)
        return logits