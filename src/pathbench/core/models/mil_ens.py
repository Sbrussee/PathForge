from __future__ import annotations
from typing import Sequence
import torch
from .mil_base import MILModel

class EnsembleMILModel(MILModel):
    def __init__(self, members: Sequence[MILModel]):
        super().__init__()
        self.members = torch.nn.ModuleList(members)
    def forward(self, bag):
        logits = [m(bag) for m in self.members]
        return torch.stack(logits).mean(dim=0)