from __future__ import annotations
from typing import Any, Sequence
import torch
from .mil_base import MILModel


class EnsembleMILModel(MILModel):
    def __init__(self, members: Sequence[MILModel]):
        super().__init__()
        self.members = torch.nn.ModuleList(members)

    @property
    def bag_size(self) -> int | None:
        return None

    def forward_bag(
        self,
        bag: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        logits = [member(bag, *args, **kwargs) for member in self.members]
        normalized_logits = [
            output["logits"] if isinstance(output, dict) else output
            for output in logits
        ]
        return torch.stack(normalized_logits).mean(dim=0)
