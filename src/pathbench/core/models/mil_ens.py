from __future__ import annotations
from typing import Any, Sequence
import torch
from .mil_base import MILModel

class EnsembleMILModel(MILModel):
    """Average the predictions from multiple MIL members.

    This lightweight utility model is intentionally not registry-exposed for
    benchmark selection. It remains available as an importable composition
    helper and as interface coverage for the shared MIL base classes.
    """

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
