from typing import Optional, Dict, Union
import torch
import torch.nn as nn
from pathbench.core.models.slide_base import SlideLevelModel
from pathbench.core.registry import MODELS

@MODELS.register("SlideVectorMLP")
class SlideVectorMLP(SlideLevelModel):
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward_slide(self, x: torch.Tensor, label: Optional[torch.Tensor] = None, loss_fn: Optional[nn.Module] = None) -> Union[torch.Tensor, Dict]:
        logits = self.mlp(x)
        if loss_fn is not None and label is not None:
            return {"logits": logits, "loss": loss_fn(logits, label)}
        return logits