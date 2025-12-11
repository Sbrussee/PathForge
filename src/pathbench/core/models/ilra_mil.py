import torch
import torch.nn as nn
from typing import Optional
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.models.layers import NystromAttention
from pathbench.core.registry import MODELS

@MODELS.register("ILRA_MIL")
class ILRA_MIL(MILModelBase):
    """Implicit Low-Rank Attention MIL."""
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=2, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.attn = NystromAttention(dim=hidden_dim, head=8, num_landmarks=hidden_dim // 8)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    @property
    def bag_size(self): return None

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, label=None, loss_fn=None, return_attention=False) -> Union[torch.Tensor, Dict]:
        h = self.fc(bag)
        h_attn = self.attn(self.norm(h), mask=mask)
        h = h + h_attn
        
        if mask is not None:
            h = h * mask.unsqueeze(-1)
            den = mask.sum(dim=1, keepdim=True) + 1e-6
            z = h.sum(dim=1) / den
        else:
            z = h.mean(dim=1)
            
        logits = self.classifier(z)
        
        results = {"logits": logits}
        if loss_fn is not None and label is not None:
            results["loss"] = loss_fn(logits, label)
        if return_attention:
            results["attention"] = None 
            
        if len(results) == 1: return logits
        return results