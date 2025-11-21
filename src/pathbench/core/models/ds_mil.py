from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.registry import MODELS

@MODELS.register("DSMIL")
class DSMIL(MILModelBase):
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.q_fc = nn.Linear(hidden_dim, hidden_dim)
        self.v_fc = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    @property
    def bag_size(self): return None

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, label=None, loss_fn=None, return_attention=False) -> Union[torch.Tensor, Dict]:
        feats = F.relu(self.fc(bag))
        max_feat, _ = torch.max(feats, dim=1)
        q = self.q_fc(max_feat).unsqueeze(1)
        v = self.v_fc(feats)
        A = torch.sum(q * v, dim=-1) / (feats.shape[-1] ** 0.5)
        if mask is not None:
            A.masked_fill_(~mask, float('-inf'))
        A = F.softmax(A, dim=1).unsqueeze(-1)
        M = torch.sum(feats * A, dim=1)
        logits = self.classifier(M)
        
        results = {"logits": logits}
        if loss_fn is not None and label is not None:
            results["loss"] = loss_fn(logits, label)
        if return_attention:
            results["attention"] = A.squeeze(-1)
            
        if len(results) == 1: return logits
        return results
