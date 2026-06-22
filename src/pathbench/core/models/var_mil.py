# =============================================================================
# VarMIL
# =============================================================================

from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.registry import MODELS

@MODELS.register("VarMIL")
class VarMIL(MILModelBase):
    r"""
    Variance-aware Multiple Instance Learning.
    
    Aggregates statistics using both the weighted mean and the weighted variance of the bag.
    
    **Mathematical Formulation:**
    Mean: $\mu = \frac{\sum a_i h_i}{\sum a_i}$
    Variance: $\sigma^2 = \frac{\sum a_i (h_i - \mu)^2}{\sum a_i}$
    Output: $\text{Classifier}([\mu, \sigma^2])$
    """
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    @property
    def bag_size(self): return None

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, label=None, loss_fn=None, return_attention=False) -> Union[torch.Tensor, Dict]:
        h = F.relu(self.fc(bag)) 
        a = self.attention(h) 
        if mask is not None:
            a = a * mask.unsqueeze(-1)
        
        sum_a = torch.sum(a, dim=1) + 1e-6
        mean = torch.sum(a * h, dim=1) / sum_a
        diff = h - mean.unsqueeze(1)
        var = torch.sum(a * (diff ** 2), dim=1) / sum_a
        
        logits = self.classifier(torch.cat([mean, var], dim=1))
        
        results = {"logits": logits}
        if loss_fn is not None and label is not None:
            results["loss"] = loss_fn(logits, label)
        if return_attention:
            results["attention"] = a
            
        if len(results) == 1:
            return logits
        return results
