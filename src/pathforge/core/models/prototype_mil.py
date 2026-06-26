# =============================================================================
# Prototype MIL
# =============================================================================
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
from pathforge.core.models.mil_base import MILModelBase
from pathforge.core.registry import MODELS

@MODELS.register("PrototypeMIL")
class PrototypeMIL(MILModelBase):
    r"""
    Prototype-based MIL.
    
    Learns a set of global prototypes. The bag representation is the similarity vector 
    indicating the presence of each prototype in the bag.
    
    **Mathematical Formulation:**
    Prototypes $P = \{p_1, \dots, p_k\}$.
    Similarity: $s_{i,j} = \exp(-||h_i - p_j||_2)$.
    Bag Feature: $z_j = \max_{i} s_{i,j}$ (presence of prototype $j$).
    """
    def __init__(self, input_dim=1024, num_prototypes=8, output_dim=2):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        self.classifier = nn.Linear(num_prototypes, output_dim)

    @property
    def bag_size(self): return None

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, label=None, loss_fn=None, return_attention=False) -> Union[torch.Tensor, Dict]:
        sim = torch.cdist(bag, self.prototypes.unsqueeze(0).expand(bag.shape[0], -1, -1))
        if mask is not None:
            sim.masked_fill_(~mask.unsqueeze(-1), float('inf'))
        sim = torch.exp(-sim) 
        proto_presence, _ = torch.max(sim, dim=1)
        logits = self.classifier(proto_presence)
        
        results = {"logits": logits}
        if loss_fn is not None and label is not None:
            results["loss"] = loss_fn(logits, label)
        if return_attention:
            results["attention"] = sim 
            
        if len(results) == 1:
            return logits
        return results
