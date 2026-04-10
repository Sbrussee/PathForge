# =============================================================================
# 9. DTFD-MIL
# =============================================================================
from typing import Optional, Dict, Union
import torch
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.models.attention_mil import AttentionMIL
from pathbench.core.registry import MODELS

@MODELS.register("DTFD_MIL")
class DTFD_MIL(MILModelBase):
    """
    Double-Tier Feature Distillation MIL.
    
    Uses a Tier-1 model to select the top-k most attended instances (pseudo-bags)
    and a Tier-2 model to classify them.
    
    **Mathematical Formulation:**
    Tier 1: $A_{tier1} = \text{Attn}(H)$. Select indices $I_{topk} = \text{topk}(A_{tier1})$.
    Tier 2: $z = \text{Attn}_{tier2}(H_{I_{topk}})$
    """
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=2):
        super().__init__()
        self.tier1 = AttentionMIL(input_dim, hidden_dim, output_dim)
        self.tier2 = AttentionMIL(input_dim, hidden_dim, output_dim)
        self.k = 10

    @property
    def bag_size(self): return None

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, label=None, loss_fn=None, return_attention=False) -> Union[torch.Tensor, Dict]:
        A = self.tier1.attention(bag)
        if mask is not None:
            A.masked_fill_(~mask.unsqueeze(-1), float("-inf"))
        
        _, idx = torch.topk(A.squeeze(-1), k=min(self.k, A.shape[1]), dim=1)
        pseudo_bag = torch.gather(bag, 1, idx.unsqueeze(-1).expand(-1, -1, bag.shape[-1]))
        
        tier2_out = self.tier2.forward_bag(pseudo_bag, label=label, loss_fn=loss_fn, return_attention=return_attention)
        
        if return_attention and isinstance(tier2_out, dict):
            tier2_out["attention"] = A 
        elif return_attention:
             tier2_out = {"logits": tier2_out, "attention": A}
             
        return tier2_out
