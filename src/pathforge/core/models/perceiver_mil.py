# =============================================================================
# Perceiver MIL
# =============================================================================
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
from pathforge.core.models.mil_base import MILModelBase
from pathforge.core.registry import MODELS

@MODELS.register("PerceiverMIL")
class PerceiverMIL(MILModelBase):
    r"""
    Perceiver-based MIL.
    
    Uses a fixed set of latent query vectors to attend to the variable-sized bag input 
    via Cross-Attention, mapping $O(N)$ complexity to $O(1)$ fixed latents.
    
    **Mathematical Formulation:**
    Latents $L \in \mathbb{R}^{M \times D}$. Bag $H \in \mathbb{R}^{N \times D}$.
    $$ O = \text{CrossAttn}(Q=L, K=H, V=H) $$
    $$ z = \text{Mean}(O) $$
    """
    def __init__(self, input_dim=1024, num_latents=32, latent_dim=256, output_dim=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, kdim=input_dim, vdim=input_dim, batch_first=True)
        self.classifier = nn.Linear(latent_dim, output_dim)

    @property
    def bag_size(self): return None

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, label=None, loss_fn=None, return_attention=False) -> Union[torch.Tensor, Dict]:
        B = bag.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        key_padding_mask = ~mask if mask is not None else None
        
        out, attn_weights = self.cross_attn(query=latents, key=bag, value=bag, key_padding_mask=key_padding_mask, need_weights=return_attention)
        logits = self.classifier(out.mean(dim=1))
        
        results = {"logits": logits}
        if loss_fn is not None and label is not None:
            results["loss"] = loss_fn(logits, label)
        if return_attention:
            results["attention"] = attn_weights
            
        if len(results) == 1:
            return logits
        return results
