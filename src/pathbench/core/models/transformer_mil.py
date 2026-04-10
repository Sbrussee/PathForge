from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.models.layers import StandardTransformerBlock
from pathbench.core.registry import MODELS

# =============================================================================
# 4. Transformer MIL (Standard)
# =============================================================================
@MODELS.register("TransformerMIL")
@MODELS.register("Transformer")
class TransformerMIL(MILModelBase):
    """
    Standard Transformer Encoder for MIL.
    
    Uses standard $O(N^2)$ self-attention. Best for smaller bag sizes or rigid tile grids.
    
    **Mathematical Formulation:**
    $$ Q, K, V = W_q H, W_k H, W_v H $$
    $$ \text{Attn}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}}) V $$
    Aggregates via [CLS] token.
    """
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=2, depth=2, heads=8, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.layers = nn.ModuleList([
            StandardTransformerBlock(hidden_dim, heads, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    @property
    def bag_size(self): return None

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, label=None, loss_fn=None, return_attention=False) -> Union[torch.Tensor, Dict]:
        x = self.fc(bag)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if mask is not None:
            cls_mask = torch.ones((B, 1), device=mask.device, dtype=mask.dtype)
            mask = torch.cat((cls_mask, mask), dim=1)

        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.classifier(x[:, 0])
        
        results = {"logits": logits}
        if loss_fn is not None and label is not None:
            results["loss"] = loss_fn(logits, label)
        if return_attention:
            results["attention"] = None
            
        if len(results) == 1:
            return logits
        return results
