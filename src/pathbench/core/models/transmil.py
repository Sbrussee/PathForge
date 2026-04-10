from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.models.layers import TransLayer, PPEG
from pathbench.core.registry import MODELS

# =============================================================================
# 3. TransMIL
# =============================================================================
@MODELS.register("TransMIL")
class TransMIL(MILModelBase):
    """
    TransMIL: Transformer based Correlated Multiple Instance Learning.
    
    Introduces PPEG (Pyramid Position Encoding Generator) and uses Nyström attention 
    to handle long sequences of patches.
    
    **Mathematical Formulation:**
    1. **Squaring/PPEG:** $H_{pos} = \text{PPEG}(H)$
    2. **Transformer:** $H_{trans} = \text{NystromBlock}(H_{pos})$
    3. **Aggregation:** Use a learnable [CLS] token.
    
    $$ z = \text{LayerNorm}(H_{trans}^{[CLS]}) $$
    """
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=2, dropout=0.1):
        super().__init__()
        self.pos_layer = PPEG(dim=hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.layer1 = TransLayer(dim=hidden_dim)
        self.layer2 = TransLayer(dim=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    @property
    def bag_size(self): return None

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, label=None, loss_fn=None, return_attention=False) -> Union[torch.Tensor, Dict]:
        h = self.fc(bag)
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h) 
        h = self.pos_layer(h)
        h = self.layer2(h)
        h = self.norm(h)
        logits = self.classifier(h[:, 0])
        
        results = {"logits": logits}
        if loss_fn is not None and label is not None:
            results["loss"] = loss_fn(logits, label)
        if return_attention:
            results["attention"] = None 
            
        if len(results) == 1:
            return logits
        return results
