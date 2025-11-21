# =============================================================================
# RRT-MIL
# =============================================================================
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.models.layers import StandardTransformerBlock
from pathbench.core.models.utils import perform_kmeans
from pathbench.core.registry import MODELS

@MODELS.register("RRT_MIL")
class RRT_MIL(MILModelBase):
    """
    Region-based Relation Transformer MIL.
    
    Clusters instances into regions and models relationships between regions using a Transformer.
    
    **Mathematical Formulation:**
    1. **Clustering:** $C_1, \dots, C_k = \text{KMeans}(H)$
    2. **Region Rep:** $r_j = \text{MaxPool}(\{h_i | i \in C_j\})$
    3. **Relation:** $R' = \text{Transformer}(R)$
    4. **Agg:** $z = \text{Mean}(R')$
    """
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=2, num_regions=16, depth=2, heads=4):
        super().__init__()
        self.num_regions = num_regions
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.ModuleList([StandardTransformerBlock(hidden_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        
    @property
    def bag_size(self): return None

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, label=None, loss_fn=None, return_attention=False) -> Union[torch.Tensor, Dict]:
        outputs = []
        for i in range(bag.shape[0]):
            x = bag[i]
            if mask is not None: x = x[mask[i]]
            c = coords[i] if coords is not None else None
            
            x_emb = self.fc(x)
            features_to_cluster = c if c is not None else x_emb.detach()
            
            k = min(self.num_regions, x.shape[0])
            assigns, _ = perform_kmeans(features_to_cluster, k)
            
            regions = []
            for cluster_idx in range(k):
                region_mask = (assigns == cluster_idx)
                if region_mask.sum() > 0:
                    regions.append(x_emb[region_mask].max(dim=0)[0])
                else:
                    regions.append(torch.zeros_like(x_emb[0]))
            
            regions = torch.stack(regions).unsqueeze(0)
            for layer in self.transformer:
                regions = layer(regions)
            regions = self.norm(regions)
            slide_rep = regions.mean(dim=1)
            outputs.append(self.classifier(slide_rep))
            
        logits = torch.cat(outputs, dim=0)
        
        results = {"logits": logits}
        if loss_fn is not None and label is not None:
            results["loss"] = loss_fn(logits, label)
        if return_attention:
            results["attention"] = None 
            
        if len(results) == 1: return logits
        return results