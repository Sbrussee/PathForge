# =============================================================================
# 8. GCNConvMIL (GCN + Attention Pooling)
# =============================================================================
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.registry import MODELS

@MODELS.register("GCNConvMIL")
class GCNConvMIL(MILModelBase):
    r"""
    Graph Convolutional Network MIL.
    
    Uses simple GCN layers on a KNN graph followed by global attention pooling.
    
    **Mathematical Formulation:**
    $$ H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}) $$
    Where $\tilde{A}$ is the adjacency matrix from KNN.
    """
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        try:
            from torch_geometric.nn import GCNConv, GlobalAttention
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))
            self.has_pyg = True
        except ImportError:
            self.has_pyg = False
        self.classifier = nn.Linear(hidden_dim, output_dim)

    @property
    def bag_size(self): return None

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, label=None, loss_fn=None, return_attention=False) -> Union[torch.Tensor, Dict]:
        assert coords is not None, "GCNConvMIL requires spatial coordinates."
        
        if not self.has_pyg: return self.classifier(self.fc(bag).mean(dim=1))
        
        import torch_geometric.nn as gnn
        outputs = []
        attentions = []
        
        for i in range(bag.shape[0]):
            xi = bag[i]
            if mask is not None: xi = xi[mask[i]]
            pos = coords[i]
            if mask is not None: pos = pos[mask[i]]
            
            xi = self.fc(xi)
            edge_index = gnn.knn_graph(pos, k=8, loop=True)
            h = F.relu(self.conv1(xi, edge_index))
            h = F.relu(self.conv2(h, edge_index))
            
            gate = self.pool.gate_nn(h).view(-1, 1)
            gate = F.softmax(gate, dim=0)
            if return_attention:
                attentions.append(gate.squeeze(1))
            h_pool = (gate * h).sum(dim=0, keepdim=True)
            
            outputs.append(self.classifier(h_pool))
            
        logits = torch.cat(outputs, dim=0)
        
        results = {"logits": logits}
        if loss_fn is not None and label is not None:
            results["loss"] = loss_fn(logits, label)
        if return_attention:
            results["attention"] = attentions
            
        if len(results) == 1: return logits
        return results
