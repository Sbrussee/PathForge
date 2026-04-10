#TO DO: CHECK IF IMPLEMENTATION IS CORRECT
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.registry import MODELS

@MODELS.register("WiKG_MIL")
class WiKG_MIL(MILModelBase):
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=2, k=8, heads=4):
        super().__init__()
        self.k = k
        self.fc = nn.Linear(input_dim, hidden_dim)
        try:
            from torch_geometric.nn import GATConv, GlobalAttention
            self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
            self.pool = GlobalAttention(gate_nn=nn.Linear(hidden_dim, 1))
            self.has_pyg = True
        except ImportError:
            self.has_pyg = False
        self.classifier = nn.Linear(hidden_dim, output_dim)

    @property
    def bag_size(self): return None

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, label=None, loss_fn=None, return_attention=False) -> Union[torch.Tensor, Dict]:
        assert coords is not None, "WiKG-MIL requires spatial coordinates (coords)."
        
        if not self.has_pyg:
            return self.classifier(self.fc(bag).mean(dim=1))
        
        import torch_geometric.nn as gnn
        outputs = []
        attentions = []
        
        for i in range(bag.shape[0]):
            x = bag[i]
            if mask is not None:
                x = x[mask[i]]
            pos = coords[i]
            if mask is not None:
                pos = pos[mask[i]]
            
            x_emb = self.fc(x)
            edge_index = gnn.knn_graph(pos, k=self.k, loop=True)
            h = F.relu(self.conv1(x_emb, edge_index))
            h = F.relu(self.conv2(h, edge_index))
            
            gate = self.pool.gate_nn(h).view(-1, 1)
            gate = F.softmax(gate, dim=0)
            if return_attention:
                attentions.append(gate.squeeze(1))
                
            h_slide = (gate * h).sum(dim=0, keepdim=True)
            outputs.append(self.classifier(h_slide))
            
        logits = torch.cat(outputs, dim=0)
        
        results = {"logits": logits}
        if loss_fn is not None and label is not None:
            results["loss"] = loss_fn(logits, label)
        if return_attention:
            results["attention"] = attentions
            
        if len(results) == 1:
            return logits
        return results
