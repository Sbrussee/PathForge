from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.models.layers import Attn_Net_Gated
from pathbench.core.registry import MODELS

# =============================================================================
# Clam MIL Model
# =============================================================================

@MODELS.register("CLAM_SB")
class CLAM_SB(MILModelBase):
    """
    Clustering-constrained Attention MIL (Single Branch).
    """
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=2, dropout=0.1, k_sample=8):
        super().__init__()
        self.k_sample = k_sample
        self.output_dim = output_dim
        self.attention_net = Attn_Net_Gated(L=input_dim, D=hidden_dim, dropout=dropout>0, n_classes=1)
        self.classifiers = nn.Linear(input_dim, output_dim) 
        self.instance_classifiers = nn.Linear(input_dim, output_dim)

    @property
    def bag_size(self): return None

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()

    def inst_eval(self, A, h, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        return logits, all_targets

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None, label=None, loss_fn=None, return_attention=False) -> Union[torch.Tensor, Dict]:
        total_loss = 0.0
        logits_list = []
        attentions_list = []
        
        for i in range(bag.shape[0]):
            b = bag[i]
            if mask is not None:
                b = b[mask[i]]
            
            A = self.attention_net(b) # (N, 1)
            A = torch.transpose(A, 1, 0) # (1, N)
            A_raw = A
            A = F.softmax(A, dim=1)
            if return_attention:
                attentions_list.append(A)
            
            M = torch.mm(A, b) # (1, D)
            logits = self.classifiers(M)
            logits_list.append(logits)

            if loss_fn is not None and label is not None:
                assert label.ndim > 0
                bag_loss = loss_fn(logits, label[i:i+1])
                inst_logits, inst_targets = self.inst_eval(A_raw, b, self.instance_classifiers)
                instance_loss = loss_fn(inst_logits, inst_targets)
                total_loss += 0.7 * bag_loss + 0.3 * instance_loss

        logits_final = torch.cat(logits_list, dim=0)
        
        results = {"logits": logits_final}
        if loss_fn is not None and label is not None:
            results["loss"] = total_loss / bag.shape[0]
        if return_attention:
            results["attention"] = attentions_list if len(attentions_list) > 1 else attentions_list[0]
            
        if len(results) == 1:
            return logits_final
        return results
