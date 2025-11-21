
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.models.layers import Attn_Net_Gated
from pathbench.core.registry import MODELS

# =============================================================================
# 1. Attention MIL (Ilse et al. 2018)
# =============================================================================
@MODELS.register("AttentionMIL")
class AttentionMIL(MILModelBase):
    """
    Attention-based Deep Multiple Instance Learning (Ilse et al., 2018).
    
    Uses a gated attention mechanism to learn a weighted average of instance embeddings.
    
    **Mathematical Formulation:**
    Let $H = \{h_1, \dots, h_N\}$ be the bag of $N$ instance embeddings.
    The attention mechanism computes weights $a_i$:
    
    $$ a_i = \\frac{\\exp(w^T (\\tanh(V h_i^T) \\odot \\text{sigm}(U h_i^T)))}{\\sum_{j=1}^N \\exp(w^T (\\tanh(V h_j^T) \\odot \\text{sigm}(U h_j^T)))} $$
    
    The slide representation $z$ is:
    $$ z = \\sum_{i=1}^N a_i h_i $$
    
    The final prediction is $\\text{Classifier}(z)$.
    """
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=2, dropout=0.1):
        super().__init__()
        self.attention = Attn_Net_Gated(L=input_dim, D=hidden_dim, dropout=dropout > 0, n_classes=1)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    @property
    def bag_size(self): return None

    def forward_bag(self, bag: torch.Tensor, mask: Optional[torch.Tensor] = None, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        A = self.attention(bag)  # (B, N, 1)
        A = torch.transpose(A, 2, 1)  # (B, 1, N)
        if mask is not None:
            A.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        A = F.softmax(A, dim=2)
        M = torch.bmm(A, bag).squeeze(1)  # (B, D)
        return self.classifier(M)