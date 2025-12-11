
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathbench.core.models.mil_base import MILModelBase
from pathbench.core.registry import MODELS
from mamba import Mamba, BiMamba, SRMamba
from pathbench.core.models.layers import GlobalAttention, GlobalGatedAttention

@MODELS.register("MambaMIL")
class MambaMIL(MILModelBase):
    """
    MambaMIL with Attention Pooling (Gated or Standard).
    Combines Mamba's long-range dependency modeling with AB-MIL's interpretability.
    """
    def __init__(self, 
                 input_dim=1024, 
                 hidden_dim=512, 
                 output_dim=2, 
                 dropout=0.1, 
                 act='relu', 
                 layer=2, 
                 rate=10, 
                 type="SRMamba",
                 survival=False,
                 gate=True,      # Enable Gated Attention
                 attn_dim=256):  # Internal dim for attention mechanism
        
        super().__init__()
        self.survival = survival
        self.gate = gate
        
        # 1. Projection
        layers = [nn.Linear(input_dim, hidden_dim)]
        if act.lower() == 'relu': layers.append(nn.ReLU())
        elif act.lower() == 'gelu': layers.append(nn.GELU())
        if dropout > 0: layers.append(nn.Dropout(dropout))
        self._fc1 = nn.Sequential(*layers)
        
        # 2. Mamba Layers
        self.layers = nn.ModuleList()
        for i in range(layer):
            if type == 'SRMamba':
                self.layers.append(SRMamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2, rate=rate))
            elif type == 'BiMamba':
                self.layers.append(BiMamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2))
            elif type == 'Mamba':
                self.layers.append(Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2))
            else:
                raise ValueError(f"Unknown Mamba type: {type}")

        # 3. Normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 4. Attention Pooling Mechanism
        # We set num_classes=1 to learn a single attention score per patch
        # to aggregate features into one bag representation.
        if self.gate:
            self.attn_net = GlobalGatedAttention(L=hidden_dim, D=attn_dim, dropout=dropout, num_classes=1)
        else:
            self.attn_net = GlobalAttention(L=hidden_dim, D=attn_dim, dropout=dropout, num_classes=1)
        
        # 5. Classifier
        final_dim = 1 if survival else output_dim
        self.classifier = nn.Linear(hidden_dim, final_dim)

        initialize_weights(self)

    @property
    def bag_size(self): return None

    def forward_bag(self, 
                    bag: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, 
                    coords: Optional[torch.Tensor] = None, 
                    label: Optional[torch.Tensor] = None, 
                    loss_fn=None) -> Union[torch.Tensor, Dict]:
        
        # bag: (B, N, D)
        
        # 1. Project & Mamba Sequence Processing
        x = self._fc1(bag)       # (B, N, Hidden)
        for layer in self.layers:
            x = layer(x)         # (B, N, Hidden) - Mamba keeps shape
        x = self.norm(x)

        # 2. Calculate Attention Scores
        # A_raw: (B, N, 1) - Unnormalized logits
        A_raw = self.attn_net(x) 

        # 3. Handle Masking (Critical for Attention)
        # If we have padding, we must set their attention logits to -inf
        # so they result in 0 weight after softmax.
        if mask is not None:
            # mask: (B, N) -> (B, N, 1)
            mask_bool = mask.unsqueeze(-1).bool()
            # Set padded areas to very small number
            A_raw = A_raw.masked_fill(~mask_bool, torch.tensor(-1e9))

        # 4. Softmax -> Attention Weights
        A = torch.softmax(A_raw, dim=1) # (B, N, 1)

        # 5. Weighted Sum (Pooling)
        # Sum( Weights * Features )
        z = torch.sum(A * x, dim=1) # (B, Hidden)

        # 6. Classification
        logits = self.classifier(z) # (B, Out)

        # 7. Return Results
        results = {"logits": logits}
        results["attention"] = A # Return attention weights for visualization/interpretability
        
        if self.survival:
            results["risk"] = logits
        else:
            results["Y_prob"] = F.softmax(logits, dim=1)
            results["Y_hat"] = torch.argmax(results["Y_prob"], dim=1)

        if loss_fn is not None and label is not None:
             results["loss"] = loss_fn(logits, label)

        if len(results) == 1: return logits
        return results