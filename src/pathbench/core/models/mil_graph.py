from __future__ import annotations
import torch
from torch import nn
from .mil_base import MILModel

class GraphMILModel(MILModel):
    """Stub: later, wrap torch_geometric layers (kept optional via extras)."""
    def __init__(self, embed_dim: int = 256, lr: float = 1e-3):
        super().__init__(embed_dim, lr)