import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

# =============================================================================
# MLP Utils
# =============================================================================

def create_mlp(
        in_dim=768, 
        hid_dims=[512, 512], 
        out_dim=512, 
        act=nn.ReLU(),
        dropout=0.,
        end_with_fc=True, 
        end_with_dropout=False,
        bias=True
    ):

    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    elif len(hid_dims) >= 0:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
        if end_with_dropout:
            layers.append(nn.Dropout(dropout))
        mlp = nn.Sequential(*layers)
    return mlp


# =============================================================================
# Attention Networks (Ilse et al.)
# =============================================================================

class GlobalAttention(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    """
    def __init__(self, L=1024, D=256, dropout=0., num_classes=1):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, num_classes)
        )

    def forward(self, x):
        return self.module(x)  # N x num_classes


class GlobalGatedAttention(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    """
    def __init__(self, L=1024, D=256, dropout=0., num_classes=1):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.attention_c = nn.Linear(D, num_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x num_classes
        return A

# Alias used by AttentionMIL
Attn_Net_Gated = GlobalGatedAttention


# =============================================================================
# Standard Transformer Components
# =============================================================================

class StandardTransformerBlock(nn.Module):
    """Standard Self-Attention Block for Transformer MIL."""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: B, N, D
        # mask: B, N (True = keep, False = ignore)
        key_padding_mask = ~mask if mask is not None else None
        
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# TransMIL / Nystrom Components
# =============================================================================

# Try to import from external library, otherwise define local fallback
try:
    from nystrom_attention import NystromAttention as _LibNystromAttention
    HAS_NYSTROM_LIB = True
except ImportError:
    HAS_NYSTROM_LIB = False
    _LibNystromAttention = None


if HAS_NYSTROM_LIB:
    class NystromAttention(nn.Module):
        """
        Wrapper around the external `nystrom-attention` library to match PathBench API.
        """
        def __init__(self, dim, head=8, num_landmarks=64, dropout=0.1):
            super().__init__()
            self.attn = _LibNystromAttention(
                dim=dim,
                dim_head=dim // head,
                heads=head,
                num_landmarks=num_landmarks,
                dropout=dropout
            )

        def forward(self, x, mask=None):
            # Library expects mask where True = keep, matching our API
            return self.attn(x, mask=mask)

else:
    class NystromAttention(nn.Module):
        """
        Native PyTorch implementation of Nystrom Attention (O(N) complexity).
        Used when `nystrom-attention` library is not installed.
        """
        def __init__(self, dim, head=8, num_landmarks=64, dropout=0.1):
            super().__init__()
            self.head = head
            self.num_landmarks = num_landmarks
            
            self.head_dim = dim // head
            self.scale = self.head_dim ** -0.5

            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
            self.to_out = nn.Sequential(
                nn.Linear(dim, dim),
                nn.Dropout(dropout)
            )

        def forward(self, x, mask=None):
            b, n, c = x.shape
            h = self.head
            
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: t.reshape(b, n, h, -1).permute(0, 2, 1, 3), qkv)

            if mask is not None:
                mask = mask[:, None, :, None]
                v.masked_fill_(~mask, 0.)
            
            # Use fewer landmarks if sequence is shorter than num_landmarks
            if n < self.num_landmarks:
                q_landmarks = q
                k_landmarks = k
            else:
                q_landmarks = F.adaptive_avg_pool2d(q, (self.num_landmarks, self.head_dim))
                k_landmarks = F.adaptive_avg_pool2d(k, (self.num_landmarks, self.head_dim))

            # Nystrom Approximation
            kernel_1 = F.softmax(torch.matmul(q, k_landmarks.transpose(-1, -2)) * self.scale, dim = -1)
            kernel_2 = F.softmax(torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2)) * self.scale, dim = -1)
            kernel_3 = F.softmax(torch.matmul(q_landmarks, k.transpose(-1, -2)) * self.scale, dim = -1)

            # Robust Inverse
            try:
                kernel_2_inv = torch.linalg.pinv(kernel_2)
            except:
                # Fallback identity if singular
                kernel_2_inv = torch.eye(kernel_2.shape[-1], device=x.device).expand_as(kernel_2)

            out = torch.matmul(kernel_1, kernel_2_inv)
            out = torch.matmul(out, kernel_3)
            out = torch.matmul(out, v)
            
            out = out.permute(0, 2, 1, 3).reshape(b, n, c)
            return self.to_out(out)


class TransLayer(nn.Module):
    """
    Transformer Layer using Nystrom Attention.
    """
    def __init__(self, dim, head=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Uses the wrapper or local implementation defined above
        self.attn = NystromAttention(dim, head=head, num_landmarks=dim // head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PPEG(nn.Module):
    """
    Pyramid Position Encoding Generator (TransMIL).
    """
    def __init__(self, dim=512):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.proj1 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.proj2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        # x: B, N, D
        B, N, D = x.shape
        if N == 0: return x
        
        # Separate CLS token
        cls_token = x[:, 0:1, :] # B, 1, D
        feats = x[:, 1:, :]      # B, N-1, D
        
        if feats.shape[1] == 0:
            return x
            
        feats_t = feats.transpose(1, 2)
        pe = self.proj(feats_t) + self.proj1(feats_t) + self.proj2(feats_t)
        feats_t = feats_t + pe
        feats = feats_t.transpose(1, 2)
        
        return torch.cat((cls_token, feats), dim=1)