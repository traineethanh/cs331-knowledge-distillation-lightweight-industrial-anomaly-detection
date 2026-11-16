"""Spatial-guided Global Attention (SGA) module (`Fusion` in official `Transformer.py`).

Reconstructs the exact mathematical operations and `state_dict` parameter names of
`Transformer.py` from `https://github.com/WHUer-cloud/CDA` while fixing:
1. Dynamic spatial grid resolution `(h, w)` instead of hard-coded `16x16` (`256`).
2. Proper device propagation via input tensor `.device`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    """Two-stage 1x1 point-wise channel compression block (`1536 -> 512 -> 1024`)."""

    def __init__(self, in_channels: int = 1536) -> None:
        super().__init__(
            nn.Conv2d(in_channels, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.GELU(),
        )


class RoPESelfAttention(nn.Module):
    """Multi-head spatial token attention with 1D Rotary Position Embedding (RoPE).

    Preserves exact weight names (`w_q`, `w_k`, `w_v`, `fc`) and mathematical
    behavior from official `Transformer.py: SelfAttention`.
    """

    def __init__(self, hidden_size: int = 256, n_heads: int = 8) -> None:
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})")
        self.hid_dim = hidden_size
        self.n_heads = n_heads

        self.w_q = nn.Linear(1, hidden_size)
        self.w_k = nn.Linear(1, hidden_size)
        self.w_v = nn.Linear(1, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def sinusoidal_position_embedding(
        batch_size: int,
        nums_head: int,
        max_len: int,
        output_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.arange(0, max_len, dtype=dtype, device=device).unsqueeze(-1)
        ids = torch.arange(0, output_dim // 2, dtype=dtype, device=device)
        theta = torch.pow(10000.0, -2.0 * ids / output_dim)

        embeddings = position * theta
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
        return embeddings

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        batch_size, nums_head, max_len, output_dim = q.shape

        pos_emb = self.sinusoidal_position_embedding(
            batch_size, nums_head, max_len, output_dim, q.device, q.dtype
        )
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape)
        q = q * cos_pos + q2 * sin_pos

        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape)
        k = k * cos_pos + k2 * sin_pos

        return q.squeeze(1), k.squeeze(1)

    def forward(
        self,
        query_tokens: torch.Tensor,
        key_value_tokens: torch.Tensor,
        spatial_hw: Tuple[int, int] = (16, 16),
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = query_tokens.shape[0]
        h, w = spatial_hw

        q = self.w_q(query_tokens)
        k = self.w_k(key_value_tokens)
        q, k = self.apply_rope(q, k)
        v = self.w_v(key_value_tokens)

        head_dim = self.hid_dim // self.n_heads
        q = q.view(bsz, -1, self.n_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(bsz, -1, self.n_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(bsz, -1, self.n_heads, head_dim).permute(0, 2, 1, 3)

        # Matches official Transformer.py line 147 (`scaling = K.shape[2]`)
        scaling = k.shape[2]
        attention = torch.matmul(q, k.permute(0, 1, 3, 2)) / math.sqrt(scaling)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(attention, dim=-1)
        x = torch.matmul(attention, v)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.hid_dim)
        x = self.fc(x)
        return x.view(bsz, self.hid_dim, h, w)


class SGAFusion(nn.Module):
    """Spatial-guided Global Attention (SGA) module (`Fusion` in `Transformer.py`)."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = RoPESelfAttention(256, n_heads=8)
        self.conv1 = nn.Conv2d(256, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1)
        self.conv3 = nn.Conv2d(1024, 1, kernel_size=1)
        self.DoubleConv = DoubleConv(1536)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = x3.shape
        seq_len = h * w

        p1 = F.interpolate(x1, size=(h, w), mode="bilinear")
        p2 = F.interpolate(x2, size=(h, w), mode="bilinear")

        t1 = self.conv1(p1).view(b, 1, seq_len).permute(0, 2, 1)
        t2 = self.conv2(p2).view(b, 1, seq_len).permute(0, 2, 1)
        t3 = self.conv3(x3).view(b, 1, seq_len).permute(0, 2, 1)

        tokens = [t1, t2, t3]
        attn_maps = []
        for i in range(len(tokens)):
            for j in range(i, len(tokens)):
                attn_maps.append(self.transformer(tokens[i], tokens[j], spatial_hw=(h, w)))

        out = torch.cat(attn_maps, dim=1)
        return self.DoubleConv(out)
