"""Paper-faithful and bug-fixed CDA baseline architecture (`model.py` in `WHUer-cloud/CDA`).

Key properties verified against `https://github.com/WHUer-cloud/CDA`:
1. Exact same module hierarchy and parameter names (`state_dict` 100% compatible).
2. Exact same parameter counts:
   - `TeacherEncoder`: `24,862,528` (frozen)
   - `GCM`: `19,704,579` (`fusion`: `1,384,451`, `local1`: `2,040,064`, `local2`: `3,679,232`, `merge`: `3,146,752`, `res`: `9,454,080`)
   - `GSUDecoder`: `6,959,616` (`l3`: `5,289,984`, `l2`: `1,334,272`, `l1`: `335,360`)
   - Cross-scale adapters `c11..c32`: `16,518,656`
   - Total `GCMAndDecoder`: `43,182,851`
3. Fixes the import-time global `device` binding bug so `.to(device)` works cleanly on CPU/CUDA.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from .sga import SGAFusion


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (`model.py: RMSNorm`)."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GroupedLinear(nn.Module):
    """Channel-grouped linear projection (`model.py: GroupedLinear`)."""

    def __init__(self, in_f: int, out_f: int, groups: int) -> None:
        super().__init__()
        if in_f % groups != 0 or out_f % groups != 0:
            raise ValueError(f"in_f ({in_f}) and out_f ({out_f}) must be divisible by groups ({groups})")
        self.g = groups
        self.ls = nn.ModuleList(
            [nn.Linear(in_f // groups, out_f // groups) for _ in range(groups)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, x.shape[-1] // self.g, dim=-1)
        return torch.cat([layer(chunk) for layer, chunk in zip(self.ls, chunks)], dim=-1)


class DepthwiseMLP(nn.Module):
    """Grouped linear MLP with SiLU activation (`model.py: DepthwiseMLP`)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        g = int(d**0.5)
        while d % g != 0:
            g -= 1
        self.dw = GroupedLinear(d, d, g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.dw(x))


class GSU(nn.Module):
    """Gated-depth Sequence Unit (`model.py: GSU`)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d, 2 * d)
        self.conv1d = nn.Conv1d(d, d, kernel_size=4, groups=d, padding=3)
        self.dw_mlp = DepthwiseMLP(d)
        self.norm = RMSNorm(d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.shape
        x_branch, z_gate = self.in_proj(x).chunk(2, dim=-1)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_branch = self.dw_mlp(x_branch)
        return self.out_proj(self.norm(x_branch) * F.silu(z_gate))


class GSULayer(nn.Module):
    """Pre-RMSNorm residual wrapper around GSU (`model.py: GSULayer`)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.gsu = GSU(d)
        self.norm = RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gsu(self.norm(x))


class ConvBlock(nn.Module):
    """Residual bottleneck convolutional block (`model.py: ConvBlock`)."""

    def __init__(self, c_in: int, k: int, filters: Sequence[int], stride: int) -> None:
        super().__init__()
        f1, f2, f3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(c_in, f1, kernel_size=1, stride=stride),
            nn.BatchNorm2d(f1),
            nn.GELU(),
            nn.Conv2d(f1, f2, kernel_size=k, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.GELU(),
            nn.Conv2d(f2, f3, kernel_size=1, stride=1),
            nn.BatchNorm2d(f3),
            nn.GELU(),
        )
        self.short = nn.Sequential(
            nn.Conv2d(c_in, f3, kernel_size=1, stride=stride),
            nn.BatchNorm2d(f3),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x) + self.short(x)


class PatchExpand2D(nn.Module):
    """2x bilinear spatial upsampling + linear channel reduction (`model.py: PatchExpand2D`)."""

    def __init__(self, dim: int, scale: int = 2) -> None:
        super().__init__()
        self.expand = nn.Linear(dim, dim // scale)
        self.norm = nn.LayerNorm(dim // scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.norm(self.expand(x))
        return x


class TeacherEncoder(nn.Module):
    """Frozen WideResNet-50-2 teacher extracting `layer1`, `layer2`, `layer3` (`model.py: Encoder`)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = Wide_ResNet50_2_Weights.DEFAULT if pretrained else None
        backbone = wide_resnet50_2(weights=weights)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def train(self, mode: bool = True) -> TeacherEncoder:
        """Keeps teacher BatchNorm statistics frozen in eval mode even if `.train()` is called."""
        return super().train(False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        e1 = self.layer1(x)   # (B, 256, H/4, W/4)
        e2 = self.layer2(e1)  # (B, 512, H/8, W/8)
        e3 = self.layer3(e2)  # (B, 1024, H/16, W/16)
        return e1, e2, e3


class GCM(nn.Module):
    """Global-local Compression Module combining SGA (`fusion`) and CLE (`local1, local2, merge`)."""

    def __init__(self) -> None:
        super().__init__()
        self.fusion = SGAFusion()
        self.local1 = ConvBlock(256, 3, [256, 512, 1024], stride=4)
        self.local2 = ConvBlock(512, 3, [512, 512, 1024], stride=2)
        self.merge = nn.Conv2d(3072, 1024, kernel_size=1)
        self.res = ConvBlock(1024, 3, [512, 1024, 2048], stride=2)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
    ) -> torch.Tensor:
        g = self.fusion(x1, x2, x3)
        l1 = self.local1(x1)
        l2 = self.local2(x2)
        l_feat = self.merge(torch.cat([l1, l2, x3], dim=1))
        return self.res(g + l_feat)


class GSUDecoder(nn.Module):
    """3-stage lightweight student decoder (`model.py: Decoder`)."""

    def __init__(self) -> None:
        super().__init__()
        self.l3 = nn.Sequential(PatchExpand2D(2048, 2), GSULayer(1024))
        self.l2 = nn.Sequential(PatchExpand2D(1024, 2), GSULayer(512))
        self.l1 = nn.Sequential(PatchExpand2D(512, 2), GSULayer(256))

    @staticmethod
    def to_2d(x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        side = int(math.isqrt(seq_len))
        return x.permute(0, 2, 1).contiguous().view(bsz, channels, side, side)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        s3 = self.to_2d(self.l3(x))   # (B, 1024, H/16, W/16)
        s2 = self.to_2d(self.l2(s3))  # (B, 512, H/8, W/8)
        s1 = self.to_2d(self.l1(s2))  # (B, 256, H/4, W/4)
        return s1, s2, s3, s2, s3, s1, s3, s1, s2


class GCMAndDecoder(nn.Module):
    """Complete CDA Student network (`GCM` + `Decoder` + Cross-Scale Alignment `c11..c32`)."""

    def __init__(self) -> None:
        super().__init__()
        self.gcm = GCM()
        self.dec = GSUDecoder()
        # Cross-scale channel alignment convolutions (16,518,656 parameters in baseline)
        self.c11 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.c12 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.c21 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.c22 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.c31 = nn.Conv2d(256, 1024, kernel_size=3, stride=1, padding=1)
        self.c32 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        h1, w1 = e1.shape[2], e1.shape[3]
        h2, w2 = e2.shape[2], e2.shape[3]
        h3, w3 = e3.shape[2], e3.shape[3]

        f = self.gcm(e1, e2, e3)
        s1, s2, s3, l1m, l1n, l2m, l2n, l3m, l3n = self.dec(f)

        l1n = self.c12(F.interpolate(l1n, size=(h1, w1), mode="bilinear", align_corners=True))
        l1m = self.c11(F.interpolate(l1m, size=(h1, w1), mode="bilinear", align_corners=True))
        l2m = self.c21(F.interpolate(l2m, size=(h2, w2), mode="bilinear", align_corners=True))
        l2n = self.c22(F.interpolate(l2n, size=(h2, w2), mode="bilinear", align_corners=True))
        l3m = self.c31(F.interpolate(l3m, size=(h3, w3), mode="bilinear", align_corners=True))
        l3n = self.c32(F.interpolate(l3n, size=(h3, w3), mode="bilinear", align_corners=True))

        return s1, s2, s3, l1m, l1n, l2m, l2n, l3m, l3n
