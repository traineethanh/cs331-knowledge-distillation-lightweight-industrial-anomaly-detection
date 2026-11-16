"""Structural Compression Variants for Direction B (Luân): `B1_aux_csd`, `B2_sep_adapter`, `B3_cda_lite`.

Addresses the Cross-Scale Adapter Parameter Paradox identified in our audit of `WHUer-cloud/CDA`:
- `B0` (Official CDA Baseline): `43,182,851` trainable params (`59.86 GFLOPs`; `c11..c32` = `16,518,656` params [38.25%] & `44.69 GFLOPs` [74.66%]).
- `B1_aux_csd`: Uses `B0` during training, prunes `c11..c32` during inference (`26,664,195` active inference params [`-38.25%`], `15.17 GFLOPs` [`-74.66%`], zero retraining).
- `B2_sep_adapter`: Replaces dense 3x3 `c11..c32` (`16.52M`) with `DepthwiseSeparableAdapter` (`1.88M`), reducing total student params to `28,542,211` (`-33.90%`) and FLOPs to `20.27 GFLOPs` (`-66.13%`).
- `B3_cda_lite`: Combines `B2_sep_adapter` with `SeparableConvBlock` inside `GCM` (`local1, local2, res`), reducing total student params to `21,216,259` (`-50.87%`) and FLOPs to `18.13 GFLOPs` (`-69.71%`).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baseline.cda import ConvBlock, GCM, GCMAndDecoder, GSUDecoder
from ..baseline.sga import SGAFusion


class DepthwiseSeparableAdapter(nn.Module):
    """Factorizes dense 3x3 cross-scale alignment into 3x3 Depthwise + 1x1 Pointwise conv.

    Reduces adapter parameters by ~88.7% (from 16,518,656 down to 1,859,328 across c11..c32).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.act(self.bn(self.dw(x))))


class SeparableConvBlock(nn.Module):
    """Lightweight residual bottleneck block replacing dense 3x3 conv with DWConv3x3 + PWConv1x1."""

    def __init__(self, c_in: int, k: int, filters: Sequence[int], stride: int) -> None:
        super().__init__()
        f1, f2, f3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(c_in, f1, kernel_size=1, stride=stride),
            nn.BatchNorm2d(f1),
            nn.GELU(),
            nn.Conv2d(f1, f1, kernel_size=k, stride=1, padding=k // 2, groups=f1, bias=False),
            nn.BatchNorm2d(f1),
            nn.GELU(),
            nn.Conv2d(f1, f2, kernel_size=1, stride=1),
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


class LiteGCM(nn.Module):
    """Factorized Global-local Compression Module using `SeparableConvBlock` (`10.64M` vs `19.70M`)."""

    def __init__(self) -> None:
        super().__init__()
        self.fusion = SGAFusion()
        self.local1 = SeparableConvBlock(256, 3, [256, 512, 1024], stride=4)
        self.local2 = SeparableConvBlock(512, 3, [512, 512, 1024], stride=2)
        self.merge = nn.Conv2d(3072, 1024, kernel_size=1)
        self.res = SeparableConvBlock(1024, 3, [512, 1024, 2048], stride=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        g = self.fusion(x1, x2, x3)
        l1 = self.local1(x1)
        l2 = self.local2(x2)
        l_feat = self.merge(torch.cat([l1, l2, x3], dim=1))
        return self.res(g + l_feat)


class LiteGCMAndDecoder(nn.Module):
    """Configurable CDA Student supporting `B0`, `B1_aux_csd`, `B2_sep_adapter`, and `B3_cda_lite`."""

    def __init__(
        self,
        adapter_mode: str = "separable",
        gcm_conv_mode: str = "dense",
        prune_adapters_at_inference: bool = False,
    ) -> None:
        super().__init__()
        self.adapter_mode = adapter_mode
        self.gcm_conv_mode = gcm_conv_mode
        self.prune_adapters_at_inference = prune_adapters_at_inference

        self.gcm = LiteGCM() if gcm_conv_mode == "separable" else GCM()
        self.dec = GSUDecoder()

        def _make_adapter(in_c: int, out_c: int) -> nn.Module:
            if adapter_mode == "dense":
                return nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)
            if adapter_mode == "separable":
                return DepthwiseSeparableAdapter(in_c, out_c)
            raise ValueError(f"Unsupported adapter_mode: {adapter_mode}")

        self.c11 = _make_adapter(512, 256)
        self.c12 = _make_adapter(1024, 256)
        self.c21 = _make_adapter(256, 512)
        self.c22 = _make_adapter(1024, 512)
        self.c31 = _make_adapter(256, 1024)
        self.c32 = _make_adapter(512, 1024)

    def active_inference_parameters(self) -> int:
        """Returns number of student parameters active during inference."""
        total = sum(p.numel() for p in self.parameters())
        if self.prune_adapters_at_inference:
            adapter_params = sum(
                p.numel()
                for name in ("c11", "c12", "c21", "c22", "c31", "c32")
                for p in getattr(self, name).parameters()
            )
            return total - adapter_params
        return total

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        h1, w1 = e1.shape[2], e1.shape[3]
        h2, w2 = e2.shape[2], e2.shape[3]
        h3, w3 = e3.shape[2], e3.shape[3]

        f = self.gcm(e1, e2, e3)
        s1, s2, s3, l1m, l1n, l2m, l2n, l3m, l3n = self.dec(f)

        # In B1_aux_csd, cross-scale adapters are skipped at inference (.eval() mode)
        if not self.training and self.prune_adapters_at_inference:
            return s1, s2, s3, None, None, None, None, None, None

        l1n = self.c12(F.interpolate(l1n, size=(h1, w1), mode="bilinear", align_corners=True))
        l1m = self.c11(F.interpolate(l1m, size=(h1, w1), mode="bilinear", align_corners=True))
        l2m = self.c21(F.interpolate(l2m, size=(h2, w2), mode="bilinear", align_corners=True))
        l2n = self.c22(F.interpolate(l2n, size=(h2, w2), mode="bilinear", align_corners=True))
        l3m = self.c31(F.interpolate(l3m, size=(h3, w3), mode="bilinear", align_corners=True))
        l3n = self.c32(F.interpolate(l3n, size=(h3, w3), mode="bilinear", align_corners=True))

        return s1, s2, s3, l1m, l1n, l2m, l2n, l3m, l3n


def build_student_model(variant: str = "B0") -> nn.Module:
    """Factory function constructing `B0`, `B1_aux_csd`, `B2_sep_adapter`, or `B3_cda_lite`."""
    if variant == "B0":
        return GCMAndDecoder()
    if variant == "B1_aux_csd":
        return LiteGCMAndDecoder(
            adapter_mode="dense",
            gcm_conv_mode="dense",
            prune_adapters_at_inference=True,
        )
    if variant == "B2_sep_adapter":
        return LiteGCMAndDecoder(
            adapter_mode="separable",
            gcm_conv_mode="dense",
            prune_adapters_at_inference=False,
        )
    if variant == "B3_cda_lite":
        return LiteGCMAndDecoder(
            adapter_mode="separable",
            gcm_conv_mode="separable",
            prune_adapters_at_inference=False,
        )
    raise ValueError(f"Unknown student model variant '{variant}'. Expected B0, B1_aux_csd, B2_sep_adapter, or B3_cda_lite.")
