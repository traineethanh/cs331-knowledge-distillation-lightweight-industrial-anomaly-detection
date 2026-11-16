"""Dual-Student Knowledge Distillation (DSKD) Architecture.

Reference Paper:
Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection
arXiv:2402.00448v1 - https://arxiv.org/abs/2402.00448
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def l2_normalize_features(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Channel-wise L2 normalization per spatial location (i, j).
    
    Formula (Eq. 3, Unit-L2 N1 variant):
        \\hat{F}(I_m)_{ij} = F(I_m)_{ij} / (||F(I_m)_{ij}||_2 + eps)
    """
    norm = torch.norm(features, p=2, dim=1, keepdim=True)
    return features / (norm + eps)


class ResNet18FeatureExtractor(nn.Module):
    """Truncated 3-stage ResNet-18 extracting features at conv2_x, conv3_x, conv4_x."""

    def __init__(self, pretrained: bool = False, frozen: bool = False) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1  # 64 channels, 64x64
        self.layer2 = base.layer2  # 128 channels, 32x32
        self.layer3 = base.layer3  # 256 channels, 16x16

        if frozen:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]


class DenseDeepFeatureEmbedding(nn.Module):
    """Deep Feature Embedding (DFE) Module (Figure 4 in arXiv:2402.00448v1).
    
    Fuses multi-scale features [F1, F2, F3] into low-dimensional representation F_emb (256x16x16).
    """

    def __init__(self, has_f3_conv: bool = True) -> None:
        super().__init__()
        self.has_f3_conv = has_f3_conv

        # Branch 1: F1 (64x64x64) -> 256x16x16 (stride 2 twice)
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Branch 2: F2 (128x32x32) -> 256x16x16 (stride 2 once)
        self.branch2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Branch 3: F3 (256x16x16) -> 256x16x16
        if has_f3_conv:
            self.branch3 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch3 = nn.Identity()

        # Bottleneck Residual Block: Concat(768) -> 256
        self.res_conv1 = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_bn1 = nn.BatchNorm2d(256)
        self.res_relu = nn.ReLU(inplace=True)

        self.res_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_bn2 = nn.BatchNorm2d(256)

        self.res_shortcut = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.res_bn_sc = nn.BatchNorm2d(256)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3 = features
        b1 = self.branch1(f1)
        b2 = self.branch2(f2)
        b3 = self.branch3(f3)

        cat = torch.cat([b1, b2, b3], dim=1)  # [B, 768, 16, 16]

        main = self.res_relu(self.res_bn1(self.res_conv1(cat)))
        main = self.res_bn2(self.res_conv2(main))

        shortcut = self.res_bn_sc(self.res_shortcut(cat))
        f_emb = F.relu(main + shortcut, inplace=True)
        return f_emb


class DecoderBasicBlock(nn.Module):
    """Basic residual block for Decoder with optional 1x1 projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + res, inplace=True)
        return out


class ReversedResNet18Decoder(nn.Module):
    """Reversed ResNet-18 Decoder using Bilinear Upsampling + Conv3x3 residual blocks."""

    def __init__(self, upsample_mode: str = "bilinear") -> None:
        super().__init__()
        self.upsample_mode = upsample_mode

        # rev_layer3: 256 -> 256 (16x16)
        self.block3_1 = DecoderBasicBlock(256, 256)
        self.block3_2 = DecoderBasicBlock(256, 256)

        # rev_layer2: upsample 16x16 -> 32x32, 256 -> 128
        self.upsample2 = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)
        self.block2_1 = DecoderBasicBlock(256, 128)
        self.block2_2 = DecoderBasicBlock(128, 128)

        # rev_layer1: upsample 32x32 -> 64x64, 128 -> 64
        self.upsample1 = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)
        self.block1_1 = DecoderBasicBlock(128, 64)
        self.block1_2 = DecoderBasicBlock(64, 64)

    def forward(self, f_emb: torch.Tensor) -> List[torch.Tensor]:
        # rev_layer3 -> F_d^3 (256x16x16)
        d3 = self.block3_1(f_emb)
        d3 = self.block3_2(d3)

        # rev_layer2 -> F_d^2 (128x32x32)
        up2 = self.upsample2(d3)
        d2 = self.block2_1(up2)
        d2 = self.block2_2(d2)

        # rev_layer1 -> F_d^1 (64x64x64)
        up1 = self.upsample1(d2)
        d1 = self.block1_1(up1)
        d1 = self.block1_2(d1)

        return [d1, d2, d3]


class DSKDModel(nn.Module):
    """Complete Dual-Student Knowledge Distillation (DSKD) Network.
    
    Contains:
    - Teacher T (pretrained ImageNet, frozen eval)
    - Student Encoder S_e (trainable)
    - Deep Feature Embedding DFE (trainable, part of Decoder system)
    - Student Decoder S_d (trainable)
    """

    def __init__(
        self,
        pretrained_teacher: bool = True,
        has_f3_conv: bool = True,
        dfe_input_type: str = "raw",       # "raw" (Fig. 4) or "norm" (Algo. 2)
        detach_dfe_input: bool = True,     # True: stop gradient from l_d to S_e
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.dfe_input_type = dfe_input_type
        self.detach_dfe_input = detach_dfe_input

        self.teacher = ResNet18FeatureExtractor(pretrained=pretrained_teacher, frozen=True)
        self.student_encoder = ResNet18FeatureExtractor(pretrained=False, frozen=False)
        self.dfe = DenseDeepFeatureEmbedding(has_f3_conv=has_f3_conv)
        self.student_decoder = ReversedResNet18Decoder(upsample_mode=upsample_mode)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass for training and inference.
        
        Returns:
            Tuple of (features_t, features_e, features_d), each a list of 3 tensors:
            [F^1 (64x64x64), F^2 (128x32x32), F^3 (256x16x16)].
        """
        with torch.no_grad():
            f_t = self.teacher(x)

        f_e = self.student_encoder(x)

        # Determine features entering DFE based on configuration
        if self.dfe_input_type == "norm":
            f_e_for_dfe = [l2_normalize_features(f) for f in f_e]
        else:
            f_e_for_dfe = f_e

        # Apply gradient stop if detach_dfe_input is enabled
        if self.detach_dfe_input:
            f_e_for_dfe = [f.detach() for f in f_e_for_dfe]

        f_emb = self.dfe(f_e_for_dfe)
        f_d = self.student_decoder(f_emb)

        return f_t, f_e, f_d

    def get_encoder_params(self) -> List[nn.Parameter]:
        """Returns parameters belonging to Student Encoder (theta_e)."""
        return [p for p in self.student_encoder.parameters() if p.requires_grad]

    def get_decoder_system_params(self) -> List[nn.Parameter]:
        """Returns parameters belonging to DFE + Student Decoder (theta_d)."""
        dfe_params = [p for p in self.dfe.parameters() if p.requires_grad]
        dec_params = [p for p in self.student_decoder.parameters() if p.requires_grad]
        return dfe_params + dec_params
