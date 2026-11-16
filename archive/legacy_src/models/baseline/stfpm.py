"""STFPM: Student-Teacher Feature Pyramid Matching for Anomaly Detection.

Reference:
    Guodong Wang, Shumin Han, Errui Ding, Di Huang.
    "Student-Teacher Feature Pyramid Matching for Anomaly Detection."
    BMVC 2021. arXiv:2103.04257.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def gaussian_blur_2d(
    tensor: torch.Tensor,
    sigma: float = 4.0,
    kernel_size: Optional[int] = None,
) -> torch.Tensor:
    """Applies 2D Gaussian blur with reflection padding natively in PyTorch.

    Args:
        tensor: Tensor of shape (B, H, W) or (B, 1, H, W).
        sigma: Standard deviation of Gaussian kernel (default: 4.0 for community STFPM-Smoothed variant).
        kernel_size: Size of Gaussian kernel (default: derived from sigma as 2 * round(2*sigma) + 1).

    Returns:
        Smoothed tensor with identical shape and device.
    """
    if sigma <= 0.0:
        return tensor

    if kernel_size is None:
        kernel_size = 2 * int(round(2.0 * sigma)) + 1

    coords = torch.arange(kernel_size, dtype=tensor.dtype, device=tensor.device) - (kernel_size // 2)
    gaussian_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = (gaussian_1d[:, None] * gaussian_1d[None, :]).unsqueeze(0).unsqueeze(0)

    pad = kernel_size // 2
    if tensor.ndim == 3:
        t = tensor.unsqueeze(1)
        t_pad = F.pad(t, (pad, pad, pad, pad), mode="reflect")
        out = F.conv2d(t_pad, gaussian_2d).squeeze(1)
        return out
    elif tensor.ndim == 4:
        channels = tensor.shape[1]
        weight = gaussian_2d.repeat(channels, 1, 1, 1)
        t_pad = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(t_pad, weight, groups=channels)
    return tensor


class ResNet18FeatureExtractor(nn.Module):
    """ResNet-18 backbone extracting multi-scale features from layer1, layer2, layer3."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Retain stages: conv1, bn1, relu, maxpool, layer1, layer2, layer3
        self.init_block = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 64 channels, spatial 1/4 (e.g. 64x64 for 256x256)
        self.layer2 = backbone.layer2  # 128 channels, spatial 1/8 (e.g. 32x32 for 256x256)
        self.layer3 = backbone.layer3  # 256 channels, spatial 1/16 (e.g. 16x16 for 256x256)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features [layer1, layer2, layer3]."""
        x = self.init_block(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]


class STFPM(nn.Module):
    """Complete STFPM Model: Frozen Teacher + Trainable Student.

    Teacher: Pretrained ResNet-18 (frozen weights and permanent eval mode).
    Student: Randomly initialized ResNet-18 (trainable).
    """

    def __init__(self, teacher_pretrained: bool = True) -> None:
        super().__init__()
        self.teacher = ResNet18FeatureExtractor(pretrained=teacher_pretrained)
        self.student = ResNet18FeatureExtractor(pretrained=False)

        # Freeze teacher parameters permanently
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "STFPM":
        """Overrides train mode to ensure teacher permanently remains in eval mode.

        In standard PyTorch, calling model.train() recursively switches all submodules
        to train mode. This override defensively keeps self.teacher in eval mode to
        prevent its BatchNorm layers from updating running stats on industrial batches.
        """
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass extracting teacher and student features.

        Args:
            x: Input image tensor (B, 3, H, W)

        Returns:
            Tuple of (teacher_features, student_features), each being a list of 3 tensors.
        """
        with torch.no_grad():
            t_feats = self.teacher(x)
        s_feats = self.student(x)
        return t_feats, s_feats


class STFPMLoss(nn.Module):
    """STFPM multi-scale normalized feature matching loss (Eq. 1 in Wang et al., 2021).

    L = 1/2 * sum_{l=1}^3 1/(H_l * W_l) sum_{h,w} || hat{F}_T^l(h,w) - hat{F}_S^l(h,w) ||_2^2
    where features are L2 normalized along the channel axis.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, t_feats: List[torch.Tensor], s_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=t_feats[0].device)
        for t_feat, s_feat in zip(t_feats, s_feats):
            # Normalize channel dimension
            t_norm = F.normalize(t_feat, p=2, dim=1)
            s_norm = F.normalize(s_feat, p=2, dim=1)
            # Pointwise squared L2 distance: sum across channels, mean across spatial & batch
            diff = torch.sum((t_norm - s_norm) ** 2, dim=1)  # (B, H, W)
            loss = loss + 0.5 * torch.mean(diff)
        return loss


def compute_stfpm_anomaly_map(
    t_feats: List[torch.Tensor],
    s_feats: List[torch.Tensor],
    target_size: Tuple[int, int] = (256, 256),
    gaussian_sigma: Optional[float] = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute STFPM anomaly map and image score according to Wang et al. (BMVC 2021).

    Computes channel-normalized squared difference per pyramid level, upsamples
    each layer map directly to target image size via bilinear interpolation,
    computes element-wise product across the 3 layers (Eq. 2-3 in Section 3.2),
    applies optional Gaussian smoothing if requested (standard community
    post-processing in RD4AD / anomalib, disabled by default for strict paper parity),
    and extracts image-level maximum anomaly score.

    Args:
        t_feats: List of teacher features [layer1, layer2, layer3]
        s_feats: List of student features [layer1, layer2, layer3]
        target_size: Output spatial resolution (H, W), default (256, 256)
        gaussian_sigma: Standard deviation for optional Gaussian smoothing (default: 0.0 / disabled for paper parity; set to 4.0 for STFPM-Smoothed variant)

    Returns:
        Tuple of (anomaly_map (B, H, W), image_score (B,))
    """
    batch_size = t_feats[0].shape[0]
    score_map = torch.ones((batch_size, 1, target_size[0], target_size[1]), device=t_feats[0].device)

    for t_feat, s_feat in zip(t_feats, s_feats):
        t_norm = F.normalize(t_feat, p=2, dim=1)
        s_norm = F.normalize(s_feat, p=2, dim=1)
        # Pointwise squared difference (Eq. 2)
        layer_map = 0.5 * torch.sum((t_norm - s_norm) ** 2, dim=1, keepdim=True)  # (B, 1, H_l, W_l)
        # Upsample each layer map directly to image size W x H (Eq. 3 in paper)
        layer_map_interp = F.interpolate(
            layer_map, size=target_size, mode="bilinear", align_corners=False
        )
        score_map = score_map * layer_map_interp

    fused_map = score_map.squeeze(1)  # (B, H, W)

    # Optional Gaussian smoothing (community post-processing variant; Eq. 2-3 in Wang et al. 2021 has no blur)
    if gaussian_sigma is not None and gaussian_sigma > 0.0:
        fused_map = gaussian_blur_2d(fused_map, sigma=gaussian_sigma)

    # Image score: maximum anomaly value across the spatial map
    img_scores = torch.amax(fused_map, dim=(-2, -1))  # (B,)

    return fused_map, img_scores

