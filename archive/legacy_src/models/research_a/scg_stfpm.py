"""Method 1 (Researcher A — Trần Quang Thành):
Scale-Calibrated & Inter-Scale Concordance-Gated Feature Pyramid Distillation (SCG-STFPM).

Scientific Basis (2025-2026):
- MVTec AD 2 (Heckler-Kram et al., IJCV 2025): Highlights severe degradation of strict
  AUPRO@0.05 and Pixel-AP caused by uncalibrated multi-scale background texture noise
  and micro-defect dilution.
- EfficientAD (Batzner et al., WACV 2024) & SK-RD4AD (Park et al., CVPRW 2025):
  Multi-scale feature distribution normalization and cross-scale structural consistency.

Solves 3 Core Limitations of Standard STFPM (Wang et al., BMVC 2021):
1. `A1_calibrated` (Scale-wise Normal Statistical Calibration — Zero Test Leakage):
   Estimates per-scale expectation mu_l and standard deviation sigma_l strictly on
   defect-free `train/good` samples, transforming raw cosine/L2 discrepancy maps into
   rectified Z-score maps:
       tilde{U}_l(h,w) = max(0, (U_l(h,w) - mu_l) / (sigma_l + eps))
2. `A2_concordance_gate` (Inter-Scale Semantic Concordance Gating):
   Replaces naive element-wise multiplication (which suffers from Zero-Collapse on
   micro-defects when deep 16x16 layer predicts ~0) with a residual concordance gate:
       G_l(h,w) = tanh(tilde{U}_l(h,w) * mean_{j != l} tilde{U}_j(h,w))
       hat{U}_l(h,w) = tilde{U}_l(h,w) * (1 + eta * G_l(h,w))
3. `A3_full_scg` (Scale-Adaptive Gaussian Filtering & Top-q% Tail Scoring):
   Applies receptive-field-matched Gaussian kernels (sigma_1 < sigma_2 < sigma_3,
   e.g. 1.5, 2.5, 4.0) to preserve sharp micro-defect boundaries at shallow layers
   while smoothing coarse deep layers, and aggregates the top-q% tail pixels for
   robust image-level anomaly scoring instead of fragile single-pixel max().
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baseline.stfpm import compute_stfpm_anomaly_map, gaussian_blur_2d


@dataclass
class SCGNormalStats:
    """Stores per-scale mean and std computed strictly on defect-free training images."""

    mu: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    sigma: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes calibration statistics to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SCGNormalStats":
        """Restores calibration statistics from a dictionary."""
        return cls(
            mu=[float(x) for x in data.get("mu", [0.0, 0.0, 0.0])],
            sigma=[float(x) for x in data.get("sigma", [1.0, 1.0, 1.0])],
            num_samples=int(data.get("num_samples", 0)),
        )


def compute_scale_discrepancy_maps(
    t_feats: Sequence[torch.Tensor],
    s_feats: Sequence[torch.Tensor],
    target_size: Tuple[int, int] = (256, 256),
) -> List[torch.Tensor]:
    """Computes per-scale L2-normalized squared discrepancy maps upsampled to target_size.

    For each pyramid level l in {1, 2, 3}:
        M_l(h, w) = 0.5 * || hat{F}_T^l(h, w) - hat{F}_S^l(h, w) ||_2^2
        U_l = BilinearUpsample(M_l, size=target_size)

    Args:
        t_feats: Teacher feature pyramid [layer1, layer2, layer3].
        s_feats: Student feature pyramid [layer1, layer2, layer3].
        target_size: Output spatial resolution (H, W).

    Returns:
        List of 3 tensors [U_1, U_2, U_3], each of shape (B, H, W).
    """
    upsampled_maps: List[torch.Tensor] = []
    for t_feat, s_feat in zip(t_feats, s_feats):
        t_norm = F.normalize(t_feat, p=2, dim=1)
        s_norm = F.normalize(s_feat, p=2, dim=1)
        layer_map = 0.5 * torch.sum((t_norm - s_norm) ** 2, dim=1, keepdim=True)  # (B, 1, H_l, W_l)
        layer_up = F.interpolate(
            layer_map, size=target_size, mode="bilinear", align_corners=False
        ).squeeze(1)  # (B, H, W)
        upsampled_maps.append(layer_up)
    return upsampled_maps


class SCGSTFPMMapper:
    """Scale-Calibrated & Inter-Scale Concordance-Gated Anomaly Mapper for STFPM / Slim-STFPM.

    Supported ablation modes:
        - `A0_baseline`: Standard STFPM multiplicative fusion + uniform Gaussian blur + max()
        - `A1_calibrated`: Scale-wise normal Z-score calibration + weighted sum + uniform blur + max()
        - `A2_concordance_gate`: `A1` + Inter-Scale Semantic Concordance Gate + uniform blur + max()
        - `A3_full_scg`: `A2` + Scale-Adaptive Gaussian Filtering (sigma_1 < sigma_2 < sigma_3)
                         + Top-q% tail quantile image scoring.
    """

    VALID_MODES = ("A0_baseline", "A1_calibrated", "A2_concordance_gate", "A3_full_scg")

    def __init__(
        self,
        mode: str = "A3_full_scg",
        target_size: Tuple[int, int] = (256, 256),
        concordance_eta: float = 0.5,
        scale_weights: Sequence[float] = (0.4, 0.3, 0.3),
        scale_sigmas: Sequence[float] = (1.5, 2.5, 4.0),
        baseline_sigma: float = 4.0,
        top_q_percent: float = 0.005,
        eps: float = 1e-6,
        stats: Optional[SCGNormalStats] = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported SCG mode '{mode}'. Choose from {self.VALID_MODES}.")
        self.mode = mode
        self.target_size = (int(target_size[0]), int(target_size[1]))
        self.concordance_eta = float(concordance_eta)
        self.scale_weights = [float(w) for w in scale_weights]
        self.scale_sigmas = [float(s) for s in scale_sigmas]
        self.baseline_sigma = float(baseline_sigma)
        self.top_q_percent = float(top_q_percent)
        self.eps = float(eps)
        self.stats = stats or SCGNormalStats()

    @torch.no_grad()
    def fit_calibration(
        self,
        model: nn.Module,
        normal_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> SCGNormalStats:
        """Estimates scale-wise mean (mu_l) and std (sigma_l) strictly on normal training data.

        Uses two-pass streaming accumulation over spatial pixels so memory consumption remains O(1)
        regardless of dataset size on Kaggle GPU.
        """
        model.eval()
        num_scales = 3
        sum_vals = [0.0] * num_scales
        total_pixels = 0
        total_samples = 0

        # Pass 1: Compute exact per-scale mean mu_l
        for b_idx, batch in enumerate(normal_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            t_feats, s_feats = model(images)
            u_maps = compute_scale_discrepancy_maps(t_feats, s_feats, target_size=self.target_size)

            bsz, h, w = u_maps[0].shape
            total_samples += bsz
            total_pixels += bsz * h * w
            for l_idx in range(num_scales):
                sum_vals[l_idx] += float(u_maps[l_idx].sum().item())

        if total_pixels == 0:
            self.stats = SCGNormalStats()
            return self.stats

        mu = [s / total_pixels for s in sum_vals]

        # Pass 2: Compute exact per-scale standard deviation sigma_l
        sq_diff_sums = [0.0] * num_scales
        for b_idx, batch in enumerate(normal_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            t_feats, s_feats = model(images)
            u_maps = compute_scale_discrepancy_maps(t_feats, s_feats, target_size=self.target_size)

            for l_idx in range(num_scales):
                diff = u_maps[l_idx] - mu[l_idx]
                sq_diff_sums[l_idx] += float(torch.sum(diff * diff).item())

        sigma = [max((sq / total_pixels) ** 0.5, self.eps) for sq in sq_diff_sums]
        self.stats = SCGNormalStats(mu=mu, sigma=sigma, num_samples=total_samples)
        return self.stats

    def compute_anomaly_map(
        self,
        t_feats: Sequence[torch.Tensor],
        s_feats: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes anomaly heatmap (B, H, W) and image-level score (B,) on GPU/CPU in pure PyTorch."""
        if self.mode == "A0_baseline":
            return compute_stfpm_anomaly_map(
                list(t_feats),
                list(s_feats),
                target_size=self.target_size,
                gaussian_sigma=self.baseline_sigma,
            )

        # Step 1: Upsampled per-scale discrepancy maps U_l of shape (B, H, W)
        u_maps = compute_scale_discrepancy_maps(t_feats, s_feats, target_size=self.target_size)
        num_scales = len(u_maps)

        # Step 2 (A1): Scale-wise Normal Statistical Calibration (rectified Z-score)
        u_cal: List[torch.Tensor] = []
        for l_idx in range(num_scales):
            mu_l = self.stats.mu[l_idx]
            sigma_l = self.stats.sigma[l_idx]
            z_map = (u_maps[l_idx] - mu_l) / (sigma_l + self.eps)
            u_cal.append(F.relu(z_map))

        if self.mode == "A1_calibrated":
            fused_map = torch.zeros_like(u_cal[0])
            for l_idx in range(num_scales):
                fused_map = fused_map + self.scale_weights[l_idx] * u_cal[l_idx]
            if self.baseline_sigma > 0.0:
                fused_map = gaussian_blur_2d(fused_map, sigma=self.baseline_sigma)
            img_scores = torch.amax(fused_map, dim=(-2, -1))
            return fused_map, img_scores

        # Step 3 (A2): Inter-Scale Semantic Concordance Gating
        # Prevents Zero-Collapse on micro-defects while amplifying multi-scale consensus
        u_gated: List[torch.Tensor] = []
        for l_idx in range(num_scales):
            other_scales = [u_cal[j] for j in range(num_scales) if j != l_idx]
            cross_context = torch.stack(other_scales, dim=0).mean(dim=0)
            concordance_gate = torch.tanh(u_cal[l_idx] * cross_context)
            gated_l = u_cal[l_idx] * (1.0 + self.concordance_eta * concordance_gate)
            u_gated.append(gated_l)

        if self.mode == "A2_concordance_gate":
            fused_map = torch.zeros_like(u_gated[0])
            for l_idx in range(num_scales):
                fused_map = fused_map + self.scale_weights[l_idx] * u_gated[l_idx]
            if self.baseline_sigma > 0.0:
                fused_map = gaussian_blur_2d(fused_map, sigma=self.baseline_sigma)
            img_scores = torch.amax(fused_map, dim=(-2, -1))
            return fused_map, img_scores

        # Step 4 (A3): Scale-Adaptive Gaussian Filtering & Top-q% Tail Scoring
        fused_map = torch.zeros_like(u_gated[0])
        for l_idx in range(num_scales):
            sigma_l = self.scale_sigmas[l_idx]
            smoothed_l = gaussian_blur_2d(u_gated[l_idx], sigma=sigma_l) if sigma_l > 0.0 else u_gated[l_idx]
            fused_map = fused_map + self.scale_weights[l_idx] * smoothed_l

        # Top-q% tail quantile aggregation for image-level score
        bsz = fused_map.shape[0]
        flat_map = fused_map.reshape(bsz, -1)
        if self.top_q_percent <= 0.0:
            img_scores = torch.amax(flat_map, dim=1)
        else:
            k_top = max(1, int(round(flat_map.shape[1] * self.top_q_percent)))
            topk_vals, _ = torch.topk(flat_map, k=k_top, dim=1)
            img_scores = torch.mean(topk_vals, dim=1)

        return fused_map, img_scores
