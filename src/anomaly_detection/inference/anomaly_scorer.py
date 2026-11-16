"""Inference and Anomaly Map generation for DSKD.

Reference Paper:
Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection
arXiv:2402.00448v1 - Algorithm 2 (DSKD Anomaly Inference), Eq. (6), Eq. (9).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from src.anomaly_detection.losses.dskd_loss import compute_layer_anomaly_map
from src.anomaly_detection.models.dskd import DSKDModel


class DSKDAnomalyScorer:
    """Generates multi-scale fused anomaly maps and image-level anomaly scores.
    
    Implements Algorithm 2:
    1. Extracts multi-scale features F_t, F_e, F_d.
    2. Computes discrepancy maps M_{t-d}^k (Eq. 6).
    3. Bilinearly upsamples all intermediate maps to (img_size, img_size) and sums them.
    4. Applies Gaussian smoothing (sigma = 4.0).
    5. Calculates image anomaly score as max(anomaly_map).
    """

    def __init__(
        self,
        img_size: int = 256,
        loss_lambda: float = 0.1,
        gaussian_sigma: float = 4.0,
        norm_variant: str = "N1",
    ) -> None:
        self.img_size = img_size
        self.loss_lambda = loss_lambda
        self.gaussian_sigma = gaussian_sigma
        self.norm_variant = norm_variant

    @torch.no_grad()
    def predict_batch(
        self,
        model: DSKDModel,
        images: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts anomaly maps and image scores for a batch of images.
        
        Args:
            model: Trained DSKDModel in eval mode.
            images: Tensor of shape [B, 3, H, W].
            
        Returns:
            Tuple of:
            - anomaly_maps: NumPy array [B, H, W] of float32 continuous scores.
            - image_scores: NumPy array [B] of float32 scalar scores.
        """
        model.eval()
        device = images.device

        f_t, f_e, f_d = model(images)
        k_stages = len(f_t)

        batch_size = images.shape[0]
        fused_maps = torch.zeros((batch_size, 1, self.img_size, self.img_size), device=device)

        for k in range(k_stages):
            # Compute anomaly map M_{t-d}^k between Teacher and Decoder
            m_td_k = compute_layer_anomaly_map(
                f_t[k], f_d[k],
                loss_lambda=self.loss_lambda,
                norm_variant=self.norm_variant,
            )

            # Algorithm 2, Line 11: M <- M + upsample(M_{t-d}^k)
            m_td_up = F.interpolate(
                m_td_k,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )
            fused_maps = fused_maps + m_td_up

        # Convert to NumPy for Gaussian filtering
        maps_np = fused_maps.squeeze(1).cpu().numpy().astype(np.float32)

        # Apply 2D Gaussian filter (sigma = 4.0) per image (Algorithm 2, Line 13)
        smoothed_maps = np.zeros_like(maps_np)
        for i in range(batch_size):
            smoothed_maps[i] = gaussian_filter(maps_np[i], sigma=self.gaussian_sigma)

        # Algorithm 2, Line 14: score = max(fM)
        image_scores = np.max(smoothed_maps, axis=(1, 2))

        return smoothed_maps, image_scores
