"""Loss functions for Dual-Student Knowledge Distillation (DSKD).

Reference Paper:
Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection
arXiv:2402.00448v1 - Equations (3), (4), (5), (6), (7).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.anomaly_detection.models.dskd import l2_normalize_features


def compute_layer_anomaly_map(
    f_t: torch.Tensor,
    f_s: torch.Tensor,
    loss_lambda: float = 0.1,
    norm_variant: str = "N1",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Computes the spatial anomaly map M_{t-s}^k at layer k (Eq. 4-6).
    
    Args:
        f_t: Teacher feature tensor [B, C, H, W].
        f_s: Student feature tensor [B, C, H, W].
        loss_lambda: Weight lambda balancing L2 and cosine terms (default 0.1).
        norm_variant: "N1" (Unit-L2, canonical), "N2" (Raw-L2 + Cosine), "N3" (Literal Inv-Sq).
        eps: Small constant for numerical stability.
        
    Returns:
        Anomaly map M of shape [B, 1, H, W].
    """
    if norm_variant == "N1":
        # Eq. (3): Unit-L2 normalized features
        norm_t = l2_normalize_features(f_t, eps=eps)
        norm_s = l2_normalize_features(f_s, eps=eps)

        # Eq. (4): L2 loss = 0.5 * ||norm_t - norm_s||_2^2
        # Sum along channel dimension to get per-pixel scalar
        loss_l2 = 0.5 * torch.sum((norm_t - norm_s) ** 2, dim=1, keepdim=True)

        # Eq. (5): Cosine similarity on normalized vectors
        cos_sim = torch.sum(norm_t * norm_s, dim=1, keepdim=True)
        # Cosine distance = 1 - cos_sim
        loss_cos = 1.0 - cos_sim

    elif norm_variant == "N2":
        # Raw L2 distance on unnormalized features
        loss_l2 = 0.5 * torch.sum((f_t - f_s) ** 2, dim=1, keepdim=True)
        # Cosine distance computed on raw features with safe normalization
        norm_t_len = torch.norm(f_t, p=2, dim=1, keepdim=True)
        norm_s_len = torch.norm(f_s, p=2, dim=1, keepdim=True)
        cos_sim = torch.sum(f_t * f_s, dim=1, keepdim=True) / torch.clamp(norm_t_len * norm_s_len, min=eps)
        loss_cos = 1.0 - cos_sim

    elif norm_variant == "N3":
        # Literal Eq. (3): division by squared norm
        sq_norm_t = torch.sum(f_t ** 2, dim=1, keepdim=True) + eps
        sq_norm_s = torch.sum(f_s ** 2, dim=1, keepdim=True) + eps
        f_t_n3 = f_t / sq_norm_t
        f_s_n3 = f_s / sq_norm_s

        loss_l2 = 0.5 * torch.sum((f_t_n3 - f_s_n3) ** 2, dim=1, keepdim=True)
        len_t = torch.norm(f_t_n3, p=2, dim=1, keepdim=True)
        len_s = torch.norm(f_s_n3, p=2, dim=1, keepdim=True)
        cos_sim = torch.sum(f_t_n3 * f_s_n3, dim=1, keepdim=True) / torch.clamp(len_t * len_s, min=eps)
        loss_cos = 1.0 - cos_sim

    else:
        raise ValueError(f"Unknown norm_variant: {norm_variant}. Expected 'N1', 'N2', or 'N3'.")

    # Eq. (6): M = lambda * L2 + L_cos
    m_k = loss_lambda * loss_l2 + loss_cos
    return m_k


class DSKDLoss(nn.Module):
    """Computes Dual-Student Knowledge Distillation Losses: l_e and l_d (Eq. 6-7)."""

    def __init__(
        self,
        loss_lambda: float = 0.1,
        norm_variant: str = "N1",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.loss_lambda = loss_lambda
        self.norm_variant = norm_variant
        self.eps = eps

    def forward(
        self,
        features_t: List[torch.Tensor],
        features_e: List[torch.Tensor],
        features_d: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Calculates l_e (for S_e) and l_d (for S_d).
        
        Args:
            features_t: Teacher multi-scale features [F_t^1, F_t^2, F_t^3].
            features_e: Student Encoder features [F_e^1, F_e^2, F_e^3].
            features_d: Student Decoder features [F_d^1, F_d^2, F_d^3].
            
        Returns:
            Tuple: (loss_e, loss_d, maps_t_e, maps_t_d)
        """
        k_stages = len(features_t)
        maps_t_e: List[torch.Tensor] = []
        maps_t_d: List[torch.Tensor] = []

        loss_e_total = torch.tensor(0.0, device=features_t[0].device)
        loss_d_total = torch.tensor(0.0, device=features_t[0].device)

        for k in range(k_stages):
            m_te_k = compute_layer_anomaly_map(
                features_t[k], features_e[k],
                loss_lambda=self.loss_lambda,
                norm_variant=self.norm_variant,
                eps=self.eps,
            )
            m_td_k = compute_layer_anomaly_map(
                features_t[k], features_d[k],
                loss_lambda=self.loss_lambda,
                norm_variant=self.norm_variant,
                eps=self.eps,
            )

            maps_t_e.append(m_te_k)
            maps_t_d.append(m_td_k)

            # Eq. (7): mean over spatial dimensions and batch
            loss_e_total = loss_e_total + torch.mean(m_te_k)
            loss_d_total = loss_d_total + torch.mean(m_td_k)

        # Average across the K intermediate layers
        loss_e = loss_e_total / k_stages
        loss_d = loss_d_total / k_stages

        return loss_e, loss_d, maps_t_e, maps_t_d
