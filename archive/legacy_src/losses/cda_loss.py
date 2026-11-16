"""Same-scale and cross-scale cosine distance distillation loss (`train.py` & `metric.py`).

Supports both:
- `cross_loss_mode="official_train"`: Exact formula from `train.py` lines 43-45
  `(0.4 * cos_dist(e1, l1n) + cos_dist(e1, l1m)).mean() + ...`
- `cross_loss_mode="symmetric_metric"`: Symmetric scale weighting matching `metric.py` line 32
  `0.4 * (cos_dist(e1, l1n) + cos_dist(e1, l1m)).mean() + ...`
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_distance_map(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    """Computes channel-wise cosine distance `1 - cos_sim(f1, f2, dim=1)` of shape `(B, H, W)`."""
    return 1.0 - F.cosine_similarity(f1, f2, dim=1)


class CDALoss(nn.Module):
    """Cross-scale Distillation Architecture (CDA) training objective."""

    def __init__(
        self,
        lambda_same: float = 2.0,
        lambda_cross: float = 0.1,
        cross_scale_weights: Sequence[float] = (0.4, 0.2, 0.4),
        cross_loss_mode: str = "official_train",
    ) -> None:
        super().__init__()
        self.lambda_same = float(lambda_same)
        self.lambda_cross = float(lambda_cross)
        if len(cross_scale_weights) != 3:
            raise ValueError("cross_scale_weights must have length 3")
        self.w1, self.w2, self.w3 = [float(w) for w in cross_scale_weights]
        if cross_loss_mode not in ("official_train", "symmetric_metric"):
            raise ValueError(f"Unsupported cross_loss_mode: {cross_loss_mode}")
        self.cross_loss_mode = cross_loss_mode

    def forward(
        self,
        teacher_features: Sequence[torch.Tensor],
        student_outputs: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        e1, e2, e3 = teacher_features
        d1, d2, d3, l1m, l1n, l2m, l2n, l3m, l3n = student_outputs

        loss_same = (
            cosine_distance_map(e1, d1).mean()
            + cosine_distance_map(e2, d2).mean()
            + cosine_distance_map(e3, d3).mean()
        )

        if self.cross_loss_mode == "official_train":
            # Exact match to train.py lines 43-45
            loss_cross = (
                (self.w1 * cosine_distance_map(e1, l1n) + cosine_distance_map(e1, l1m)).mean()
                + (self.w2 * cosine_distance_map(e2, l2n) + cosine_distance_map(e2, l2m)).mean()
                + (self.w3 * cosine_distance_map(e3, l3n) + cosine_distance_map(e3, l3m)).mean()
            )
        else:
            # Symmetric scale weighting matching metric.py lines 21-23 & 32
            loss_cross = (
                self.w1 * (cosine_distance_map(e1, l1n) + cosine_distance_map(e1, l1m)).mean()
                + self.w2 * (cosine_distance_map(e2, l2n) + cosine_distance_map(e2, l2m)).mean()
                + self.w3 * (cosine_distance_map(e3, l3n) + cosine_distance_map(e3, l3m)).mean()
            )

        total_loss = self.lambda_same * loss_same + self.lambda_cross * loss_cross
        components = {
            "loss_total": float(total_loss.detach().item()),
            "loss_same": float(loss_same.detach().item()),
            "loss_cross": float(loss_cross.detach().item()),
        }
        return total_loss, components
