"""Inference pipeline for CDA Baseline and Calibrated Reliability variants.

Guarantees:
1. Gradient tracking is disabled (`torch.no_grad()`).
2. Supports arbitrary `img_size` (fixes official `metric.py` line 47 hard-coded `256*256`).
3. Supports both baseline (`A0_baseline`), calibrated (`A1`-`A3`), and pruned (`B1_aux_csd`) inference.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..models.research_a.calibrated_fusion import CalibratedReliabilityMapper, NormalScaleStats


class CDAAnomalyMapper:
    """Wrapper unifying baseline (`metric.py: anomaly_score`) and Direction A fusion modes."""

    def __init__(
        self,
        img_size: int = 256,
        fusion_mode: str = "A0_baseline",
        gaussian_sigma: float = 4.0,
        reliability_eta: float = 0.5,
        scale_sigmas: Sequence[float] = (2.0, 3.0, 4.0),
        top_q_percent: float = 0.005,
        stats: Optional[NormalScaleStats] = None,
    ) -> None:
        self.img_size = int(img_size)
        self.mapper = CalibratedReliabilityMapper(
            mode=fusion_mode,
            img_size=img_size,
            reliability_eta=reliability_eta,
            scale_sigmas=scale_sigmas,
            gaussian_sigma=gaussian_sigma,
            top_q_percent=top_q_percent,
            stats=stats,
        )

    @torch.no_grad()
    def __call__(
        self,
        e_list: Sequence[torch.Tensor],
        d_list: Sequence[Optional[torch.Tensor]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes `(anomaly_maps, image_scores)` of shapes `(B, H, W)` and `(B,)`."""
        return self.mapper.compute_maps_and_scores(e_list, d_list)


@torch.no_grad()
def predict_batch(
    encoder: nn.Module,
    student: nn.Module,
    images: torch.Tensor,
    mapper: CDAAnomalyMapper,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs forward inference without gradient tracking and returns `(maps, img_scores)`."""
    encoder.eval()
    student.eval()
    e1, e2, e3 = encoder(images)
    d_list = student(e1, e2, e3)
    return mapper([e1, e2, e3], d_list)
