"""Mathematical prototype for Direction A (Thành): Calibrated Reliability Fusion (`A1`-`A3`).

Operates on frozen `B0` checkpoints without retraining:
- `A0_baseline`: Official CDA static fusion + sigma=4 + max()
- `A1_calibrated`: Normal-distribution Z-score calibration on train/good
- `A2_reliability_gate`: Multiplicative concordance gate between same-scale and cross-scale maps
- `A3_adaptive_topq`: Scale-adaptive Gaussian smoothing + Top-q% tail image scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def _cos_dist(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(f1, f2, dim=1)


@dataclass
class NormalScaleStats:
    """Stores per-scale mean and std computed strictly on defect-free training images."""

    mu_a: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    std_a: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    mu_ac: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    std_ac: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


class CalibratedReliabilityMapper:
    """Configurable anomaly map fusion supporting `A0_baseline`, `A1`, `A2`, and `A3`."""

    def __init__(
        self,
        mode: str = "A0_baseline",
        img_size: int = 256,
        reliability_eta: float = 0.5,
        scale_sigmas: Sequence[float] = (2.0, 3.0, 4.0),
        gaussian_sigma: float = 4.0,
        top_q_percent: float = 0.005,
        stats: Optional[NormalScaleStats] = None,
    ) -> None:
        self.mode = mode
        self.img_size = int(img_size)
        self.reliability_eta = float(reliability_eta)
        self.scale_sigmas = list(scale_sigmas)
        self.gaussian_sigma = float(gaussian_sigma)
        self.top_q_percent = float(top_q_percent)
        self.stats = stats or NormalScaleStats()

    @torch.no_grad()
    def fit_normal_stats(
        self,
        encoder: torch.nn.Module,
        student: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> NormalScaleStats:
        """Computes scale-wise mean and std over defect-free `train/good` batches."""
        encoder.eval()
        student.eval()

        a_vals: List[List[torch.Tensor]] = [[], [], []]
        ac_vals: List[List[torch.Tensor]] = [[], [], []]

        for b_idx, batch in enumerate(train_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            img = batch["image"].to(device)
            e1, e2, e3 = encoder(img)
            d1, d2, d3, l1m, l1n, l2m, l2n, l3m, l3n = student(e1, e2, e3)

            a_maps = [_cos_dist(e1, d1), _cos_dist(e2, d2), _cos_dist(e3, d3)]
            ac_maps = [
                _cos_dist(e1, l1n) + _cos_dist(e1, l1m),
                _cos_dist(e2, l2n) + _cos_dist(e2, l2m),
                _cos_dist(e3, l3n) + _cos_dist(e3, l3m),
            ]
            for k in range(3):
                a_vals[k].append(a_maps[k].detach().cpu().reshape(-1))
                ac_vals[k].append(ac_maps[k].detach().cpu().reshape(-1))

        mu_a, std_a, mu_ac, std_ac = [], [], [], []
        for k in range(3):
            cat_a = torch.cat(a_vals[k], dim=0)
            cat_ac = torch.cat(ac_vals[k], dim=0)
            mu_a.append(float(cat_a.mean().item()))
            std_a.append(max(float(cat_a.std(unbiased=False).item()), 1e-6))
            mu_ac.append(float(cat_ac.mean().item()))
            std_ac.append(max(float(cat_ac.std(unbiased=False).item()), 1e-6))

        self.stats = NormalScaleStats(mu_a=mu_a, std_a=std_a, mu_ac=mu_ac, std_ac=std_ac)
        return self.stats

    @torch.no_grad()
    def compute_maps_and_scores(
        self,
        e_list: Sequence[torch.Tensor],
        d_list: Sequence[Optional[torch.Tensor]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns `(pixel_maps, image_scores)` of shapes `(B, H, W)` and `(B,)`."""
        e1, e2, e3 = e_list
        d1, d2, d3, l1m, l1n, l2m, l2n, l3m, l3n = d_list

        a_raw = [_cos_dist(e1, d1), _cos_dist(e2, d2), _cos_dist(e3, d3)]
        a_up = [
            F.interpolate(a.unsqueeze(1), size=self.img_size, mode="bilinear", align_corners=True).squeeze(1)
            for a in a_raw
        ]

        has_cross = all(t is not None for t in (l1m, l1n, l2m, l2n, l3m, l3n))
        if has_cross:
            ac_raw = [
                _cos_dist(e1, l1n) + _cos_dist(e1, l1m),
                _cos_dist(e2, l2n) + _cos_dist(e2, l2m),
                _cos_dist(e3, l3n) + _cos_dist(e3, l3m),
            ]
            ac_up = [
                F.interpolate(ac.unsqueeze(1), size=self.img_size, mode="bilinear", align_corners=True).squeeze(1)
                for ac in ac_raw
            ]
        else:
            ac_up = [torch.zeros_like(a) for a in a_up]

        if self.mode == "A0_baseline":
            raw_map = 2.0 * (a_up[0] + a_up[1] + a_up[2])
            if has_cross:
                raw_map = raw_map + 0.1 * (0.4 * ac_up[0] + 0.2 * ac_up[1] + 0.4 * ac_up[2])
            np_maps = raw_map.detach().cpu().numpy()
            for i in range(np_maps.shape[0]):
                np_maps[i] = gaussian_filter(np_maps[i], sigma=self.gaussian_sigma)
            img_scores = np_maps.reshape(np_maps.shape[0], -1).max(axis=1)
            return np_maps, img_scores

        # Standardized maps for A1, A2, A3
        a_norm = [
            torch.clamp((a_up[k] - self.stats.mu_a[k]) / (self.stats.std_a[k] + 1e-6), min=0.0)
            for k in range(3)
        ]
        ac_norm = [
            torch.clamp((ac_up[k] - self.stats.mu_ac[k]) / (self.stats.std_ac[k] + 1e-6), min=0.0)
            for k in range(3)
        ]

        if self.mode == "A1_calibrated":
            weights = [0.4, 0.2, 0.4]
            fused = sum(a_norm[k] + weights[k] * ac_norm[k] for k in range(3))
            np_maps = fused.detach().cpu().numpy()
            for i in range(np_maps.shape[0]):
                np_maps[i] = gaussian_filter(np_maps[i], sigma=self.gaussian_sigma)
            img_scores = np_maps.reshape(np_maps.shape[0], -1).max(axis=1)
            return np_maps, img_scores

        # Concordance reliability gate for A2 and A3
        gated_scales = []
        for k in range(3):
            rel_gate = torch.tanh(a_norm[k] * ac_norm[k])
            gated_scales.append(a_norm[k] * (1.0 + self.reliability_eta * rel_gate))

        if self.mode == "A2_reliability_gate":
            fused = sum(gated_scales)
            np_maps = fused.detach().cpu().numpy()
            for i in range(np_maps.shape[0]):
                np_maps[i] = gaussian_filter(np_maps[i], sigma=self.gaussian_sigma)
            img_scores = np_maps.reshape(np_maps.shape[0], -1).max(axis=1)
            return np_maps, img_scores

        if self.mode == "A3_adaptive_topq":
            bsz = e1.shape[0]
            np_maps = np.zeros((bsz, self.img_size, self.img_size), dtype=np.float32)
            for k in range(3):
                scale_np = gated_scales[k].detach().cpu().numpy()
                sigma_k = self.scale_sigmas[k]
                for i in range(bsz):
                    np_maps[i] += gaussian_filter(scale_np[i], sigma=sigma_k)

            flat = np_maps.reshape(bsz, -1)
            k_top = max(1, int(flat.shape[1] * self.top_q_percent))
            part = np.partition(flat, -k_top, axis=1)[:, -k_top:]
            img_scores = part.mean(axis=1)
            return np_maps, img_scores

        raise ValueError(f"Unsupported fusion mode: {self.mode}")
