"""Comprehensive benchmark metrics: Image/Pixel AUROC, Image/Pixel AP, and AUPRO.

Fixes the hard-coded `256*256` reshape bug in official `metric.py` line 47 and
adds Average Precision (AP) and Per-Region Overlap (`AUPRO@0.30` and `AUPRO@0.05`).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.ndimage import label as connected_components
from sklearn import metrics


def compute_aupro(
    anomaly_maps: np.ndarray,
    gt_masks: np.ndarray,
    max_fpr: float = 0.30,
    num_thresholds: int = 100,
) -> float:
    """Computes normalized Area Under the Per-Region Overlap (AUPRO) curve up to `max_fpr`.

    Args:
        anomaly_maps: Array of shape `(N, H, W)` with continuous anomaly scores.
        gt_masks: Binary array of shape `(N, H, W)` with values in `{0, 1}`.
        max_fpr: Upper False Positive Rate integration limit (default `0.30` for MVTec AD,
            `0.05` for strict MVTec AD 2 evaluation).
        num_thresholds: Number of threshold steps used for numerical integration.
    """
    maps = np.asarray(anomaly_maps, dtype=np.float64)
    gts = (np.asarray(gt_masks) > 0.5).astype(np.int32)

    if gts.sum() == 0 or (gts == 0).sum() == 0:
        return 0.0

    # Identify connected defect components across all anomalous images
    region_Masks = []
    for i in range(gts.shape[0]):
        if gts[i].sum() == 0:
            continue
        labeled, num_comp = connected_components(gts[i])
        for comp_id in range(1, num_comp + 1):
            comp_mask = labeled == comp_id
            area = int(comp_mask.sum())
            if area > 0:
                region_Masks.append((i, comp_mask, area))

    if not region_Masks:
        return 0.0

    bg_mask = gts == 0
    total_bg = float(bg_mask.sum())

    min_score, max_score = float(maps.min()), float(maps.max())
    if max_score <= min_score:
        return 0.0

    thresholds = np.linspace(max_score, min_score, num=num_thresholds)
    fprs = []
    pros = []

    for th in thresholds:
        pred = maps >= th
        fp = float(np.logical_and(pred, bg_mask).sum())
        fpr = fp / total_bg
        if fpr > max_fpr:
            # Interpolate final point at max_fpr and stop
            if fprs:
                prev_fpr, prev_pro = fprs[-1], pros[-1]
                # Compute current PRO for linear interpolation
                cur_pro = float(
                    np.mean([pred[idx][cmask].sum() / area for idx, cmask, area in region_Masks])
                )
                if fpr > prev_fpr:
                    weight = (max_fpr - prev_fpr) / (fpr - prev_fpr)
                    interp_pro = prev_pro + weight * (cur_pro - prev_pro)
                    fprs.append(max_fpr)
                    pros.append(interp_pro)
            break

        cur_pro = float(
            np.mean([pred[idx][cmask].sum() / area for idx, cmask, area in region_Masks])
        )
        fprs.append(fpr)
        pros.append(cur_pro)

    if len(fprs) < 2:
        return 0.0

    fpr_arr = np.asarray(fprs, dtype=np.float64)
    pro_arr = np.asarray(pros, dtype=np.float64)
    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    pro_arr = pro_arr[order]

    area_under = float(np.trapezoid(pro_arr, fpr_arr))
    denom = float(fpr_arr[-1]) if fpr_arr[-1] > 0 else max_fpr
    return float(np.clip(area_under / denom, 0.0, 1.0))


def compute_auroc_and_ap(
    anomaly_maps: Sequence[np.ndarray] | np.ndarray,
    image_labels: Sequence[int] | np.ndarray,
    gt_masks: Sequence[np.ndarray] | np.ndarray,
    image_scores: Optional[Sequence[float] | np.ndarray] = None,
) -> Dict[str, float]:
    """Computes Image-AUROC, Image-AP, Pixel-AUROC, and Pixel-AP without hard-coded resolution."""
    maps_arr = np.asarray(anomaly_maps, dtype=np.float64)
    labels_arr = np.asarray(image_labels, dtype=np.int32).reshape(-1)
    gts_arr = (np.asarray(gt_masks) > 0.5).astype(np.int32)

    if maps_arr.ndim != 3:
        raise ValueError(f"anomaly_maps must have shape (N, H, W), got {maps_arr.shape}")
    if gts_arr.ndim == 4 and gts_arr.shape[1] == 1:
        gts_arr = gts_arr.squeeze(1)
    if gts_arr.shape != maps_arr.shape:
        raise ValueError(f"gt_masks shape {gts_arr.shape} does not match maps shape {maps_arr.shape}")

    if image_scores is None:
        img_scores_arr = maps_arr.reshape(maps_arr.shape[0], -1).max(axis=1)
    else:
        img_scores_arr = np.asarray(image_scores, dtype=np.float64).reshape(-1)

    flat_scores = maps_arr.ravel()
    flat_gts = gts_arr.ravel()

    auroc_img = float(metrics.roc_auc_score(labels_arr, img_scores_arr))
    ap_img = float(metrics.average_precision_score(labels_arr, img_scores_arr))
    auroc_pix = float(metrics.roc_auc_score(flat_gts, flat_scores))
    ap_pix = float(metrics.average_precision_score(flat_gts, flat_scores))

    return {
        "auroc_img": round(auroc_img, 4),
        "ap_img": round(ap_img, 4),
        "auroc_pix": round(auroc_pix, 4),
        "ap_pix": round(ap_pix, 4),
    }


def evaluate_predictions(
    anomaly_maps: Sequence[np.ndarray] | np.ndarray,
    image_labels: Sequence[int] | np.ndarray,
    gt_masks: Sequence[np.ndarray] | np.ndarray,
    image_scores: Optional[Sequence[float] | np.ndarray] = None,
    include_aupro: bool = True,
) -> Dict[str, float]:
    """Computes full suite of detection and localization metrics."""
    maps_arr = np.asarray(anomaly_maps, dtype=np.float64)
    gts_arr = (np.asarray(gt_masks) > 0.5).astype(np.int32)
    if gts_arr.ndim == 4 and gts_arr.shape[1] == 1:
        gts_arr = gts_arr.squeeze(1)

    result = compute_auroc_and_ap(
        anomaly_maps=maps_arr,
        image_labels=image_labels,
        gt_masks=gts_arr,
        image_scores=image_scores,
    )
    if include_aupro:
        result["aupro_30"] = round(compute_aupro(maps_arr, gts_arr, max_fpr=0.30), 4)
        result["aupro_05"] = round(compute_aupro(maps_arr, gts_arr, max_fpr=0.05), 4)
    return result
