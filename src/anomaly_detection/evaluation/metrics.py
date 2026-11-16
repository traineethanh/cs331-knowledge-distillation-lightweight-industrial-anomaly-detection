"""Comprehensive benchmark metrics: Image/Pixel AUROC, Image/Pixel AP, and AUPRO.

Calculates industrial anomaly detection metrics:
- Image-level AUROC & Average Precision (AP)
- Pixel-level AUROC & Average Precision (AP)
- Per-Region Overlap (AUPRO@0.30 and AUPRO@0.05) strictly normalized by max_fpr
  according to Bergmann et al. (IJCV 2021) and Wang et al. (BMVC 2021).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.ndimage import label as connected_components
from sklearn import metrics


def _trapezoid_integrate(y: np.ndarray, x: np.ndarray) -> float:
    """Integrates `y` over `x` using the composite trapezoidal rule across NumPy 1.x and 2.x."""
    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:
        trapz_fn = getattr(np, "trapz", None)
    if trapz_fn is None or not callable(trapz_fn):
        raise AttributeError(
            "Neither `numpy.trapezoid` (NumPy >= 2.0) nor `numpy.trapz` (NumPy 1.x) is available "
            "for AUPRO trapezoidal integration."
        )
    return float(trapz_fn(y, x))


def compute_aupro(
    anomaly_maps: np.ndarray,
    gt_masks: np.ndarray,
    max_fpr: float = 0.30,
    num_thresholds: int = 500,
) -> float:
    """Computes normalized Area Under the Per-Region Overlap (AUPRO) curve up to `max_fpr`.

    Follows the standard MVTec AD protocol (Bergmann et al., IJCV 2021):
    1. Extracts connected defect components in ground-truth masks.
    2. Measures Per-Region Overlap (PRO) as the mean intersection ratio across all regions.
    3. Measures False Positive Rate (FPR) strictly over non-defective (background) pixels.
    4. Integrates PRO(FPR) over [0, max_fpr] and normalizes by dividing strictly by max_fpr:
       AUPRO = (1 / max_fpr) * int_0^{max_fpr} PRO(FPR) dFPR.

    Args:
        anomaly_maps: Array of shape (N, H, W) with continuous anomaly scores.
        gt_masks: Binary array of shape (N, H, W) with values in {0, 1}.
        max_fpr: Upper FPR integration limit (0.30 for standard MVTec AD, 0.05 for strict localization).
        num_thresholds: Number of quantile threshold steps used for fine-grained numerical integration.

    Returns:
        Normalized AUPRO value in [0.0, 1.0].
    """
    maps = np.asarray(anomaly_maps, dtype=np.float64)
    gts = (np.asarray(gt_masks) > 0.5).astype(np.int32)

    if maps.ndim != 3 or gts.ndim != 3:
        raise ValueError(f"anomaly_maps and gt_masks must be 3D (N, H, W). Got {maps.shape} and {gts.shape}")

    if gts.sum() == 0 or (gts == 0).sum() == 0:
        return 0.0

    # Extract connected defect components across all anomalous images
    region_masks = []
    for i in range(gts.shape[0]):
        if gts[i].sum() == 0:
            continue
        labeled, num_comp = connected_components(gts[i])
        for comp_id in range(1, num_comp + 1):
            comp_mask = labeled == comp_id
            area = int(comp_mask.sum())
            if area > 0:
                region_masks.append((i, comp_mask, area))

    if not region_masks:
        return 0.0

    bg_mask = gts == 0
    total_bg = float(bg_mask.sum())
    bg_scores = maps[bg_mask]

    min_score, max_score = float(maps.min()), float(maps.max())
    if max_score <= min_score or len(bg_scores) == 0:
        return 0.0

    # Sample thresholds densely from background quantiles matching [0, max_fpr]
    # and foreground (defect) score quantiles so discrete/tied background maps
    # never skip defect threshold transitions.
    target_fprs = np.linspace(0.0, max_fpr, num=num_thresholds)
    bg_quantiles = np.quantile(bg_scores, 1.0 - target_fprs[1:])

    fg_scores = maps[gts == 1]
    fg_quantiles = (
        np.quantile(fg_scores, np.linspace(0.0, 1.0, num=min(100, max(10, num_thresholds // 5))))
        if len(fg_scores) > 0
        else np.array([], dtype=np.float64)
    )

    bg_max = float(bg_scores.max())
    high_th = np.linspace(max_score, bg_max, num=min(50, max(5, num_thresholds // 10)))
    all_th = np.unique(np.concatenate(([max_score + 1e-6], high_th, bg_quantiles, fg_quantiles)))
    thresholds = np.sort(all_th)[::-1]  # Descending order

    fprs = [0.0]
    pros = [0.0]

    for th in thresholds:
        pred = maps >= th
        fp = float(np.logical_and(pred, bg_mask).sum())
        fpr = fp / total_bg

        if fpr > max_fpr:
            # Interpolate final point exactly at max_fpr
            if fprs:
                prev_fpr, prev_pro = fprs[-1], pros[-1]
                cur_pro = float(
                    np.mean([pred[idx][cmask].sum() / area for idx, cmask, area in region_masks])
                )
                if fpr > prev_fpr:
                    weight = (max_fpr - prev_fpr) / (fpr - prev_fpr)
                    interp_pro = prev_pro + weight * (cur_pro - prev_pro)
                    fprs.append(max_fpr)
                    pros.append(interp_pro)
            break

        cur_pro = float(
            np.mean([pred[idx][cmask].sum() / area for idx, cmask, area in region_masks])
        )
        fprs.append(fpr)
        pros.append(cur_pro)

    # Ensure max_fpr boundary is reached
    if fprs[-1] < max_fpr:
        fprs.append(max_fpr)
        pros.append(pros[-1])

    if len(fprs) < 2:
        return 0.0

    fpr_arr = np.asarray(fprs, dtype=np.float64)
    pro_arr = np.asarray(pros, dtype=np.float64)

    order = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[order]
    pro_arr = pro_arr[order]

    area_under = _trapezoid_integrate(pro_arr, fpr_arr)
    # Standard protocol: divide strictly by max_fpr
    aupro = area_under / float(max_fpr)
    return float(np.clip(aupro, 0.0, 1.0))


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
    anomaly_maps: Optional[Sequence[np.ndarray] | np.ndarray] = None,
    image_labels: Optional[Sequence[int] | np.ndarray] = None,
    gt_masks: Optional[Sequence[np.ndarray] | np.ndarray] = None,
    image_scores: Optional[Sequence[float] | np.ndarray] = None,
    include_aupro: bool = True,
    *,
    maps: Optional[Sequence[np.ndarray] | np.ndarray] = None,
    labels: Optional[Sequence[int] | np.ndarray] = None,
    masks: Optional[Sequence[np.ndarray] | np.ndarray] = None,
    scores: Optional[Sequence[float] | np.ndarray] = None,
) -> Dict[str, float]:
    """Computes full suite of anomaly detection and localization metrics.

    Defensively accepts both canonical argument names (anomaly_maps, image_labels, gt_masks, image_scores)
    and shorthand keyword aliases (maps, labels, masks, scores).
    """
    raw_maps = anomaly_maps if anomaly_maps is not None else maps
    raw_labels = image_labels if image_labels is not None else labels
    raw_masks = gt_masks if gt_masks is not None else masks
    raw_scores = image_scores if image_scores is not None else scores

    if raw_maps is None:
        raise ValueError("anomaly_maps (or maps) must be provided.")
    if raw_labels is None:
        raise ValueError("image_labels (or labels) must be provided.")
    if raw_masks is None:
        raise ValueError("gt_masks (or masks) must be provided.")

    maps_arr = np.asarray(raw_maps, dtype=np.float64)
    gts_arr = (np.asarray(raw_masks) > 0.5).astype(np.int32)
    if gts_arr.ndim == 4 and gts_arr.shape[1] == 1:
        gts_arr = gts_arr.squeeze(1)

    result = compute_auroc_and_ap(
        anomaly_maps=maps_arr,
        image_labels=raw_labels,
        gt_masks=gts_arr,
        image_scores=raw_scores,
    )
    if include_aupro:
        result["aupro_30"] = round(compute_aupro(maps_arr, gts_arr, max_fpr=0.30), 4)
        result["aupro_05"] = round(compute_aupro(maps_arr, gts_arr, max_fpr=0.05), 4)
    return result
