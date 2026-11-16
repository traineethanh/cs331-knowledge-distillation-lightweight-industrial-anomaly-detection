#!/usr/bin/env python3
"""CLI Evaluation Script for DSKD.

Computes:
- Image-level AUROC & Average Precision
- Pixel-level AUROC & Average Precision
- Pixel-level AUPRO @ max_fpr = 0.30 and 0.05

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/baseline_b0/carpet/best_checkpoint.pth --category carpet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.anomaly_detection.data.mvtec import MVTecTestDataset, build_image_transform, build_mask_transform
from src.anomaly_detection.evaluation.metrics import compute_aupro, compute_auroc_and_ap
from src.anomaly_detection.inference.anomaly_scorer import DSKDAnomalyScorer
from src.anomaly_detection.models.dskd import DSKDModel
from src.anomaly_detection.training.checkpoint import CheckpointManager
from src.anomaly_detection.utils.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DSKD Anomaly Detection Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="configs/baseline_b0.yaml", help="Path to YAML config")
    parser.add_argument("--category", type=str, default=None, help="MVTec category")
    parser.add_argument("--data_root", "--data_dir", type=str, default=None, help="Path to MVTec AD dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    category = args.category or config.get("dataset", {}).get("category", "carpet")
    data_root = args.data_root or config.get("dataset", {}).get("data_root", "/kaggle/input/mvtec-ad")
    img_size = config.get("dataset", {}).get("img_size", 256)

    gaussian_sigma = config.get("evaluation", {}).get("gaussian_sigma", 4.0)
    loss_lambda = config.get("loss", {}).get("lambda", 0.1)
    norm_variant = config.get("loss", {}).get("norm_variant", "N1")

    dfe_input_type = config.get("model", {}).get("dfe_input_type", "raw")
    detach_dfe_input = config.get("model", {}).get("detach_dfe_input", True)
    has_f3_conv = config.get("model", {}).get("has_f3_conv", True)
    upsample_mode = config.get("model", {}).get("upsample_mode", "bilinear")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir or Path(args.checkpoint).parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"==================================================================")
    print(f"DSKD Evaluation Setup:")
    print(f"  Category:        {category}")
    print(f"  Data Root:       {data_root}")
    print(f"  Checkpoint:      {args.checkpoint}")
    print(f"  Device:          {device}")
    print(f"==================================================================")

    # 1. Load Model
    model = DSKDModel(
        pretrained_teacher=True,
        has_f3_conv=has_f3_conv,
        dfe_input_type=dfe_input_type,
        detach_dfe_input=detach_dfe_input,
        upsample_mode=upsample_mode,
    ).to(device)

    chk_manager = CheckpointManager(output_dir)
    state = chk_manager.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # 2. Load Test Dataset
    img_transform = build_image_transform(resize_shape=img_size)
    mask_transform = build_mask_transform(resize_shape=img_size)

    test_dataset = MVTecTestDataset(
        root_dir=data_root,
        obj_name=category,
        resize_shape=img_size,
        transform=img_transform,
        gt_transform=mask_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if device.type == "cuda" else 0,
    )

    print(f"Test dataset: {len(test_dataset)} total images.")

    # 3. Generate Anomaly Maps and Predictions
    scorer = DSKDAnomalyScorer(
        img_size=img_size,
        loss_lambda=loss_lambda,
        gaussian_sigma=gaussian_sigma,
        norm_variant=norm_variant,
    )

    all_pred_maps = []
    all_pred_scores = []
    all_gt_masks = []
    all_gt_labels = []

    print("Running inference across test batches...")
    for batch in tqdm(test_loader, desc="Inference"):
        if isinstance(batch, dict):
            images = batch["image"]
            masks = batch["gt_mask"]
            is_anomaly = batch["label"]
        else:
            images, masks, is_anomaly, _, _ = batch
        images = images.to(device)

        batch_maps, batch_scores = scorer.predict_batch(model, images)

        all_pred_maps.append(batch_maps)
        all_pred_scores.append(batch_scores)

        # Convert masks to numpy binary (N, H, W)
        masks_np = masks.squeeze(1).cpu().numpy().astype(np.int32)
        all_gt_masks.append(masks_np)
        all_gt_labels.append(is_anomaly.cpu().numpy().astype(np.int32))

    pred_maps = np.concatenate(all_pred_maps, axis=0)        # [N, H, W]
    pred_scores = np.concatenate(all_pred_scores, axis=0)    # [N]
    gt_masks = np.concatenate(all_gt_masks, axis=0)          # [N, H, W]
    gt_labels = np.concatenate(all_gt_labels, axis=0)        # [N]

    # 4. Compute Metrics
    print("Computing AUROC, Average Precision, and AUPRO metrics...")

    metrics_dict = compute_auroc_and_ap(
        anomaly_maps=pred_maps,
        image_labels=gt_labels,
        gt_masks=gt_masks,
        image_scores=pred_scores,
    )

    # AUPRO @ 0.30 and 0.05
    aupro_30 = compute_aupro(pred_maps, gt_masks, max_fpr=0.30, num_thresholds=500)
    aupro_05 = compute_aupro(pred_maps, gt_masks, max_fpr=0.05, num_thresholds=500)

    results = {
        "category": category,
        "image_auroc": float(metrics_dict["auroc_img"]),
        "image_ap": float(metrics_dict["ap_img"]),
        "pixel_auroc": float(metrics_dict["auroc_pix"]),
        "pixel_ap": float(metrics_dict["ap_pix"]),
        "pixel_aupro_30": float(aupro_30),
        "pixel_aupro_05": float(aupro_05),
        "num_test_samples": int(len(gt_labels)),
        "num_anomalous_samples": int(np.sum(gt_labels)),
    }

    print(f"\n==================================================================")
    print(f"EVALUATION RESULTS FOR '{category.upper()}':")
    print(f"  Image-level AUROC:   {results['image_auroc'] * 100:.2f}%")
    print(f"  Image-level AP:      {results['image_ap'] * 100:.2f}%")
    print(f"  Pixel-level AUROC:   {results['pixel_auroc'] * 100:.2f}%")
    print(f"  Pixel-level AP:      {results['pixel_ap'] * 100:.2f}%")
    print(f"  Pixel-level PRO@0.30:{results['pixel_aupro_30'] * 100:.2f}%")
    print(f"  Pixel-level PRO@0.05:{results['pixel_aupro_05'] * 100:.2f}%")
    print(f"==================================================================")

    # Save to JSON
    json_path = output_dir / f"{category}_eval_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save to CSV
    csv_path = output_dir / f"{category}_eval_metrics.csv"
    pd.DataFrame([results]).to_csv(csv_path, index=False)

    print(f"Results saved to:\n  - {json_path}\n  - {csv_path}")


if __name__ == "__main__":
    main()
