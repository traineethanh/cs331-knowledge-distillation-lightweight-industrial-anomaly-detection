"""Reproducible Evaluation Script for STFPM Baseline (Wang et al., BMVC 2021).

Evaluates trained STFPM models on MVTec AD test sets:
- Computes Image AUROC, Image AP, Pixel AUROC, Pixel AP, AUPRO@0.30, and AUPRO@0.05
- Measures isolated forward inference latency with GPU warmup and CUDA synchronization
- Generates predictions JSON and qualitative heatmap visualizations
- Records results in results/baseline_stfpm/<category>/metrics.json and results/baseline_stfpm/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from anomaly_detection.data import MVTecTestDataset
from anomaly_detection.evaluation import evaluate_predictions
from anomaly_detection.models.baseline.stfpm import STFPM, compute_stfpm_anomaly_map
from anomaly_detection.utils import load_yaml_config


def measure_hardware_latency(
    model: STFPM,
    img_size: int = 256,
    gaussian_sigma: float = 0.0,
    device: torch.device = torch.device("cpu"),
    warmup: int = 10,
    iters: int = 30,
) -> Dict[str, float]:
    """Measures pure model forward latency without DataLoader I/O interference."""
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup):
            t_feats, s_feats = model(dummy)
            _ = compute_stfpm_anomaly_map(
                t_feats, s_feats, target_size=(img_size, img_size), gaussian_sigma=gaussian_sigma
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        timings = []
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            t_feats, s_feats = model(dummy)
            _ = compute_stfpm_anomaly_map(
                t_feats, s_feats, target_size=(img_size, img_size), gaussian_sigma=gaussian_sigma
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    arr = np.asarray(timings, dtype=np.float64)
    return {
        "latency_forward_ms_mean": round(float(arr.mean()), 2),
        "latency_forward_ms_std": round(float(arr.std()), 2),
        "throughput_fps": round(float(1000.0 / max(arr.mean(), 1e-4)), 1),
    }


def save_sample_visualizations(
    out_dir: Path,
    image_paths: List[str],
    labels: List[int],
    masks: List[np.ndarray],
    maps: List[np.ndarray],
    scores: List[float],
    max_samples: int = 5,
) -> None:
    """Generates side-by-side diagnostic images: Original, GT Mask, Heatmap, Overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Pick up to max_samples (prioritizing anomalous samples)
    anom_indices = [i for i, lbl in enumerate(labels) if lbl == 1]
    good_indices = [i for i, lbl in enumerate(labels) if lbl == 0]
    selected = (anom_indices[: max(1, max_samples - 1)] + good_indices[:1])[:max_samples]

    for idx in selected:
        img_path = Path(image_paths[idx])
        gt_mask = masks[idx]
        anom_map = maps[idx]
        score = scores[idx]
        lbl_str = "ANOMALY" if labels[idx] == 1 else "GOOD"

        raw_img = Image.open(img_path).convert("RGB").resize((anom_map.shape[1], anom_map.shape[0]))
        raw_arr = np.asarray(raw_img, dtype=np.float32) / 255.0

        # Normalize map for display
        map_norm = (anom_map - anom_map.min()) / max(float(anom_map.max() - anom_map.min()), 1e-6)

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        axes[0].imshow(raw_arr)
        axes[0].set_title(f"Image: {img_path.name}\n({lbl_str})")
        axes[0].axis("off")

        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        im = axes[2].imshow(map_norm, cmap="jet")
        axes[2].set_title(f"Anomaly Heatmap\nScore: {score:.4f}")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        # Overlay heatmap on original image
        overlay = 0.6 * raw_arr + 0.4 * plt.cm.jet(map_norm)[:, :, :3]
        axes[3].imshow(np.clip(overlay, 0.0, 1.0))
        axes[3].set_title("Overlay Prediction")
        axes[3].axis("off")

        plt.tight_layout()
        save_file = vis_dir / f"sample_{idx:03d}_{img_path.stem}.png"
        plt.savefig(save_file, dpi=120)
        plt.close(fig)


def update_summary_csv(
    summary_csv_path: Path,
    category: str,
    metrics: Dict[str, Any],
) -> None:
    """Appends or updates a row for category in the aggregate summary CSV."""
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "model",
        "auroc_img",
        "ap_img",
        "auroc_pix",
        "ap_pix",
        "aupro_30",
        "aupro_05",
        "latency_forward_ms_mean",
        "throughput_fps",
        "num_test_samples",
    ]

    rows = []
    updated = False
    if summary_csv_path.exists():
        with open(summary_csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("category") == category:
                    rows.append({k: metrics.get(k, r.get(k, "")) for k in fieldnames})
                    updated = True
                else:
                    rows.append(r)

    if not updated:
        rows.append({k: metrics.get(k, "") for k in fieldnames})

    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate STFPM checkpoint on MVTec AD.")
    parser.add_argument("--config", type=str, default="configs/baseline/stfpm_mvtec.yaml")
    parser.add_argument("--data-root", "--root-dir", dest="data_root", type=str, default=None)
    parser.add_argument("--category", "--obj-name", dest="category", type=str, default="carpet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to student checkpoint .pth")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--summary-csv", type=str, default="results/baseline_stfpm/summary.csv")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--gaussian-sigma", type=float, default=None, help="Gaussian smoothing sigma (0.0 for paper-faithful STFPM; 4.0 for STFPM-Smoothed)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-visualizations", type=int, default=5, help="Number of qualitative sample heatmaps to save")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config) if Path(args.config).exists() else {}
    data_root = args.data_root if args.data_root is not None else cfg.get("dataset", {}).get("data_root", "/kaggle/input/mvtec-ad")
    img_size = args.img_size if args.img_size is not None else int(cfg.get("dataset", {}).get("img_size", 256))
    gaussian_sigma = (
        args.gaussian_sigma
        if args.gaussian_sigma is not None
        else float(cfg.get("evaluation", {}).get("gaussian_sigma", cfg.get("model", {}).get("gaussian_sigma", 0.0)))
    )
    variant_name = "STFPM-Paper" if gaussian_sigma <= 0.0 else f"STFPM-Smoothed(sigma={gaussian_sigma})"

    out_dir_path = args.output_dir or f"results/baseline_stfpm/{args.category}"
    out_dir = Path(out_dir_path)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[*] Evaluating STFPM on category: '{args.category}' using device: {device}")
    print(f"[*] Checkpoint: {args.checkpoint}")

    # Load test dataset
    test_ds = MVTecTestDataset(
        root_dir=data_root,
        obj_name=args.category,
        resize_shape=img_size,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"[*] Loaded {len(test_ds)} test images.")

    # Initialize model and load trained student weights
    model = STFPM(teacher_pretrained=True).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    student_state = ckpt.get("student_state_dict", ckpt)
    model.student.load_state_dict(student_state)
    model.eval()

    all_maps = []
    all_scores = []
    all_labels = []
    all_masks = []
    all_paths = []
    predictions_record = []

    print("[*] Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            gt_mask = batch["gt_mask"].squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            label = int(batch["label"].item())
            img_path = batch["path"][0]
            defect_type = batch.get("defect_type", ["unknown"])[0]

            t_feats, s_feats = model(image)
            anomaly_map, img_score = compute_stfpm_anomaly_map(
                t_feats, s_feats, target_size=(img_size, img_size), gaussian_sigma=gaussian_sigma
            )

            map_np = anomaly_map.squeeze(0).cpu().numpy()
            score_val = float(img_score.cpu().item())

            all_maps.append(map_np)
            all_scores.append(score_val)
            all_labels.append(label)
            all_masks.append(gt_mask)
            all_paths.append(img_path)

            predictions_record.append({
                "path": img_path,
                "defect_type": defect_type,
                "label": label,
                "anomaly_score": round(score_val, 6),
            })

    # Save detailed predictions JSON
    with open(pred_dir / "test_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions_record, f, indent=2)

    maps_arr = np.stack(all_maps, axis=0)
    scores_arr = np.array(all_scores, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)
    masks_arr = np.stack(all_masks, axis=0)

    print("[*] Computing AUROC, AP, and AUPRO metrics...")
    metrics_result = evaluate_predictions(
        anomaly_maps=maps_arr,
        image_labels=labels_arr,
        gt_masks=masks_arr,
        image_scores=scores_arr,
        include_aupro=True,
    )

    # Measure isolated hardware latency
    print("[*] Measuring isolated GPU/CPU forward latency (warmup + sync)...")
    latency_info = measure_hardware_latency(model, img_size=img_size, gaussian_sigma=gaussian_sigma, device=device)
    metrics_result.update(latency_info)

    metrics_result["category"] = args.category
    metrics_result["model"] = variant_name
    metrics_result["gaussian_sigma"] = gaussian_sigma
    metrics_result["num_test_samples"] = len(test_ds)
    metrics_result["checkpoint"] = str(args.checkpoint)

    print("\n" + "=" * 55)
    print(f"--- STFPM Evaluation Results for [{args.category}] ---")
    print(f"  Image-Level AUROC  : {metrics_result['auroc_img']:.4f}")
    print(f"  Image-Level AP     : {metrics_result['ap_img']:.4f}")
    print(f"  Pixel-Level AUROC  : {metrics_result['auroc_pix']:.4f}")
    print(f"  Pixel-Level AP     : {metrics_result['ap_pix']:.4f}")
    print(f"  AUPRO (FPR <= 0.30): {metrics_result['aupro_30']:.4f}")
    print(f"  AUPRO (FPR <= 0.05): {metrics_result['aupro_05']:.4f}")
    print(f"  Forward Latency    : {metrics_result['latency_forward_ms_mean']:.2f} ms ({metrics_result['throughput_fps']:.1f} FPS)")
    print("=" * 55 + "\n")

    # Save metrics JSON in category root
    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_result, f, indent=2)
    print(f"[*] Saved metrics JSON to: {metrics_file}")

    # Generate sample visualization heatmaps if requested
    if args.save_visualizations > 0:
        print(f"[*] Saving {args.save_visualizations} qualitative visual overlays...")
        save_sample_visualizations(
            out_dir=out_dir,
            image_paths=all_paths,
            labels=all_labels,
            masks=all_masks,
            maps=all_maps,
            scores=all_scores,
            max_samples=args.save_visualizations,
        )
        print(f"[*] Saved heatmaps to: {out_dir / 'visualizations'}")

    # Update global summary CSV
    summary_csv_path = Path(args.summary_csv)
    update_summary_csv(summary_csv_path, category=args.category, metrics=metrics_result)
    print(f"[*] Updated summary CSV at: {summary_csv_path}")


if __name__ == "__main__":
    main()
