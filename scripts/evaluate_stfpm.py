"""Reproducible Evaluation Script for STFPM Baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from anomaly_detection.data import MVTecTestDataset
from anomaly_detection.evaluation import evaluate_predictions
from anomaly_detection.models.baseline.stfpm import STFPM, compute_stfpm_anomaly_map
from anomaly_detection.utils import load_yaml_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate STFPM checkpoint on MVTec AD.")
    parser.add_argument("--config", type=str, default="configs/baseline/stfpm_mvtec.yaml")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--category", type=str, default="carpet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config) if Path(args.config).exists() else {}
    data_root = args.data_root or cfg.get("dataset", {}).get("data_root", "/kaggle/input/mvtec-ad")
    img_size = int(cfg.get("dataset", {}).get("img_size", 256))
    out_dir = Path(args.output_dir or f"./outputs/stfpm_eval/{args.category}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating STFPM on category: '{args.category}' using device: {device}")

    # Load test dataset
    test_ds = MVTecTestDataset(data_root=data_root, category=args.category, img_size=img_size)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"[*] Loaded {len(test_ds)} test images.")

    # Initialize model and load trained student weights
    model = STFPM(teacher_pretrained=True).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    student_state = ckpt.get("student_state_dict", ckpt)
    model.student.load_state_dict(student_state)
    model.eval()

    all_maps = []
    all_scores = []
    all_labels = []
    all_masks = []
    all_paths = []

    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            mask = batch["mask"].squeeze(0).numpy().astype(np.uint8)
            label = int(batch["label"].item())
            img_path = batch["img_path"][0]

            t_feats, s_feats = model(image)
            anomaly_map, img_score = compute_stfpm_anomaly_map(t_feats, s_feats, target_size=(img_size, img_size))

            all_maps.append(anomaly_map.squeeze(0).cpu().numpy())
            all_scores.append(float(img_score.cpu().item()))
            all_labels.append(label)
            all_masks.append(mask)
            all_paths.append(img_path)

    total_infer_time = time.time() - start_time
    avg_latency_ms = (total_infer_time / max(len(test_ds), 1)) * 1000.0
    print(f"[*] Inference finished in {total_infer_time:.2f}s. Avg Latency: {avg_latency_ms:.2f} ms/img ({1000.0/max(avg_latency_ms, 1e-4):.1f} FPS)")

    maps_arr = np.stack(all_maps, axis=0)
    scores_arr = np.array(all_scores, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)
    masks_arr = np.stack(all_masks, axis=0)

    print("[*] Computing AUROC, AP, and AUPRO metrics...")
    metrics = evaluate_predictions(
        maps=maps_arr,
        scores=scores_arr,
        masks=masks_arr,
        labels=labels_arr,
    )
    metrics["latency_ms_per_image"] = avg_latency_ms
    metrics["throughput_fps"] = 1000.0 / max(avg_latency_ms, 1e-4)
    metrics["category"] = args.category
    metrics["model"] = "STFPM"

    print("\n" + "=" * 50)
    print(f"--- STFPM Evaluation Results for [{args.category}] ---")
    print(f"  Image-Level AUROC : {metrics['auroc_img']:.4f}")
    print(f"  Image-Level AP    : {metrics['ap_img']:.4f}")
    print(f"  Pixel-Level AUROC : {metrics['auroc_pix']:.4f}")
    print(f"  Pixel-Level AP    : {metrics['ap_pix']:.4f}")
    print(f"  AUPRO (FPR <= 0.30): {metrics['aupro_30']:.4f}")
    print(f"  AUPRO (FPR <= 0.05): {metrics['aupro_05']:.4f}")
    print("=" * 50 + "\n")

    summary_file = out_dir / "metrics_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[*] Saved metrics summary to: {summary_file}")


if __name__ == "__main__":
    main()
