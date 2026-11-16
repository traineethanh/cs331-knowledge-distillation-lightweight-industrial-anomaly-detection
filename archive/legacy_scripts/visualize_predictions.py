"""Qualitative Heatmap Visualization Script for Error Analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from anomaly_detection.data import IMAGENET_MEAN, IMAGENET_STD, MVTecTestDataset
from anomaly_detection.inference import CDAAnomalyMapper, predict_batch
from anomaly_detection.models.baseline import TeacherEncoder
from anomaly_detection.models.research_b import build_student_model
from anomaly_detection.training import load_checkpoint


def denormalize_image(tensor_img: torch.Tensor) -> np.ndarray:
    img = tensor_img.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img = np.clip(img * std + mean, 0.0, 1.0)
    return img


def save_visualization_panel(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    anomaly_map: np.ndarray,
    save_path: str | Path,
    title: str = "",
) -> Path:
    target = Path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground-Truth Mask")
    axes[1].axis("off")

    im = axes[2].imshow(anomaly_map, cmap="jet")
    axes[2].set_title("Anomaly Map")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(image_rgb)
    axes[3].imshow(anomaly_map, cmap="jet", alpha=0.5)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Export qualitative anomaly localization heatmaps.")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--variant", type=str, default="B0")
    parser.add_argument("--fusion-mode", type=str, default="A0_baseline")
    parser.add_argument("--output-dir", type=str, default="./outputs/visualizations")
    parser.add_argument("--max-samples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TeacherEncoder(pretrained=True).to(device).eval()
    student = build_student_model(variant=args.variant).to(device)
    load_checkpoint(args.checkpoint, student=student, map_location=device)
    student.eval()

    ds = MVTecTestDataset(root_dir=args.data_root, obj_name=args.category, resize_shape=256)
    mapper = CDAAnomalyMapper(img_size=256, fusion_mode=args.fusion_mode)

    count = 0
    for idx in range(len(ds)):
        if count >= args.max_samples:
            break
        sample = ds[idx]
        img_t = sample["image"].unsqueeze(0).to(device)
        maps, scores = predict_batch(encoder, student, img_t, mapper)
        img_rgb = denormalize_image(sample["image"])
        gt = sample["gt_mask"].squeeze(0).numpy()
        out_file = Path(args.output_dir) / args.category / f"sample_{idx:03d}_{sample['defect_type']}.png"
        save_visualization_panel(
            image_rgb=img_rgb,
            gt_mask=gt,
            anomaly_map=maps[0],
            save_path=out_file,
            title=f"{args.category} | {sample['defect_type']} | score={scores[0]:.4f}",
        )
        count += 1


if __name__ == "__main__":
    main()
