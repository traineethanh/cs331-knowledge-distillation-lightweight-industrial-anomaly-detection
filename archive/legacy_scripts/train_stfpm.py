"""Reproducible Training Script for Foundational STFPM Baseline (Wang et al., BMVC 2021).

Trains Student-Teacher Feature Pyramid Matching on individual MVTec AD categories.
Supports:
- 80/20 train/validation split on normal training images (paper protocol)
- Periodic checkpoint saving and automatic best validation checkpointing
- Seamless training resumption from checkpoint (--resume)
- Clean standardized directory structure: results/baseline_stfpm/<category>/
- Accurate hardware VRAM, runtime, and loss tracking
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from anomaly_detection.data import IMAGENET_MEAN, IMAGENET_STD, MVTecTrainDataset
from anomaly_detection.models.baseline.stfpm import STFPM, STFPMLoss
from anomaly_detection.utils import load_yaml_config, save_yaml_config, set_seed


def _get_git_commit() -> str:
    """Safely retrieves current git commit hash if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def evaluate_val_loss(
    model: STFPM,
    loss_fn: STFPMLoss,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluates distillation loss on the 20% holdout normal validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            t_feats, s_feats = model(images)
            loss = loss_fn(t_feats, s_feats)
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train STFPM foundational baseline on MVTec AD.")
    parser.add_argument("--config", type=str, default="configs/baseline/stfpm_mvtec.yaml")
    parser.add_argument("--data-root", "--root-dir", dest="data_root", type=str, default=None)
    parser.add_argument("--category", "--obj-name", dest="category", type=str, default="carpet")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation ratio from normal train images (paper: 0.2)")
    parser.add_argument("--save-interval", type=int, default=None, help="Epoch interval to save periodic checkpoints")
    parser.add_argument("--output-dir", "--save-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu')")
    parser.add_argument("--shuffle", dest="shuffle", action="store_true", default=None, help="Shuffle training dataloader")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable training dataloader shuffling")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    # Load configuration file if present
    cfg = load_yaml_config(args.config) if Path(args.config).exists() else {}

    seed = args.seed if args.seed is not None else int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    data_root = args.data_root if args.data_root is not None else cfg.get("dataset", {}).get("data_root", "/kaggle/input/mvtec-ad")
    epochs = args.epochs if args.epochs is not None else int(cfg.get("training", {}).get("epochs", 100))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("training", {}).get("batch_size", 32))
    lr = args.lr if args.lr is not None else float(cfg.get("training", {}).get("lr", 0.4))
    momentum = args.momentum if args.momentum is not None else float(cfg.get("training", {}).get("momentum", 0.9))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(cfg.get("training", {}).get("weight_decay", 1e-4))
    img_size = int(cfg.get("dataset", {}).get("img_size", 256))
    val_ratio = args.val_ratio if args.val_ratio is not None else float(cfg.get("training", {}).get("val_split_ratio", 0.2))
    save_interval = args.save_interval if args.save_interval is not None else int(cfg.get("training", {}).get("save_interval", 10))
    shuffle = args.shuffle if args.shuffle is not None else bool(cfg.get("training", {}).get("shuffle", True))
    drop_last = bool(cfg.get("training", {}).get("drop_last", False))
    gaussian_sigma = float(cfg.get("evaluation", {}).get("gaussian_sigma", cfg.get("model", {}).get("gaussian_sigma", 0.0)))

    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 0 if sys.platform == "win32" else int(cfg.get("training", {}).get("num_workers", 2))

    # Standardized output directory structure: results/baseline_stfpm/<category>/
    out_dir_path = args.output_dir or f"results/baseline_stfpm/{args.category}"
    out_dir = Path(out_dir_path)
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Determine execution device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[*] Initializing STFPM on category: '{args.category}' using device: {device}")
    print(f"[*] Hyperparameters: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, WD={weight_decay}, ValRatio={val_ratio}, Seed={seed}")

    # Build Train & Validation datasets on normal images
    train_ds = MVTecTrainDataset(
        root_dir=data_root,
        obj_name=args.category,
        resize_shape=img_size,
        split="train",
        val_ratio=val_ratio,
        seed=seed,
    )
    val_ds = MVTecTrainDataset(
        root_dir=data_root,
        obj_name=args.category,
        resize_shape=img_size,
        split="val",
        val_ratio=val_ratio,
        seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    ) if len(val_ds) > 0 else None

    print(f"[*] Loaded {len(train_ds)} train normal images and {len(val_ds)} validation normal images.")

    # Save deterministic split manifest for traceability and audit verification
    root_path = Path(data_root)
    train_rel = [str(p.relative_to(root_path)) if p.is_relative_to(root_path) else str(p) for p in train_ds.image_paths]
    val_rel = [str(p.relative_to(root_path)) if p.is_relative_to(root_path) else str(p) for p in val_ds.image_paths]
    split_manifest = {
        "category": args.category,
        "seed": seed,
        "val_ratio": val_ratio,
        "num_train_good": len(train_rel),
        "num_val_good": len(val_rel),
        "zero_train_val_overlap": len(set(train_rel).intersection(set(val_rel))) == 0,
        "train_files": train_rel,
        "val_files": val_rel,
    }
    for manifest_target in (out_dir / "split_manifest.json", log_dir / "split_manifest.json"):
        with open(manifest_target, "w", encoding="utf-8") as f:
            json.dump(split_manifest, f, indent=2)

    # Initialize STFPM model and loss
    model = STFPM(teacher_pretrained=True).to(device)
    loss_fn = STFPMLoss().to(device)
    optimizer = torch.optim.SGD(
        model.student.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    start_epoch = 1
    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    peak_vram_mb = 0.0

    # Resume training if requested
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint for resume not found: {resume_path}")
        print(f"[*] Resuming training from checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.student.load_state_dict(ckpt["student_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        history = list(ckpt.get("history", []))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        print(f"[*] Successfully resumed from epoch {start_epoch - 1}. Target epochs: {epochs}")

    # Prepare training log CSV
    history_csv = log_dir / "training_history.csv"
    csv_mode = "a" if (args.resume and history_csv.exists()) else "w"
    with open(history_csv, csv_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if csv_mode == "w":
            writer.writerow(["epoch", "train_loss", "val_loss", "lr", "epoch_time_sec", "peak_vram_mb"])

    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()  # STFPM.train() ensures model.teacher permanently remains in eval()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()

            t_feats, s_feats = model(images)
            loss = loss_fn(t_feats, s_feats)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Evaluate on validation holdout
        if val_loader is not None:
            val_loss = evaluate_val_loss(model, loss_fn, val_loader, device=device)
        else:
            val_loss = avg_train_loss

        epoch_duration = time.time() - epoch_start

        if torch.cuda.is_available() and device.type == "cuda":
            current_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_vram_mb = max(peak_vram_mb, current_vram)

        record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": lr,
            "epoch_time_sec": round(epoch_duration, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }
        history.append(record)

        # Append to CSV log
        with open(history_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, record["train_loss"], record["val_loss"], lr, record["epoch_time_sec"], record["peak_vram_mb"]])

        # Check for best validation model
        is_best_val = val_loss < best_val_loss
        if is_best_val:
            best_val_loss = val_loss
            torch.save(
                {
                    "category": args.category,
                    "epoch": epoch,
                    "student_state_dict": model.student.state_dict(),
                    "teacher_weights": "IMAGENET1K_V1",
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "seed": seed,
                },
                ckpt_dir / "best_val.pth",
            )

        # Periodic checkpoint
        if save_interval > 0 and (epoch % save_interval == 0 or epoch == epochs):
            torch.save(
                {
                    "category": args.category,
                    "epoch": epoch,
                    "student_state_dict": model.student.state_dict(),
                    "teacher_weights": "IMAGENET1K_V1",
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "seed": seed,
                },
                ckpt_dir / "last.pth",
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs or is_best_val:
            star = " [*BEST*]" if is_best_val else ""
            print(f"    Epoch [{epoch:03d}/{epochs:03d}] - Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}{star} ({epoch_duration:.2f}s)")

    total_duration = time.time() - start_time
    print(f"[*] Training finished in {total_duration:.2f}s ({total_duration/60:.2f} min). Peak VRAM: {peak_vram_mb:.1f} MB")

    # Ensure last.pth is saved
    torch.save(
        {
            "category": args.category,
            "epoch": epochs,
            "student_state_dict": model.student.state_dict(),
            "teacher_weights": "IMAGENET1K_V1",
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "peak_vram_mb": peak_vram_mb,
            "total_duration_sec": total_duration,
            "seed": seed,
        },
        ckpt_dir / "last.pth",
    )
    print(f"[*] Final checkpoint saved to: {ckpt_dir / 'last.pth'}")
    if (ckpt_dir / "best_val.pth").exists():
        print(f"[*] Best validation checkpoint saved to: {ckpt_dir / 'best_val.pth'}")

    git_commit = _get_git_commit()
    dataset_mean = list(cfg.get("dataset", {}).get("mean", IMAGENET_MEAN))
    dataset_std = list(cfg.get("dataset", {}).get("std", IMAGENET_STD))

    # Save resolved configuration in output directory
    resolved_cfg = {
        "model": "STFPM",
        "variant": "STFPM-Paper" if gaussian_sigma <= 0.0 else f"STFPM-Smoothed(sigma={gaussian_sigma})",
        "category": args.category,
        "dataset": cfg.get("dataset", {}).get("name", "mvtec_ad"),
        "data_root": str(data_root),
        "preprocessing": {
            "img_size": img_size,
            "interpolation": "bilinear",
            "mean": dataset_mean,
            "std": dataset_std,
            "mask_threshold": 0.5,
        },
        "optimizer": "sgd",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "seed": seed,
        "val_ratio": val_ratio,
        "save_interval": save_interval,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "img_size": img_size,
        "gaussian_sigma": gaussian_sigma,
        "metric_protocol": {
            "metrics": ["auroc_img", "ap_img", "auroc_pix", "ap_pix", "aupro_30", "aupro_05"],
            "aupro_fpr_limits": [0.30, 0.05],
            "aupro_num_thresholds": 500,
            "aupro_normalization": "strict_max_fpr",
        },
        "git_commit": git_commit,
        "device": str(device),
    }
    save_yaml_config(resolved_cfg, out_dir / "config.yaml")

    # Save training summary JSON
    summary_path = log_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "STFPM",
                "variant": resolved_cfg["variant"],
                "category": args.category,
                "dataset": resolved_cfg["dataset"],
                "preprocessing": resolved_cfg["preprocessing"],
                "optimizer": "sgd",
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "val_ratio": val_ratio,
                "shuffle": shuffle,
                "seed": seed,
                "gaussian_sigma": gaussian_sigma,
                "metric_protocol": resolved_cfg["metric_protocol"],
                "git_commit": git_commit,
                "device": str(device),
                "peak_vram_mb": peak_vram_mb,
                "total_duration_sec": total_duration,
                "final_train_loss": history[-1]["train_loss"] if history else None,
                "final_val_loss": history[-1]["val_loss"] if history else None,
                "best_val_loss": best_val_loss if val_loader is not None else None,
                "last_checkpoint": str(ckpt_dir / "last.pth"),
                "best_checkpoint": str(ckpt_dir / "best_val.pth") if (ckpt_dir / "best_val.pth").exists() else None,
                "split_manifest": str(out_dir / "split_manifest.json"),
            },
            f,
            indent=2,
        )
    print(f"[*] Saved training summary to: {summary_path}")


if __name__ == "__main__":
    main()
