#!/usr/bin/env python3
"""CLI Training Script for Dual-Student Knowledge Distillation (DSKD).

Usage:
    python scripts/train.py --config configs/baseline_b0.yaml --category carpet --epochs 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.anomaly_detection.data.mvtec import MVTecTrainDataset, build_image_transform
from src.anomaly_detection.losses.dskd_loss import DSKDLoss
from src.anomaly_detection.models.dskd import DSKDModel
from src.anomaly_detection.training.trainer import DSKDTrainer
from src.anomaly_detection.utils.config import load_yaml_config
from src.anomaly_detection.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DSKD Anomaly Detection Model")
    parser.add_argument("--config", type=str, default="configs/baseline_b0.yaml", help="Path to YAML config")
    parser.add_argument("--category", type=str, default=None, help="MVTec category (e.g., carpet, bottle)")
    parser.add_argument("--data_root", "--data_dir", type=str, default=None, help="Path to MVTec AD dataset directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--dfe_input_type", type=str, default=None, choices=["raw", "norm"], help="DFE input type")
    parser.add_argument("--detach_dfe_input", type=lambda x: str(x).lower() in ("true", "1"), default=None, help="Stop gradient to Se")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save checkpoints and logs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    # Apply CLI overrides
    category = args.category or config.get("dataset", {}).get("category", "carpet")
    data_root = args.data_root or config.get("dataset", {}).get("data_root", "/kaggle/input/mvtec-ad")
    img_size = config.get("dataset", {}).get("img_size", 256)
    val_split_ratio = config.get("dataset", {}).get("val_split_ratio", 0.2)
    seed = args.seed if args.seed is not None else config.get("dataset", {}).get("seed", 42)

    epochs = args.epochs or config.get("training", {}).get("epochs", 200)
    batch_size = args.batch_size or config.get("training", {}).get("batch_size", 16)
    num_workers = config.get("training", {}).get("num_workers", 2)
    val_frequency = config.get("training", {}).get("val_frequency", 5)

    learning_rate = args.lr or config.get("optimizer", {}).get("learning_rate", 0.001)
    betas = tuple(config.get("optimizer", {}).get("betas", [0.5, 0.999]))
    weight_decay = config.get("optimizer", {}).get("weight_decay", 0.0)
    clip_grad_norm = config.get("optimizer", {}).get("clip_grad_norm", 1.0)

    dfe_input_type = args.dfe_input_type or config.get("model", {}).get("dfe_input_type", "raw")
    detach_dfe_input = args.detach_dfe_input if args.detach_dfe_input is not None else config.get("model", {}).get("detach_dfe_input", True)
    has_f3_conv = config.get("model", {}).get("has_f3_conv", True)
    upsample_mode = config.get("model", {}).get("upsample_mode", "bilinear")

    loss_lambda = config.get("loss", {}).get("lambda", 0.1)
    norm_variant = config.get("loss", {}).get("norm_variant", "N1")

    base_checkpoint_dir = args.output_dir or config.get("training", {}).get("checkpoint_dir", "checkpoints/baseline_b0")
    checkpoint_dir = Path(base_checkpoint_dir) / category
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    set_seed(seed)

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"==================================================================")
    print(f"DSKD Training Setup:")
    print(f"  Category:        {category}")
    print(f"  Data Root:       {data_root}")
    print(f"  Device:          {device}")
    print(f"  Epochs:          {epochs}")
    print(f"  Batch Size:      {batch_size}")
    print(f"  Learning Rate:   {learning_rate}")
    print(f"  DFE Input Type:  {dfe_input_type}")
    print(f"  Detach DFE:      {detach_dfe_input}")
    print(f"  Checkpoint Dir:  {checkpoint_dir}")
    print(f"==================================================================")

    # Build datasets
    train_transform = build_image_transform(resize_shape=img_size)

    train_dataset = MVTecTrainDataset(
        root_dir=data_root,
        obj_name=category,
        resize_shape=img_size,
        transform=train_transform,
        split="train",
        val_ratio=val_split_ratio,
        seed=seed,
    )

    val_dataset = MVTecTrainDataset(
        root_dir=data_root,
        obj_name=category,
        resize_shape=img_size,
        transform=train_transform,
        split="val",
        val_ratio=val_split_ratio,
        seed=seed,
    )

    print(f"Dataset loaded: {len(train_dataset)} train samples, {len(val_dataset)} validation samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
    ) if len(val_dataset) > 0 else None

    # Instantiate model
    model = DSKDModel(
        pretrained_teacher=True,
        has_f3_conv=has_f3_conv,
        dfe_input_type=dfe_input_type,
        detach_dfe_input=detach_dfe_input,
        upsample_mode=upsample_mode,
    )

    # Instantiate loss
    loss_fn = DSKDLoss(
        loss_lambda=loss_lambda,
        norm_variant=norm_variant,
    )

    # Instantiate trainer
    trainer = DSKDTrainer(
        model=model,
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
        device=device,
        checkpoint_dir=checkpoint_dir,
        clip_grad_norm=clip_grad_norm,
    )

    # Run training
    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        val_frequency=val_frequency,
    )

    # Save results summary
    history_file = checkpoint_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Training completed successfully! Saved history to: {history_file}")


if __name__ == "__main__":
    main()
