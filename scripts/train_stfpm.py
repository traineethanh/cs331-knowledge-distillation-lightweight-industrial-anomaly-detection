"""Reproducible Training Script for Foundational STFPM Baseline (Wang et al., BMVC 2021)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from anomaly_detection.data import MVTecTrainDataset
from anomaly_detection.models.baseline.stfpm import STFPM, STFPMLoss
from anomaly_detection.utils import load_yaml_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train STFPM foundational baseline on MVTec AD.")
    parser.add_argument("--config", type=str, default="configs/baseline/stfpm_mvtec.yaml")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--category", type=str, default="carpet")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config) if Path(args.config).exists() else {}
    seed = args.seed or int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    data_root = args.data_root or cfg.get("dataset", {}).get("data_root", "/kaggle/input/mvtec-ad")
    epochs = args.epochs or int(cfg.get("training", {}).get("epochs", 100))
    batch_size = args.batch_size or int(cfg.get("training", {}).get("batch_size", 32))
    lr = args.lr or float(cfg.get("training", {}).get("lr", 0.4))
    momentum = float(cfg.get("training", {}).get("momentum", 0.9))
    weight_decay = float(cfg.get("training", {}).get("weight_decay", 1e-4))
    img_size = int(cfg.get("dataset", {}).get("img_size", 256))

    save_dir = Path(args.save_dir or f"./outputs/stfpm/{args.category}")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing STFPM on category: '{args.category}' using device: {device}")
    print(f"[*] Hyperparameters: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, WD={weight_decay}, Seed={seed}")

    train_ds = MVTecTrainDataset(data_root=data_root, category=args.category, img_size=img_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,  # Following official STFPM protocol
        drop_last=False,
        num_workers=0 if sys.platform == "win32" else 2,
    )
    print(f"[*] Loaded {len(train_ds)} normal training images across {len(train_loader)} batches.")

    model = STFPM(teacher_pretrained=True).to(device)
    loss_fn = STFPMLoss().to(device)
    optimizer = torch.optim.SGD(
        model.student.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    history = []
    start_time = time.time()
    peak_vram_mb = 0.0

    model.teacher.eval()
    for epoch in range(1, epochs + 1):
        model.student.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()

            t_feats, s_feats = model(images)
            loss = loss_fn(t_feats, s_feats)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss})

        if torch.cuda.is_available():
            current_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_vram_mb = max(peak_vram_mb, current_vram)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"    Epoch [{epoch:03d}/{epochs:03d}] - Loss: {avg_loss:.6f}")

    total_duration = time.time() - start_time
    print(f"[*] Training finished in {total_duration:.2f}s ({total_duration/60:.2f} min). Peak VRAM: {peak_vram_mb:.1f} MB")

    # Save final checkpoint and training metadata
    checkpoint_path = save_dir / "last.pth"
    torch.save(
        {
            "category": args.category,
            "epoch": epochs,
            "student_state_dict": model.student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "peak_vram_mb": peak_vram_mb,
            "total_duration_sec": total_duration,
            "seed": seed,
        },
        checkpoint_path,
    )
    print(f"[*] Saved checkpoint to: {checkpoint_path}")

    # Save training summary JSON
    summary_path = save_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "STFPM",
                "category": args.category,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "device": str(device),
                "peak_vram_mb": peak_vram_mb,
                "total_duration_sec": total_duration,
                "final_loss": history[-1]["loss"] if history else None,
            },
            f,
            indent=2,
        )
    print(f"[*] Saved training summary to: {summary_path}")


if __name__ == "__main__":
    main()
