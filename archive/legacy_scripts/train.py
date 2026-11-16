"""Reproducible Training Script for CDA Baseline (`B0`) and Research B Variants (`B2`, `B3`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from anomaly_detection.data import MVTecTestDataset, MVTecTrainDataset
from anomaly_detection.losses import CDALoss
from anomaly_detection.models.baseline import TeacherEncoder
from anomaly_detection.models.research_b import build_student_model
from anomaly_detection.training import CDATrainer, load_checkpoint
from anomaly_detection.utils import load_yaml_config, save_yaml_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CDA baseline or lightweight variants.")
    parser.add_argument("--config", type=str, default="configs/baseline/cda_mvtec.yaml")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)

    data_root = args.data_root or cfg["dataset"]["data_root"]
    categories = args.categories or cfg["dataset"].get("core_categories", ["carpet", "bottle", "screw"])
    variant = args.variant or cfg["model"].get("variant", "B0")
    epochs = args.epochs or int(cfg["training"]["epochs"])
    img_size = int(cfg["dataset"].get("img_size", 256))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = TeacherEncoder(pretrained=bool(cfg["model"].get("teacher_pretrained", True)))

    loss_cfg = cfg.get("loss", {})
    loss_fn = CDALoss(
        lambda_same=loss_cfg.get("lambda_same", 2.0),
        lambda_cross=loss_cfg.get("lambda_cross", 0.1),
        cross_scale_weights=loss_cfg.get("cross_scale_weights", [0.4, 0.2, 0.4]),
        cross_loss_mode=loss_cfg.get("cross_loss_mode", "official_train"),
    )

    base_out = Path(cfg["experiment"].get("output_dir", "./outputs/baseline")) / variant

    for cat in categories:
        set_seed(seed)
        student = build_student_model(variant=variant)
        train_cfg = cfg["training"]
        optimizer = torch.optim.Adam(
            student.parameters(),
            lr=float(train_cfg.get("lr", 0.005)),
            betas=tuple(train_cfg.get("betas", [0.5, 0.999])),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )

        start_epoch = 0
        if args.resume:
            meta = load_checkpoint(args.resume, student=student, optimizer=optimizer, map_location=device)
            start_epoch = int(meta.get("epoch", 0))

        train_ds = MVTecTrainDataset(root_dir=data_root, obj_name=cat, resize_shape=img_size)
        test_ds = MVTecTestDataset(root_dir=data_root, obj_name=cat, resize_shape=img_size)

        train_loader = DataLoader(
            train_ds,
            batch_size=int(train_cfg.get("batch_size", 4)),
            shuffle=bool(train_cfg.get("shuffle", False)),
            drop_last=bool(train_cfg.get("drop_last", True)),
            num_workers=int(train_cfg.get("num_workers", 0)),
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

        cat_out = base_out / cat
        save_yaml_config(cfg, cat_out / "resolved_config.yaml")

        trainer = CDATrainer(
            encoder=encoder,
            student=student,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            output_dir=cat_out,
            img_size=img_size,
            config=cfg,
        )
        summary = trainer.fit(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            start_epoch=start_epoch,
            save_interval=int(train_cfg.get("save_interval", 10)),
            eval_interval=int(train_cfg.get("eval_interval", 20)),
        )
        print(f"[{variant} | {cat}] Completed {epochs} epochs: {json.dumps(summary['last_epoch_metrics'])}")


if __name__ == "__main__":
    main()
