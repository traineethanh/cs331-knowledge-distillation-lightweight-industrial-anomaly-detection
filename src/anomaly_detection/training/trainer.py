"""Reproducible CDA trainer supporting full-state checkpointing and Kaggle session resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..evaluation.metrics import evaluate_predictions
from ..inference.anomaly_mapper import CDAAnomalyMapper, predict_batch
from ..losses.cda_loss import CDALoss


def save_checkpoint(
    path: str | Path,
    student: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Saves both full training state and raw state_dict compatibility."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": student.state_dict(),
        "best_metrics": best_metrics or {},
        "config": config or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path,
    student: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Loads either our full checkpoint dict or an official CDA raw `state_dict`."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        student.load_state_dict(raw["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in raw:
            optimizer.load_state_dict(raw["optimizer_state_dict"])
        return {
            "epoch": int(raw.get("epoch", 0)),
            "best_metrics": raw.get("best_metrics", {}),
            "config": raw.get("config", {}),
        }

    # Official WHUer-cloud/CDA checkpoints store raw student.state_dict() directly
    student.load_state_dict(raw)
    return {"epoch": 0, "best_metrics": {}, "config": {}}


class CDATrainer:
    """Manages training, periodic evaluation, and leak-free checkpoint archiving."""

    def __init__(
        self,
        encoder: nn.Module,
        student: nn.Module,
        loss_fn: CDALoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str | Path,
        img_size: int = 256,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.encoder = encoder.to(device).eval()
        self.student = student.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.img_size = int(img_size)
        self.config = config or {}

    def train_one_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.encoder.eval()
        self.student.train()

        total_loss = 0.0
        total_same = 0.0
        total_cross = 0.0
        num_batches = 0

        for batch in train_loader:
            img = batch["image"].to(self.device)
            with torch.no_grad():
                e1, e2, e3 = self.encoder(img)

            outputs = self.student(e1, e2, e3)
            loss, parts = self.loss_fn([e1, e2, e3], outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += parts["loss_total"]
            total_same += parts["loss_same"]
            total_cross += parts["loss_cross"]
            num_batches += 1

        denom = max(1, num_batches)
        return {
            "loss_total": total_loss / denom,
            "loss_same": total_same / denom,
            "loss_cross": total_cross / denom,
        }

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        mapper: Optional[CDAAnomalyMapper] = None,
        include_aupro: bool = True,
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        active_mapper = mapper or CDAAnomalyMapper(img_size=self.img_size)
        all_maps: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_labels: List[int] = []
        all_gts: List[np.ndarray] = []

        for batch in test_loader:
            imgs = batch["image"].to(self.device)
            maps, scores = predict_batch(self.encoder, self.student, imgs, active_mapper)
            all_maps.append(maps)
            all_scores.append(scores)
            labels_np = batch["label"].reshape(-1).cpu().numpy()
            for lbl in labels_np:
                all_labels.append(int(lbl))
            gts_np = batch["gt_mask"].squeeze(1).cpu().numpy()
            all_gts.append(gts_np)

        maps_cat = np.concatenate(all_maps, axis=0)
        scores_cat = np.concatenate(all_scores, axis=0)
        labels_arr = np.asarray(all_labels, dtype=np.int32)
        gts_cat = np.concatenate(all_gts, axis=0)

        metrics_dict = evaluate_predictions(
            anomaly_maps=maps_cat,
            image_labels=labels_arr,
            gt_masks=gts_cat,
            image_scores=scores_cat,
            include_aupro=include_aupro,
        )
        return metrics_dict, maps_cat, labels_arr, gts_cat

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        epochs: int = 160,
        start_epoch: int = 0,
        save_interval: int = 10,
        eval_interval: int = 20,
    ) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        best_metrics: Dict[str, Any] = {
            "best_test_auroc_img": 0.0,
            "best_test_auroc_pix": 0.0,
        }

        for epoch in range(start_epoch, epochs):
            train_stats = self.train_one_epoch(train_loader)
            record: Dict[str, Any] = {"epoch": epoch + 1, **train_stats}

            is_last = (epoch + 1) == epochs
            if (save_interval > 0 and (epoch + 1) % save_interval == 0) or is_last:
                save_checkpoint(
                    self.ckpt_dir / "last.pth",
                    student=self.student,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    best_metrics=best_metrics,
                    config=self.config,
                )

            if test_loader is not None and (
                (eval_interval > 0 and (epoch + 1) % eval_interval == 0) or is_last
            ):
                eval_res, _, _, _ = self.evaluate(test_loader, include_aupro=is_last)
                record.update(eval_res)
                if eval_res["auroc_img"] > best_metrics["best_test_auroc_img"]:
                    best_metrics["best_test_auroc_img"] = eval_res["auroc_img"]
                    best_metrics["best_img_epoch"] = epoch + 1
                    save_checkpoint(
                        self.ckpt_dir / "best_test_img.pth",
                        student=self.student,
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        best_metrics=best_metrics,
                        config=self.config,
                    )
                if eval_res["auroc_pix"] > best_metrics["best_test_auroc_pix"]:
                    best_metrics["best_test_auroc_pix"] = eval_res["auroc_pix"]
                    best_metrics["best_pix_epoch"] = epoch + 1
                    save_checkpoint(
                        self.ckpt_dir / "best_test_pix.pth",
                        student=self.student,
                        optimizer=self.optimizer,
                        epoch=epoch + 1,
                        best_metrics=best_metrics,
                        config=self.config,
                    )

            history.append(record)

        summary = {
            "epochs_completed": epochs,
            "last_epoch_metrics": history[-1] if history else {},
            "best_test_reference": best_metrics,
            "history": history,
        }
        with open(self.output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary
