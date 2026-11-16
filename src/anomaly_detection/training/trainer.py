"""Training pipeline for Dual-Student Knowledge Distillation (DSKD).

Reference Paper:
Dual-Student Knowledge Distillation Networks for Unsupervised Anomaly Detection
arXiv:2402.00448v1 - Algorithm 1 (DSKD Training Procedure).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.anomaly_detection.losses.dskd_loss import DSKDLoss
from src.anomaly_detection.models.dskd import DSKDModel
from src.anomaly_detection.training.checkpoint import CheckpointManager


class DSKDTrainer:
    """Manages the dual-optimizer training procedure of DSKD."""

    def __init__(
        self,
        model: DSKDModel,
        loss_fn: DSKDLoss,
        learning_rate: float = 0.001,       # [PAPER FACT]: lr = 0.001
        betas: Tuple[float, float] = (0.5, 0.999),  # [PAPER FACT]: beta = (0.5, 0.999)
        weight_decay: float = 0.0,
        device: Optional[torch.device] = None,
        checkpoint_dir: str | Path = "checkpoints",
        clip_grad_norm: float = 1.0,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.clip_grad_norm = clip_grad_norm

        # Separate optimizers according to Algorithm 1, lines 16-17
        self.optimizer_e = torch.optim.Adam(
            self.model.get_encoder_params(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )

        self.optimizer_d = torch.optim.Adam(
            self.model.get_decoder_system_params(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )

        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss_e": [],
            "train_loss_d": [],
            "train_loss_total": [],
            "val_loss_total": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Runs one full training epoch across the dataset."""
        self.model.train()
        self.model.teacher.eval()  # Teacher must always remain frozen in eval mode

        total_loss_e = 0.0
        total_loss_d = 0.0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            # Support both tensor and dict batch returns
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            elif isinstance(batch, dict):
                images = batch["image"]
            else:
                images = batch

            images = images.to(self.device)

            self.optimizer_e.zero_grad()
            self.optimizer_d.zero_grad()

            # Forward pass
            f_t, f_e, f_d = self.model(images)

            # Compute losses
            loss_e, loss_d, _, _ = self.loss_fn(f_t, f_e, f_d)

            # Backward pass
            if self.model.detach_dfe_input:
                # Independent gradient graphs
                loss_e.backward()
                loss_d.backward()
            else:
                # Joint graph: retain graph on first backward
                loss_e.backward(retain_graph=True)
                loss_d.backward()

            # Optional gradient clipping
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.get_encoder_params(), self.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model.get_decoder_system_params(), self.clip_grad_norm)

            self.optimizer_e.step()
            self.optimizer_d.step()

            total_loss_e += loss_e.item()
            total_loss_d += loss_d.item()

        avg_loss_e = total_loss_e / max(1, num_batches)
        avg_loss_d = total_loss_d / max(1, num_batches)

        return {
            "loss_e": avg_loss_e,
            "loss_d": avg_loss_d,
            "loss_total": avg_loss_e + avg_loss_d,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluates loss convergence on validation images."""
        self.model.eval()

        total_loss_e = 0.0
        total_loss_d = 0.0
        num_batches = len(val_loader)

        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            elif isinstance(batch, dict):
                images = batch["image"]
            else:
                images = batch

            images = images.to(self.device)

            f_t, f_e, f_d = self.model(images)
            loss_e, loss_d, _, _ = self.loss_fn(f_t, f_e, f_d)

            total_loss_e += loss_e.item()
            total_loss_d += loss_d.item()

        avg_loss_e = total_loss_e / max(1, num_batches)
        avg_loss_d = total_loss_d / max(1, num_batches)

        return {
            "val_loss_e": avg_loss_e,
            "val_loss_d": avg_loss_d,
            "val_loss_total": avg_loss_e + avg_loss_d,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 200,
        val_frequency: int = 5,
        save_best: bool = True,
        on_epoch_end: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> Dict[str, Any]:
        """Executes the complete multi-epoch training schedule."""
        best_val_loss = float("inf")
        start_time = time.time()

        print(f"Starting DSKD Training on {self.device} for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            train_metrics = self.train_epoch(train_loader)
            epoch_duration = time.time() - epoch_start

            val_metrics = {}
            if val_loader is not None and (epoch % val_frequency == 0 or epoch == epochs):
                val_metrics = self.validate(val_loader)

            val_total = val_metrics.get("val_loss_total", train_metrics["loss_total"])

            # Track history
            self.history["epoch"].append(epoch)
            self.history["train_loss_e"].append(train_metrics["loss_e"])
            self.history["train_loss_d"].append(train_metrics["loss_d"])
            self.history["train_loss_total"].append(train_metrics["loss_total"])
            self.history["val_loss_total"].append(val_total)

            is_best = val_total < best_val_loss
            if is_best:
                best_val_loss = val_total

            # Save latest checkpoint
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_e_state_dict": self.optimizer_e.state_dict(),
                "optimizer_d_state_dict": self.optimizer_d.state_dict(),
                "best_val_loss": best_val_loss,
                "history": self.history,
            }

            self.checkpoint_manager.save(
                checkpoint_state,
                is_best=(is_best and save_best),
            )

            # Console output
            val_str = f" | Val: {val_total:.5f}" if val_metrics else ""
            best_mark = " [*Best*]" if is_best and val_metrics else ""
            print(
                f"Epoch [{epoch:3d}/{epochs:3d}] ({epoch_duration:.2f}s) "
                f"Loss_E: {train_metrics['loss_e']:.5f} | "
                f"Loss_D: {train_metrics['loss_d']:.5f} | "
                f"Total: {train_metrics['loss_total']:.5f}"
                f"{val_str}{best_mark}"
            )

            if on_epoch_end is not None:
                combined_metrics = {**train_metrics, **val_metrics}
                on_epoch_end(epoch, combined_metrics)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time / 60:.2f} minutes. Best Val Loss: {best_val_loss:.5f}")

        return {
            "total_time_seconds": total_time,
            "best_val_loss": best_val_loss,
            "history": self.history,
        }
