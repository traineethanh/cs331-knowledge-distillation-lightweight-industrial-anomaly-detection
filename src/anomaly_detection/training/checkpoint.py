"""Atomic checkpoint management for training persistence and resumption."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """Manages atomic saving, loading, and tracking of model checkpoints."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        state: Dict[str, Any],
        is_best: bool = False,
        filename: str = "latest_checkpoint.pth",
    ) -> Path:
        """Atomically saves checkpoint to avoid partial writes upon interruption."""
        target_path = self.checkpoint_dir / filename
        tmp_path = self.checkpoint_dir / f"{filename}.tmp"

        torch.save(state, tmp_path)
        os.replace(tmp_path, target_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pth"
            best_tmp = self.checkpoint_dir / "best_checkpoint.pth.tmp"
            torch.save(state, best_tmp)
            os.replace(best_tmp, best_path)

        return target_path

    def load(
        self,
        checkpoint_path: str | Path,
        map_location: Optional[str | torch.device] = None,
    ) -> Dict[str, Any]:
        """Loads checkpoint from disk."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        return torch.load(path, map_location=map_location, weights_only=False)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Returns path to latest_checkpoint.pth if it exists."""
        candidate = self.checkpoint_dir / "latest_checkpoint.pth"
        return candidate if candidate.exists() else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Returns path to best_checkpoint.pth if it exists."""
        candidate = self.checkpoint_dir / "best_checkpoint.pth"
        return candidate if candidate.exists() else None
