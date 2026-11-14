"""Training loop and checkpoint management utilities."""

from .trainer import CDATrainer, load_checkpoint, save_checkpoint

__all__ = ["save_checkpoint", "load_checkpoint", "CDATrainer"]
