"""Distillation loss functions for CDA."""

from .cda_loss import CDALoss, cosine_distance_map

__all__ = ["cosine_distance_map", "CDALoss"]
