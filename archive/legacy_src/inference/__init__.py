"""Inference-stage anomaly map generation and scoring."""

from .anomaly_mapper import CDAAnomalyMapper, predict_batch

__all__ = ["CDAAnomalyMapper", "predict_batch"]
