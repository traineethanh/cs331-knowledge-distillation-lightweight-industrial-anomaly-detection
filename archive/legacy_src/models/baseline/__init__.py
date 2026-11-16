"""Baseline anomaly detection models: STFPM (Foundational) and CDA (Modern Comparator)."""

from .cda import (
    ConvBlock,
    DepthwiseMLP,
    GCM,
    GCMAndDecoder,
    GroupedLinear,
    GSU,
    GSUDecoder,
    GSULayer,
    PatchExpand2D,
    RMSNorm,
    TeacherEncoder,
)
from .sga import DoubleConv, RoPESelfAttention, SGAFusion
from .stfpm import (
    ResNet18FeatureExtractor,
    STFPM,
    STFPMLoss,
    compute_stfpm_anomaly_map,
)

__all__ = [
    "RMSNorm",
    "GroupedLinear",
    "DepthwiseMLP",
    "GSU",
    "GSULayer",
    "ConvBlock",
    "PatchExpand2D",
    "TeacherEncoder",
    "RoPESelfAttention",
    "DoubleConv",
    "SGAFusion",
    "GCM",
    "GSUDecoder",
    "GCMAndDecoder",
    "ResNet18FeatureExtractor",
    "STFPM",
    "STFPMLoss",
    "compute_stfpm_anomaly_map",
]
