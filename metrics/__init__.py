"""
Spatial Metrics Module for Vision Encoder Evaluation

This module provides metrics for evaluating the spatial structure
of vision encoder feature representations.

Available metrics:
- LDS: Local-vs-Distant Similarity
- CDS: Correlation-Decay Slope
- SRSS: Semantic-Region Self-Similarity
- RMSC: RMS Spatial Contrast
"""

from .spatial_metrics import (
    compute_spatial_metrics,
    metric_lds,
    metric_srss,
    metric_cds,
    metric_rmsc,
    METRICS_REGISTRY,
)

__all__ = [
    'compute_spatial_metrics',
    'metric_lds',
    'metric_srss',
    'metric_cds',
    'metric_rmsc',
    'METRICS_REGISTRY',
]
