"""Spatial normalization for vision encoder features."""
import torch


ALL_SPNORM_METHODS = ["none", "zscore"]


def spatial_zscore(feat: torch.Tensor, alpha: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Z-score normalization along spatial dimension.

    Args:
        feat: (B, T, D) patch tokens
        alpha: scaling factor for mean subtraction (default 1.0)
        eps: small constant for numerical stability

    Returns:
        Normalized features (B, T, D)
    """
    mean = feat.mean(dim=1, keepdim=True)
    std = feat.std(dim=1, keepdim=True)
    return (feat - alpha * mean) / (std + eps)


class SpatialNormalization:
    """
    Spatial normalization wrapper for backward compatibility.

    Only supports "none" and "zscore" methods.
    """
    def __init__(self, method: str, *, eps: float = 1e-6):
        assert method in ALL_SPNORM_METHODS, f"Invalid method: {method}. Must be one of {ALL_SPNORM_METHODS}"
        self.method = method
        self.eps = eps

    def __call__(self, feat: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.method == "none":
            return feat
        alpha = kwargs.get('zscore_alpha', 1.0)
        return spatial_zscore(feat, alpha=alpha, eps=self.eps)
