"""
Spatial metrics for evaluating vision encoder feature representations.

This module provides 4 key metrics:
- LDS (Local-vs-Distant Similarity)
- CDS (Correlation-Decay Slope)
- SRSS (Semantic-Region Self-Similarity)
- RMSC (RMS Spatial Contrast)
"""

import math
from typing import Dict, List, Callable, Optional, Union
import torch
import torch.nn.functional as F


# ---------------- Registry ----------------
METRICS_REGISTRY: Dict[str, Callable] = {}


def register_metric(name: Optional[str] = None):
    """Decorator to register a metric function."""
    def deco(fn: Callable):
        METRICS_REGISTRY[name or fn.__name__] = fn
        return fn
    return deco


# --------------- Grid cache ---------------
class _GridCache:
    """Cache for grid coordinates and Manhattan distances."""

    def __init__(self):
        self._dist = {}   # key: (H,W,device) -> (T,T) manhattan dist
        self._coords = {} # key: (H,W,device) -> (T,2)

    def coords(self, H: int, W: int, device: torch.device):
        key = (H, W, str(device))
        if key not in self._coords:
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            self._coords[key] = torch.stack([yy.flatten(), xx.flatten()], dim=1)  # (T,2)
        return self._coords[key]

    def dist(self, H: int, W: int, device: torch.device):
        key = (H, W, str(device))
        if key not in self._dist:
            c = self.coords(H, W, device)  # (T,2)
            self._dist[key] = (c[:, None, :] - c[None, :, :]).abs().sum(-1)  # (T,T) manhattan
        return self._dist[key]


_GRID = _GridCache()


# --------------- Helpers ------------------
def _check_grid_from_T(T: int) -> tuple:
    """Check that T is a perfect square and return (H, W)."""
    H = int(math.isqrt(T))
    if H * H != T:
        raise ValueError(f"T must be a square (got {T}). If not square, adapt reshape logic.")
    return H, H


def _normalize(u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2 normalize along last dimension."""
    return F.normalize(u.to(torch.float32), dim=-1, eps=eps)


def _reshape_hw(u: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Reshape (B,T,D) -> (B,H,W,D)."""
    return u.view(u.size(0), H, W, u.size(-1))


def _gram_cos(u: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity Gram matrix. u: (B,T,D) already normalized -> (B,T,T)."""
    return torch.einsum("btd,bsd->bts", u, u)


# --------------- Metrics ------------------

@register_metric("lds")
@torch.no_grad()
def metric_lds(u: torch.Tensor, H: int, W: int, far_dist: int = 6, **_) -> torch.Tensor:
    """
    Local-vs-Distant Similarity (LDS):
    Mean cosine over 4-neighbors minus mean cosine over pairs with Manhattan distance >= far_dist.

    Args:
        u: Features tensor of shape (B, T, D)
        H, W: Height and width of spatial grid
        far_dist: Minimum Manhattan distance for "far" pairs (default=6)

    Returns:
        (B,) tensor of LDS scores per image
    """
    B, T, D = u.shape
    u = _normalize(u)
    uHW = _reshape_hw(u, H, W)  # (B,H,W,D)

    # local (4-neighbors) via shifts
    up = (uHW[:, 1:, :, :] * uHW[:, :-1, :, :]).sum(-1)    # (B,H-1,W)
    left = (uHW[:, :, 1:, :] * uHW[:, :, :-1, :]).sum(-1)  # (B,H,W-1)
    local = torch.cat([up.flatten(1), left.flatten(1)], dim=1).mean(dim=1)  # (B,)

    # far
    G = _gram_cos(u)  # (B,T,T)
    dist = _GRID.dist(H, W, u.device)
    far_mask = (dist >= far_dist) & (dist > 0)
    denom = far_mask.sum().clamp(min=1)
    far = (G.masked_fill(~far_mask, 0.0).sum(dim=(1, 2)) / denom)  # (B,)

    return local - far


@register_metric("srss")
@torch.no_grad()
def metric_srss(
    u: torch.Tensor,               # (B,T,D)
    H: int,
    W: int,
    *,
    masks: torch.Tensor,           # (B,T) boolean/int {1=FG, 0=BG}
    d_pos: int = 1,                # near threshold (inclusive): 1..d_pos
    d_pos_min: int = 1,
    d_neg: int = 6,                # far threshold (inclusive): >= d_neg
    **_,
) -> torch.Tensor:
    """
    Mask-aware LDS (SRSS - Spatial Ratio of Same-Scale Similarities):
    For each anchor token inside the mask (FG), compare mean cosine to
    nearby FG tokens (<= d_pos) vs. far BG tokens (>= d_neg).

    Args:
        u: Features tensor of shape (B, T, D)
        H, W: Height and width of spatial grid
        masks: Boolean mask tensor (B, T) where 1=foreground, 0=background
        d_pos: Maximum distance for "near" FG neighbors (default=1)
        d_pos_min: Minimum distance for "near" neighbors (default=1, excludes self)
        d_neg: Minimum distance for "far" BG neighbors (default=6)

    Returns:
        (B,) tensor of SRSS scores per image. NaN for images with no valid pairs.
    """
    if masks is None:
        raise ValueError("metric_lds_mask requires `masks` (B,T) tensor.")
    if masks.ndim != 2 or masks.shape[0] != u.shape[0] or masks.shape[1] != u.shape[1]:
        raise ValueError(f"`masks` must be shape (B,T) matching feats, got {tuple(masks.shape)} vs {(u.shape[0], u.shape[1])}")

    B, T, D = u.shape
    u = F.normalize(u.to(torch.float32), dim=-1)
    G = u @ u.transpose(1, 2)  # (B,T,T) cosine
    dist = _GRID.dist(H, W, u.device)  # (T,T)

    # Precompute distance windows once
    pos_dist_mask = (dist <= d_pos) & (dist >= d_pos_min)  # exclude self
    neg_dist_mask = (dist >= d_neg)

    out_vals = []
    for b in range(B):
        fg = masks[b].to(torch.bool)  # (T,)
        if fg.sum() < 2 or (~fg).sum() < 1:
            out_vals.append(torch.tensor(float('nan'), device=u.device))
            continue

        # Row-wise (anchor-wise) masks
        # positives: anchor in FG, neighbor in FG, and near
        pos_mask_b = (fg[:, None] & fg[None, :]) & pos_dist_mask  # (T,T)
        # negatives: anchor in FG, neighbor in BG, and far
        neg_mask_b = (fg[:, None] & (~fg)[None, :]) & neg_dist_mask

        # Row-wise counts & sums
        Gb = G[b]  # (T,T)
        pos_counts = pos_mask_b.sum(dim=1)  # (T,)
        neg_counts = neg_mask_b.sum(dim=1)  # (T,)

        # Elementwise mask multiply -> row sums
        pos_sums = (Gb * pos_mask_b).sum(dim=1)  # (T,)
        neg_sums = (Gb * neg_mask_b).sum(dim=1)  # (T,)

        # Row means (avoid /0 with clamp; we'll filter invalid anyway)
        pos_means = pos_sums / pos_counts.clamp_min(1)
        neg_means = neg_sums / neg_counts.clamp_min(1)

        # Valid anchors: inside FG and have both pos & neg neighbors
        valid = fg & (pos_counts > 0) & (neg_counts > 0)
        if valid.any():
            delta = (pos_means[valid] - neg_means[valid]).mean()
            out_vals.append(delta)
        else:
            out_vals.append(torch.tensor(float('nan'), device=u.device))

    return torch.stack(out_vals, dim=0)  # (B,)


@register_metric("cds")
@torch.no_grad()
def metric_cds(u: torch.Tensor, H: int, W: int, dmax: int = 8, **_) -> torch.Tensor:
    """
    Correlation-Decay Slope (CDS):
    Fit a line to mean cosine vs Manhattan distance d=1..dmax; return -slope.
    Higher values indicate stronger spatial structure (correlation decays with distance).

    Args:
        u: Features tensor of shape (B, T, D)
        H, W: Height and width of spatial grid
        dmax: Maximum distance for fitting (default=8)

    Returns:
        (B,) tensor of CDS scores per image
    """
    B, T, D = u.shape
    u = _normalize(u)
    G = _gram_cos(u)  # (B,T,T)
    dist = _GRID.dist(H, W, u.device)

    sims = []
    ds = []
    for d in range(1, dmax + 1):
        m = (dist == d)
        if m.any():
            sims.append(G[:, m].mean(dim=1))  # (B,)
            ds.append(d)
    if len(sims) == 0:
        return torch.zeros(B, device=u.device)

    S = torch.stack(sims, dim=1)  # (B,K)
    x = torch.tensor(ds, device=u.device, dtype=S.dtype)  # (K,)
    x0 = x - x.mean()
    denom = (x0 @ x0).clamp(min=1e-12)
    # slope per image
    b = ((S - S.mean(dim=1, keepdim=True)) @ x0) / denom  # (B,)
    return -b


@register_metric("rmsc")
@torch.no_grad()
def metric_rmsc(u, H, W, *, sqrt: bool = True, eps: float = 1e-8, **_):
    """
    RMS Token Contrast (RMSC):
    Measures spread/diversity of token representations.
    Computes RMS of centered normalized tokens.

    Args:
        u: Features tensor of shape (B, T, D)
        H, W: Height and width of spatial grid (unused but kept for interface consistency)
        sqrt: Whether to take sqrt of mean squared norm (default=True)
        eps: Small epsilon for numerical stability

    Returns:
        (B,) tensor of RMSC scores per image
    """
    x = F.normalize(u.to(torch.float32), dim=-1, eps=eps)  # L2 over D (per token)
    xc = x - x.mean(dim=1, keepdim=True)                   # mean-center over tokens
    ms = xc.square().sum(dim=-1).mean(dim=1)               # mean ||.||^2 across tokens
    return ms.sqrt() if sqrt else ms                        # (B,)


# --------------- Runner -------------------

@torch.no_grad()
def compute_spatial_metrics(
    feats_: Union[Dict[str, torch.Tensor], torch.Tensor],
    masks_: Optional[torch.Tensor] = None,   # (B,T) for mask-aware metrics
    metrics: Optional[List[str]] = None,
    *,
    H: Optional[int] = None,
    W: Optional[int] = None,
    device: Optional[torch.device] = None,
    metric_kwargs: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Compute spatial structure metrics for encoder features.

    Parameters
    ----------
    feats_ : Dict[str, Tensor] or Tensor
        If dict: {encoder_name: Tensor[B, T, D]}.
        If Tensor: a single Tensor[B, T, D] (wrapped under key "_unnamed").
        T must correspond to a square grid (e.g., 256 -> 16x16), unless H,W are provided.
    masks_ : Tensor[B, T], optional
        Foreground/background mask for mask-aware metrics (e.g., lds_mask/SRSS).
    metrics : List[str], optional
        Names of metrics to compute. Defaults to ['lds', 'cds', 'lds_mask', 'rms_token'].
    H, W : int, optional
        Grid height/width. If omitted, inferred from T as sqrt(T) x sqrt(T).
    device : torch.device, optional
        Device to run metrics on. Defaults to device of the first tensor.
    metric_kwargs : Dict[str, Dict], optional
        Per-metric keyword arguments, e.g. {"cds": {"dmax": 8}}.

    Returns
    -------
    results : Dict[str, Dict[str, Dict[str, Tensor]]]
        {
          encoder_name: {
            metric_name: {
              "per_image": Tensor[B],
              "mean":      Tensor[()],  # scalar
            }, ...
          }, ...
        }
    """
    # ---- validate / normalize feats_ into a dict ----
    if feats_ is None:
        raise ValueError("feats_ is None.")

    if isinstance(feats_, torch.Tensor):
        feats_dict: Dict[str, torch.Tensor] = {"_unnamed": feats_}
    elif isinstance(feats_, dict):
        feats_dict = feats_
    else:
        raise TypeError(f"feats_ must be a Dict[str, Tensor] or a Tensor; got {type(feats_)}")

    if len(feats_dict) == 0:
        raise ValueError("feats_ is empty.")

    first = next(iter(feats_dict.values()))
    if not isinstance(first, torch.Tensor):
        raise TypeError("feats_[name] must be a Tensor.")
    if first.ndim != 3:
        raise ValueError(f"Expected feats_[name] of shape (B,T,D); got {tuple(first.shape)}")

    B, T, _ = first.shape

    # ---- infer grid if not provided ----
    if H is None or W is None:
        H = int(math.isqrt(T))
        if H * H != T:
            raise ValueError(f"T must be a square if H,W not provided (got T={T}).")
        W = H

    # Default metrics
    if metrics is None:
        metrics = ['lds', 'cds', 'srss', 'rmsc']

    metric_kwargs = metric_kwargs or {}
    device = device or first.device

    # ---- move tensors & sanity check shapes ----
    feats_norm: Dict[str, torch.Tensor] = {}
    for name, z in feats_dict.items():
        if not isinstance(z, torch.Tensor) or z.ndim != 3:
            raise ValueError(f"{name}: expected Tensor (B,T,D); got {type(z)} with shape {getattr(z,'shape',None)}")
        if z.shape[0] != B or z.shape[1] != T:
            raise ValueError(
                f"{name}: inconsistent (B,T)=({z.shape[0]},{z.shape[1]}) vs reference ({B},{T})"
            )
        feats_norm[name] = z.to(device)

    # Move mask to device if provided
    if masks_ is not None:
        if not isinstance(masks_, torch.Tensor) or masks_.ndim != 2:
            raise ValueError(f"masks_ must be Tensor of shape (B,T); got {getattr(masks_, 'shape', None)}")
        masks_ = masks_.to(device)

    # ---- compute metrics ----
    results: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    for enc_name, z in feats_norm.items():
        enc_res: Dict[str, Dict[str, torch.Tensor]] = {}
        for mname in metrics:
            if mname not in METRICS_REGISTRY:
                raise KeyError(f"Metric '{mname}' is not registered.")
            fn = METRICS_REGISTRY[mname]
            kw = metric_kwargs.get(mname, {})
            per_img = fn(z, H=H, W=W, masks=masks_, **kw)  # expected shape (B,)
            if not isinstance(per_img, torch.Tensor) or per_img.ndim != 1 or per_img.shape[0] != B:
                raise RuntimeError(
                    f"Metric '{mname}' must return Tensor of shape (B,), got {type(per_img)} with shape {getattr(per_img,'shape',None)}"
                )
            enc_res[mname] = {
                "per_image": per_img,
                "mean": torch.nanmean(per_img) if torch.isnan(per_img).any() else per_img.mean(),
            }
        results[enc_name] = enc_res

    return results
