#!/usr/bin/env python3
"""
Spatial Metrics vs FID Correlation Plots

Generates correlation plots comparing spatial metrics (LDS, CDS, SRSS, RMSC)
against FID scores across multiple vision encoders.

Usage:
    python spatial_metrics_comparison.py --model_type sit-xl-2
    python spatial_metrics_comparison.py --model_type sit-xl-2 --steps 100000
"""

import os
import json
import argparse
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Optional

from spatial_metrics import compute_spatial_metrics

# ============== Matplotlib Config ==============
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300

# Font sizes
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 11
CORR_FONTSIZE = 20

# Scatter plot styling
SCATTER_SIZE = 80
SCATTER_ALPHA = 0.7
SCATTER_EDGEWIDTH = 1
LINE_STYLE = '--'
LINE_WIDTH = 2
LINE_ALPHA = 0.7
GRID_ALPHA = 0.3

# ============== Encoder Names ==============
ENCODER_NAMES = [
    "dinov2-vit-b", "dinov2-vit-l", "dinov2-vit-g",
    "dinov3-vit-b16", "dinov3-vit-l16", "dinov3-vit-h16plus", "dinov3-vit-7b16",
    "webssl-vit-dino300m_full2b_224", "webssl-vit-dino1b_full2b_224",
    "webssl-vit-dino2b_full2b_224", "webssl-vit-dino3b_full2b_224",
    "pe-vit-b", "pe-vit-l", "pe-vit-g",
    "langpe-vit-l", "langpe-vit-g",
    "spatialpe-vit-b", "spatialpe-vit-l", "spatialpe-vit-g",
    "cradio-vit-b", "cradio-vit-l",
    "dino-vit-b", "clip-vit-L",
    "mocov3-vit-b", "mocov3-vit-l",
    "jepa-vit-h", "mae-vit-l",
]


def get_eval_names_xl() -> Dict[str, str]:
    """Hardcoded eval names for sit-xl-2 model."""
    return {
        "dinov2-vit-b": "sit-xl-2-sdvae-enc8-dinov2-b_cfg1.0-seed0-modesde-steps250.csv",
        "dinov2-vit-l": "sit-xl-2-sdvae-enc8-dinov2-l_cfg1.0-seed0-modesde-steps250.csv",
        "dinov2-vit-g": "sit-xl-2-sdvae-enc8-dinov2-g_cfg1.0-seed0-modesde-steps250.csv",
        "dinov3-vit-b16": "sit-xl-2-sdvae-dinov3-b16-enc8-cosine0.5_cfg1.0-seed0-modesde-steps250.csv",
        "dinov3-vit-l16": "sit-xl-2-sdvae-enc8-dinov3-l16_cfg1.0-seed0-modesde-steps250.csv",
        "dinov3-vit-h16plus": "sit-xl-2-sdvae-enc8-dinov3-h16plus_cfg1.0-seed0-modesde-steps250.csv",
        "dinov3-vit-7b16": "sit-xl-2-sdvae-enc8-dinov3-7b16_cfg1.0-seed0-modesde-steps250.csv",
        "webssl-vit-dino300m_full2b_224": "sit-xl-2-sdvae-enc8-webssl300m224_cfg1.0-seed0-modesde-steps250.csv",
        "webssl-vit-dino1b_full2b_224": "sit-xl-2-sdvae-enc8-webssl1b224_cfg1.0-seed0-modesde-steps250.csv",
        "webssl-vit-dino2b_full2b_224": "sit-xl-2-sdvae-enc8-webssl2b224_cfg1.0-seed0-modesde-steps250.csv",
        "webssl-vit-dino3b_full2b_224": "sit-xl-2-sdvae-enc8-webssl3b224_cfg1.0-seed0-modesde-steps250.csv",
        "pe-vit-b": "sit-xl-2-sdvae-enc8-pe-b_cfg1.0-seed0-modesde-steps250.csv",
        "pe-vit-l": "sit-xl-2-sdvae-enc8-pe-l_cfg1.0-seed0-modesde-steps250.csv",
        "pe-vit-g": "sit-xl-2-sdvae-enc8-pe-g_cfg1.0-seed0-modesde-steps250.csv",
        "langpe-vit-l": "sit-xl-2-sdvae-enc8-langpe-l_cfg1.0-seed0-modesde-steps250.csv",
        "langpe-vit-g": "sit-xl-2-sdvae-enc8-langpe-g_cfg1.0-seed0-modesde-steps250.csv",
        "spatialpe-vit-b": "sit-xl-2-sdvae-enc8-spatialpe-b_cfg1.0-seed0-modesde-steps250.csv",
        "spatialpe-vit-l": "sit-xl-2-sdvae-enc8-langpe-l_cfg1.0-seed0-modesde-steps250.csv",
        "spatialpe-vit-g": "sit-xl-2-sdvae-enc8-spatialpe-g_cfg1.0-seed0-modesde-steps250.csv",
        "cradio-vit-b": "sit-xl-2-sdvae-cradio-b-enc8-coeff0.5_cfg1.0-seed0-modesde-steps250.csv",
        "cradio-vit-l": "sit-xl-2-sdvae-cradio-l-enc8-cosine0.5_cfg1.0-seed0-modesde-steps250.csv",
        "dino-vit-b": "sit-xl-2-sdvae-dino-b-enc8-cosine0.5_cfg1.0-seed0-modesde-steps250.csv",
        "clip-vit-L": "sit-xl-2-sdvae-clip-l-enc8-cosine0.5_cfg1.0-seed0-modesde-steps250.csv",
        "mocov3-vit-b": "sit-xl-2-sdvae-mocov3-b-enc8-cosine0.5_cfg1.0-seed0-modesde-steps250.csv",
        "mocov3-vit-l": "sit-xl-2-sdvae-mocov3-l-enc8-cosine0.5_cfg1.0-seed0-modesde-steps250.csv",
        "jepa-vit-h": "sit-xl-2-sdvae-jepa-h-enc8-cosine0.5_cfg1.0-seed0-modesde-steps250.csv",
        "mae-vit-l": "sit-xl-2-sdvae-mae-l-enc8-cosine0.5_cfg1.0-seed0-modesde-steps250.csv",
    }


def get_eval_names_auto(model_type: str, encoder_names: List[str]) -> Dict[str, str]:
    """Auto-generate eval names for non-xl models."""
    eval_names = {}
    for enc in encoder_names:
        eval_names[enc] = (
            f"{model_type}-sdvae-{enc}-enc8-coeff0.5-loss-cosine-"
            f"projlayer-mlp-100000-v1_cfg1.0-seed0-modesde-steps250.csv"
        )
    return eval_names


def load_fid_scores(
    eval_names: Dict[str, str],
    steps: int,
    eval_dir: str = "./data/stats/fid"
) -> Dict[str, float]:
    """Load FID scores from CSV files."""
    fid_scores = {}
    for model_name, eval_file in eval_names.items():
        try:
            file_path = os.path.join(eval_dir, eval_file)
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            df_step = df[df["Steps"] == steps]
            if len(df_step) == 1:
                fid_scores[model_name] = float(df_step["FID"].values[0])
        except Exception:
            pass
    print(f"Loaded FID scores for {len(fid_scores)} encoders")
    return fid_scores


def load_lp_accuracies(
    encoder_names: List[str],
    lp_dir: str = "./data/stats/lp"
) -> Dict[str, float]:
    """Load Linear Probing accuracies from JSON files."""
    lp_acc = {}
    for enc in encoder_names:
        try:
            exp_path = os.path.join(lp_dir, f"{enc}-none-mean", "metrics.json")
            if not os.path.exists(exp_path):
                continue
            with open(exp_path, 'r') as f:
                acc = json.load(f)['best_val_acc']
            lp_acc[enc] = float(acc)
        except Exception:
            pass
    print(f"Loaded LP accuracies for {len(lp_acc)} encoders")
    return lp_acc


def load_spatial_features(
    encoder_names: List[str],
    feats_dir: str = "./data/features"
) -> Dict[str, torch.Tensor]:
    """Load pre-extracted spatial features."""
    feats = {}
    for enc in encoder_names:
        try:
            feat_file = os.path.join(feats_dir, f"{enc}.pt")
            if os.path.exists(feat_file):
                feats[enc] = torch.load(feat_file, map_location='cpu')
        except Exception:
            pass
    print(f"Loaded features for {len(feats)} encoders")
    return feats


def load_and_filter_masks(
    masks_dir: str = "./data/masks",
    target_resolution: int = 256,
    num_target_count: int = 64,
) -> torch.Tensor:
    """Load and filter masks based on minimum patch counts."""
    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {masks_dir}")

    masks = [Image.open(m).resize((target_resolution, target_resolution)) for m in mask_files]
    masks_tensor = torch.from_numpy(np.stack(masks))

    # Filter based on patch counts
    bin_masks = (masks_tensor > 0).unsqueeze(1).float()
    patch_pos = F.max_pool2d(bin_masks, kernel_size=16, stride=16)
    pos_counts = patch_pos.sum(dim=(1, 2, 3)).to(torch.int64)
    neg_counts = (1.0 - patch_pos).sum(dim=(1, 2, 3)).to(torch.int64)
    keep = (pos_counts >= num_target_count) & (neg_counts >= num_target_count)

    masks_tensor = masks_tensor[keep.squeeze(0) if keep.ndim > 1 else keep]
    print(f"Kept {keep.sum().item()} / {len(keep)} mask samples")

    # Downsample to 16x16 = 256 patches
    masks_16x16 = F.max_pool2d(
        masks_tensor.unsqueeze(1).float(),
        kernel_size=16, stride=16
    ).squeeze(1)
    masks_16x16 = masks_16x16.to(bool).view(masks_tensor.shape[0], -1)

    return masks_16x16


def plot_correlation(ax, x_vals, y_vals, xlabel, ylabel,
                     marker_color='#3498db', line_color='#e74c3c'):
    """Create a correlation scatter plot."""
    ax.scatter(x_vals, y_vals, s=SCATTER_SIZE, alpha=SCATTER_ALPHA, c=marker_color,
               edgecolors='black', linewidth=SCATTER_EDGEWIDTH)

    if len(x_vals) > 2:
        # Fit line
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x_vals), max(x_vals), 100)
        ax.plot(x_line, p(x_line), LINE_STYLE, color=line_color,
                alpha=LINE_ALPHA, linewidth=LINE_WIDTH)

        # Pearson correlation
        correlation = np.corrcoef(x_vals, y_vals)[0, 1]
        ax.text(0.95, 0.95, f'Pearson Corr.\nr = {correlation:.3f}',
                transform=ax.transAxes, fontsize=CORR_FONTSIZE,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.9))

    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_spatial_metrics_comparison(
    model_type: str = "sit-xl-2",
    steps: int = 100000,
    output_path: Optional[str] = None,
    data_dir: str = "./data",
    device: str = "cuda",
):
    """Create correlation plots comparing spatial metrics vs FID."""
    print(f"\nGenerating spatial metrics comparison for {model_type} at {steps//1000}K steps")
    print("=" * 60)

    # Set device
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Get eval names
    if model_type == "sit-xl-2":
        eval_names = get_eval_names_xl()
    else:
        eval_names = get_eval_names_auto(model_type, ENCODER_NAMES)

    # Find available encoders
    available_encoders = []
    fid_dir = os.path.join(data_dir, "stats/fid")
    for enc in ENCODER_NAMES:
        if enc in eval_names:
            eval_file = os.path.join(fid_dir, eval_names[enc])
            if os.path.exists(eval_file):
                available_encoders.append(enc)
    print(f"Found {len(available_encoders)} available encoders")

    # Load data
    print("\nLoading data...")
    fid_scores = load_fid_scores(eval_names, steps, os.path.join(data_dir, "stats/fid"))
    lp_acc = load_lp_accuracies(available_encoders, os.path.join(data_dir, "stats/lp"))
    feats = load_spatial_features(available_encoders, os.path.join(data_dir, "features"))
    masks_16x16 = load_and_filter_masks(os.path.join(data_dir, "masks"))

    # Match feature batch size to masks
    n_masks = masks_16x16.shape[0]
    for enc in list(feats.keys()):
        if feats[enc].shape[0] > n_masks:
            feats[enc] = feats[enc][:n_masks]
        elif feats[enc].shape[0] < n_masks:
            masks_16x16 = masks_16x16[:feats[enc].shape[0]]
            n_masks = masks_16x16.shape[0]

    # Compute spatial metrics
    if feats:
        print(f"\nComputing spatial metrics on {device}...")
        spatial_metrics = compute_spatial_metrics(
            feats, masks_16x16,
            metrics=['lds', 'srss', 'cds', 'rmsc'],
            metric_kwargs={
                "lds": {"far_dist": 6},
                "srss": {"d_pos": 1, "d_neg": 6},
                "cds": {"dmax": 8},
                "rmsc": {"sqrt": True}
            },
            device=device,
        )
        print(f"Computed metrics for {len(spatial_metrics)} encoders")
    else:
        print("Warning: No features loaded")
        return

    # Extract data for plotting
    def get_metric_vs_fid(metric_name):
        x_vals, y_vals = [], []
        for enc in available_encoders:
            if enc in spatial_metrics and enc in fid_scores:
                if metric_name in spatial_metrics[enc]:
                    val = spatial_metrics[enc][metric_name]["mean"]
                    if torch.is_tensor(val):
                        val = val.item()
                    if not np.isnan(val):
                        x_vals.append(val)
                        y_vals.append(fid_scores[enc])
        return x_vals, y_vals

    # Linear Probing vs FID
    lp_x, lp_y = [], []
    for enc in available_encoders:
        if enc in lp_acc and enc in fid_scores:
            lp_x.append(lp_acc[enc])
            lp_y.append(fid_scores[enc])

    lds_x, lds_y = get_metric_vs_fid('lds')
    cds_x, cds_y = get_metric_vs_fid('cds')
    srss_x, srss_y = get_metric_vs_fid('srss')
    rmsc_x, rmsc_y = get_metric_vs_fid('rmsc')

    print(f"\nData points: LP={len(lp_x)}, LDS={len(lds_x)}, CDS={len(cds_x)}, SRSS={len(srss_x)}, RMSC={len(rmsc_x)}")

    # Create figure (1x5: LP + 4 spatial metrics)
    fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))

    fid_label = fr'$\longleftarrow$ gFID ({steps//1000}K steps)'

    # Plot 0: Linear Probing (leftmost)
    if lp_x:
        plot_correlation(axes[0], lp_x, lp_y,
                         r'Linear Probing Accuracy $\longrightarrow$', fid_label,
                         marker_color='#3498db')
    if lds_x:
        plot_correlation(axes[1], lds_x, lds_y,
                         r'Spatial Structure (LDS) $\longrightarrow$', fid_label,
                         marker_color='#9b59b6')
    if cds_x:
        plot_correlation(axes[2], cds_x, cds_y,
                         r'Spatial Structure (CDS) $\longrightarrow$', fid_label,
                         marker_color='#f39c12')
    if srss_x:
        plot_correlation(axes[3], srss_x, srss_y,
                         r'Spatial Structure (SRSS) $\longrightarrow$', fid_label,
                         marker_color='#2ecc71')
    if rmsc_x:
        plot_correlation(axes[4], rmsc_x, rmsc_y,
                         r'Spatial Structure (RMSC) $\longrightarrow$', fid_label,
                         marker_color='#e67e22')

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = f"assets/spatial_metrics_comparison-{model_type}.png"
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate spatial metrics comparison plots')
    parser.add_argument('--model_type', type=str, default='sit-xl-2',
                        choices=['sit-xl-2', 'sit-l-2', 'sit-b-2'],
                        help='Model type for FID scores')
    parser.add_argument('--steps', type=int, default=100000,
                        help='Training steps for FID evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for figure')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device for computation (default: cuda)')

    args = parser.parse_args()

    create_spatial_metrics_comparison(
        model_type=args.model_type,
        steps=args.steps,
        output_path=args.output,
        data_dir=args.data_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
