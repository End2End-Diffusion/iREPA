#!/usr/bin/env python3
"""
Visualize spatial normalization effects across different encoders.
Compares encoders with and without spatial normalization.

Usage:
    python scripts/viz_spatialnorm.py --encoders dinov3-vit-b16 pe-vit-g --device cuda
"""

import os
import sys
import math
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from PIL import Image
from typing import Dict, List, Tuple, Optional

# Add ldm directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ldm'))
from vision_encoder import load_encoders
from utils import SpatialNormalization

# ============================================
# AESTHETIC CONTROL PARAMETERS
# ============================================

# Figure layout
IMG_SIZE_PER_CELL = 2.0  # Size in inches for each image/map cell
LABEL_COL_WIDTH = 0.08   # Width ratio for encoder label column itself
LABEL_COL_SPACING = 0.02  # Gap between encoder names and first image (what you want!)
HEADER_ROW_HEIGHT = 0.05  # Height ratio for header row
ROW_SPACING = 0.1       # Vertical spacing between encoder rows
COL_SPACING = 0.005       # Spacing between the two main condition columns (w/o vs w/)
INNER_SPACING = 0.003     # Spacing between images within a condition
FIG_LEFT_MARGIN = 0.01    # Space to the left of encoder names
FIG_RIGHT_MARGIN = 0.96   # Right figure margin
FIG_TOP_MARGIN = 0.98     # Top figure margin
FIG_BOTTOM_MARGIN = 0.02  # Bottom figure margin

# Text sizes
HEADER_FONT_SIZE = 24     # Column header font size
ENCODER_LABEL_SIZE = 24   # Encoder name font size
COLORBAR_LABEL_SIZE = 20  # Colorbar label font size
COLORBAR_TICK_SIZE = 16   # Colorbar tick font size

# Colorbar settings
COLORBAR_WIDTH = 0.01     # Width of colorbar
COLORBAR_HEIGHT_RATIO = 0.6  # Height as fraction of figure
COLORBAR_X_POS = 0.97     # X position (from left)
COLORBAR_LABEL_PAD = 10   # Padding for colorbar label

# Star marker settings
STAR_SIZE = 120           # Size of star markers
STAR_COLOR = 'red'        # Color of star markers
STAR_EDGE_COLOR = 'darkred'  # Edge color of stars
STAR_LINEWIDTH = 1.2      # Line width of star edges

# Encoder name mappings
ENCODER_DISPLAY_NAMES = {
    'dinov3-vit-b16': 'DINOv3-B',
    'dinov3-vit-7b16': 'DINOv3-7B',
    'dinov3-vit-l16': 'DINOv3-L',
    'dinov3-vit-h16plus': 'DINOv3-H',
    'webssl-vit-dino1b_full2b_224': 'WebSSL-1B',
    'pe-vit-g': 'PE-G',
    'dinov2-vit-b': 'DINOv2-B',
    'dinov2-vit-l': 'DINOv2-L',
    'mae-vit-l': 'MAE-L',
    'mocov3-vit-b': 'MoCoV3-B',
    'mocov3-vit-l': 'MoCoV3-L',
    'pe-vit-b': 'PE-B',
    'pe-vit-l': 'PE-L',
    'spatialpe-vit-b': 'SpatialPE-B',
    'spatialpe-vit-l': 'SpatialPE-L',
    'spatialpe-vit-g': 'SpatialPE-G',
    'langpe-vit-l': 'LangPE-L',
    'langpe-vit-g': 'LangPE-G',
    'cradio-vit-b': 'CRADIO-B',
    'cradio-vit-l': 'CRADIO-L',
    'dino-vit-b': 'DINO-B',
    'clip-vit-L': 'CLIP-L',
    'jepa-vit-h': 'JEPA-H',
    'sam2-vit-s': 'SAM2-S',
    'sam2-vit-b': 'SAM2-B',
    'sam2-vit-l': 'SAM2-L',
}

# Configure matplotlib for professional paper-quality plots
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300


def generate_flux_image(prompt: str = "a beautiful landscape with mountains and lake",
                        seed: int = 0,
                        device: str = 'cuda:7') -> Tuple[Image.Image, torch.Tensor]:
    """Generate an image using Flux model."""
    from diffusers import FluxPipeline

    print(f"Generating image with Flux on {device}")
    print(f"Prompt: '{prompt}'")
    print(f"Seed: {seed}")

    # Load Flux pipeline on the specified device
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    ).to(device)

    # Generate image
    pil_images = pipe(
        prompt,
        height=768,
        width=768,
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        num_images_per_prompt=1,
        generator=torch.Generator("cpu").manual_seed(seed)
    ).images

    # Get the first image and resize to 256x256
    img = pil_images[0].resize((256, 256))

    # Convert to tensor in [0, 255] range
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()

    print(f"Generated image successfully")
    # save image to assets folder
    # img.save('assets/image1.png')
    # print(f"Saved image to assets/image1.png")

    # Clean up pipeline to free memory
    del pipe
    torch.cuda.empty_cache()

    return img, img_tensor


def load_test_image(image_path: Optional[str] = None) -> Tuple[Image.Image, torch.Tensor]:
    """Load a test image from file."""
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
    else:
        # Use first image from metrics data if available
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_path = os.path.join(script_dir, '..', 'metrics', 'data', 'images', '0.png')
        if os.path.exists(default_path):
            img = Image.open(default_path).convert('RGB')
            print(f"Using default image from: {default_path}")
        else:
            raise ValueError("No image found. Please provide --image_path or use --use_flux to generate one.")

    # Resize to 256x256
    img = img.resize((256, 256))

    # Convert to tensor
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()

    return img, img_tensor


def infer_grid_hw(T: int) -> Tuple[int, int]:
    """Infer grid height and width from number of tokens."""
    r = int(math.isqrt(T))
    if r * r == T:
        return r, r
    c = int(math.ceil(math.sqrt(T)))
    r = int(math.ceil(T / c))
    return r, c


def sample_reference_points(num_refs: int = 3, min_sep: float = 0.15) -> List[Tuple[float, float]]:
    """Sample reference points with minimum separation."""
    points = [(0.5, 0.5)]  # Always include center

    max_tries = 500
    tries = 0

    while len(points) < num_refs and tries < max_tries:
        y, x = random.random(), random.random()
        # Check minimum separation from existing points
        if all(max(abs(y - yy), abs(x - xx)) >= min_sep for (yy, xx) in points):
            points.append((y, x))
        tries += 1

    # Fill remaining with random points if needed
    while len(points) < num_refs:
        points.append((random.random(), random.random()))

    return points


def normalized_to_grid(norm_points: List[Tuple[float, float]], H: int, W: int) -> List[Tuple[int, int]]:
    """Convert normalized coordinates to grid indices."""
    grid_points = []
    for (y, x) in norm_points:
        r = int(round(y * (H - 1)))
        c = int(round(x * (W - 1)))
        r = max(0, min(H - 1, r))
        c = max(0, min(W - 1, c))
        grid_points.append((r, c))
    return grid_points


def compute_cosine_similarity_maps(
    features: torch.Tensor,
    ref_points: List[Tuple[int, int]],
    H: int, W: int
) -> List[torch.Tensor]:
    """Compute cosine similarity maps for reference points."""
    # features shape: (B, T, D)
    B = features.shape[0]
    features = features[0] if B == 1 else features.mean(0)  # Use first sample or average

    # L2 normalize features for cosine similarity
    features = features / (features.norm(dim=-1, keepdim=True) + 1e-12)

    # Reshape to spatial grid
    features_grid = features.reshape(H, W, -1)  # (H, W, D)

    similarity_maps = []
    for (r, c) in ref_points:
        ref_feat = features_grid[r, c]  # (D,)
        # Compute cosine similarity with all positions
        sim_map = torch.matmul(features_grid, ref_feat)  # (H, W)
        similarity_maps.append(sim_map)

    return similarity_maps


def extract_encoder_features(
    encoder_name: str,
    image_tensor: torch.Tensor,
    device: str = "cuda",
    apply_spatial_norm: bool = False,
    zscore_alpha: float = 0.6
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract features from encoder with optional spatial normalization."""

    # Load encoder
    encoder = load_encoders(encoder_name, device, resolution=256)[0]

    # Move image to device and preprocess
    image = image_tensor.to(device)
    image = encoder.preprocess(image)

    # Extract features
    with torch.no_grad():
        out = encoder.forward_features(image)
        patch_tokens = out['x_norm_patchtokens']
        cls_token = out.get('x_norm_clstoken', None)

        # Apply spatial normalization if requested
        if apply_spatial_norm:
            spatial_norm = SpatialNormalization("zscore")
            patch_tokens = spatial_norm(patch_tokens, zscore_alpha=zscore_alpha)

        # Move to CPU for visualization
        patch_tokens = patch_tokens.cpu()

    return patch_tokens, cls_token


def create_spatial_norm_comparison(
    image: Image.Image,
    image_tensor: torch.Tensor,
    encoders: List[str],
    num_refs: int = 4,
    device: str = "cuda",
    zscore_alpha: float = 0.6,
    output_path: str = "scripts/outputs/viz-spatialnorm.png"
):
    """Create the main figure comparing spatial normalization across encoders."""

    # Sample reference points (shared across all visualizations)
    random.seed(42)  # For reproducibility
    torch.manual_seed(42)
    ref_points_norm = sample_reference_points(num_refs)

    # Process each encoder with and without spatial normalization
    all_data = {}
    global_vmin, global_vmax = float('inf'), float('-inf')

    print("Processing encoders...")
    for encoder_name in encoders:
        print(f"  {encoder_name}...")

        # Extract features without spatial normalization
        features_no_norm, _ = extract_encoder_features(
            encoder_name, image_tensor, device,
            apply_spatial_norm=False
        )

        # Extract features with spatial normalization
        features_with_norm, _ = extract_encoder_features(
            encoder_name, image_tensor, device,
            apply_spatial_norm=True, zscore_alpha=zscore_alpha
        )

        # Infer grid dimensions
        T = features_no_norm.shape[1]
        H, W = infer_grid_hw(T)

        # Convert reference points to grid coordinates
        ref_points_grid = normalized_to_grid(ref_points_norm, H, W)

        # Compute similarity maps
        maps_no_norm = compute_cosine_similarity_maps(features_no_norm, ref_points_grid, H, W)
        maps_with_norm = compute_cosine_similarity_maps(features_with_norm, ref_points_grid, H, W)

        all_data[encoder_name] = {
            'no_norm': maps_no_norm,
            'with_norm': maps_with_norm,
            'ref_points': ref_points_grid,
            'H': H, 'W': W
        }

        # Update global min/max for consistent color scale
        for maps in [maps_no_norm, maps_with_norm]:
            for m in maps:
                global_vmin = min(global_vmin, m.min().item())
                global_vmax = max(global_vmax, m.max().item())

    print(f"Global similarity range: [{global_vmin:.3f}, {global_vmax:.3f}]")

    # Create compact figure with minimal white space
    n_encoders = len(encoders)
    n_cols_per_condition = num_refs + 1  # image + similarity maps

    # Calculate tight figure size
    fig_width = IMG_SIZE_PER_CELL * n_cols_per_condition * 2 + 0.8  # 2 conditions + small margin for labels
    fig_height = IMG_SIZE_PER_CELL * n_encoders + 0.4  # encoders + small margin for headers

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create main grid with minimal spacing
    gs_main = gridspec.GridSpec(
        n_encoders + 1, 4,  # +1 for header row, 4 columns (label + gap + 2 conditions)
        figure=fig,
        height_ratios=[HEADER_ROW_HEIGHT] + [1]*n_encoders,  # Small header, equal encoder rows
        width_ratios=[LABEL_COL_WIDTH, LABEL_COL_SPACING, 1, 1],  # Label, gap, two conditions
        hspace=ROW_SPACING, wspace=COL_SPACING,  # Use control parameters
        left=FIG_LEFT_MARGIN, right=FIG_RIGHT_MARGIN,
        top=FIG_TOP_MARGIN, bottom=FIG_BOTTOM_MARGIN  # Use margin parameters
    )

    # Add column headers (skip column 1 which is the gap)
    ax_no_norm = fig.add_subplot(gs_main[0, 2])
    ax_no_norm.text(0.5, 0.5, 'w/o Spatial Normalization Layer', ha='center', va='center',
                    fontsize=HEADER_FONT_SIZE, fontweight='normal')
    ax_no_norm.axis('off')

    ax_with_norm = fig.add_subplot(gs_main[0, 3])
    ax_with_norm.text(0.5, 0.5, 'w/ Spatial Normalization Layer', ha='center', va='center',
                      fontsize=HEADER_FONT_SIZE, fontweight='normal')
    ax_with_norm.axis('off')

    # Keep track of last image for colorbar
    last_im = None

    # Plot each encoder
    for enc_idx, encoder_name in enumerate(encoders):
        data = all_data[encoder_name]

        # Add row label (encoder name) - use display name mapping
        ax_label = fig.add_subplot(gs_main[enc_idx + 1, 0])
        display_name = ENCODER_DISPLAY_NAMES.get(encoder_name, encoder_name.replace('_full2b_224', '').replace('-vit-', '-'))
        ax_label.text(0.5, 0.5, display_name, ha='center', va='center',
                     fontsize=ENCODER_LABEL_SIZE, fontweight='normal', rotation=90)
        ax_label.axis('off')

        # Create subgrids for each condition (columns 2 and 3, skipping gap column 1)
        for cond_idx, cond_key in enumerate(['no_norm', 'with_norm']):
            # Create inner grid for this encoder-condition combination
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                1, n_cols_per_condition,
                subplot_spec=gs_main[enc_idx + 1, cond_idx + 2],  # +2 to skip label and gap columns
                wspace=INNER_SPACING  # Use control parameter
            )

            # First column: original image
            ax = fig.add_subplot(inner_gs[0, 0])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])

            # Remove spines for cleaner look
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Plot similarity maps
            for col_idx, sim_map in enumerate(data[cond_key]):
                ax = fig.add_subplot(inner_gs[0, col_idx + 1])

                # Plot similarity map
                last_im = ax.imshow(sim_map, vmin=global_vmin, vmax=global_vmax, cmap='viridis')

                # Mark reference point with a star
                r, c = data['ref_points'][col_idx]
                ax.scatter(c, r, s=STAR_SIZE, c=STAR_COLOR, marker='*',
                          edgecolors=STAR_EDGE_COLOR, linewidths=STAR_LINEWIDTH, zorder=10)

                ax.set_xticks([])
                ax.set_yticks([])

                # Remove spines
                for spine in ax.spines.values():
                    spine.set_visible(False)

    # Add shared colorbar - scale based on figure height
    if last_im is not None:
        # Calculate colorbar position to center it vertically
        cbar_height = COLORBAR_HEIGHT_RATIO
        cbar_bottom = (1 - cbar_height) / 2  # Center vertically

        cbar_ax = fig.add_axes([COLORBAR_X_POS, cbar_bottom, COLORBAR_WIDTH, cbar_height])
        cbar = fig.colorbar(last_im, cax=cbar_ax, orientation='vertical')

        # Scale text based on figure height
        scaled_label_size = COLORBAR_LABEL_SIZE if fig_height < 5 else COLORBAR_LABEL_SIZE + 2
        scaled_tick_size = COLORBAR_TICK_SIZE if fig_height < 5 else COLORBAR_TICK_SIZE + 2

        cbar.set_label('Cosine Similarity', fontsize=scaled_label_size, labelpad=COLORBAR_LABEL_PAD)
        cbar.ax.tick_params(labelsize=scaled_tick_size)

    # Save figure with high quality in both PNG and PDF formats
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved PNG figure to {output_path}")

    # Save as PDF (replace .png extension with .pdf)
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    print(f"Saved PDF figure to {pdf_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize spatial normalization effects across encoders')

    # Image input options
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to input image')
    parser.add_argument('--use_flux', action='store_true',
                       help='Use Flux to generate image instead of loading from disk')
    parser.add_argument('--prompt', type=str,
                       default='a cute dog playing in a garden',
                       help='Prompt for Flux image generation')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for Flux generation')

    # Visualization settings
    parser.add_argument('--num_refs', type=int, default=4,
                       help='Number of reference points (default: 4)')
    parser.add_argument('--zscore_alpha', type=float, default=0.6,
                       help='Alpha parameter for zscore normalization (default: 0.6)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for encoders (default: cuda)')
    parser.add_argument('--flux_device', type=str, default='cuda:1',
                       help='Device to use for Flux model (default: cuda:1)')
    parser.add_argument('--output_dir', type=str, default='scripts/outputs',
                       help='Output directory for figures')
    parser.add_argument('--output_index', type=int, default=1,
                       help='Output index (default: 1)')

    # Encoder selection
    parser.add_argument('--encoders', type=str, nargs='+',
                       default=['dinov3-vit-b16', 'pe-vit-g'],
                       help='Encoders to compare (default: dinov3-vit-b16, pe-vit-g)')

    args = parser.parse_args()

    # Load or generate image
    if args.use_flux:
        image, image_tensor = generate_flux_image(
            prompt=args.prompt,
            seed=args.seed,
            device=args.flux_device
        )
    else:
        image, image_tensor = load_test_image(args.image_path)

    # Create output path
    output_filename = f'viz-spatialnorm-v{args.output_index}.png'
    output_path = os.path.join(args.output_dir, output_filename)

    # Generate figure
    create_spatial_norm_comparison(
        image=image,
        image_tensor=image_tensor,
        encoders=args.encoders,
        num_refs=args.num_refs,
        device=args.device,
        zscore_alpha=args.zscore_alpha,
        output_path=output_path
    )


if __name__ == "__main__":
    main()
