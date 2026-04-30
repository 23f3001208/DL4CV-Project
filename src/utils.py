"""
Utilities
=========
Shared helpers used across the project:
  - Logging setup
  - Image I/O helpers
  - Layer grid visualiser
  - SSIM / PSNR computations
  - Seam-carving / mask dilation helpers
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure root logger with a clean format."""
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    handlers: list = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: Union[str, Path]) -> Image.Image:
    """Load any image file as RGB PIL Image."""
    return Image.open(path).convert("RGB")


def save_image(img: Image.Image, path: Union[str, Path], quality: int = 95) -> None:
    """Save PIL image; infer format from extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = path.suffix.lower().lstrip(".")
    if fmt in ("jpg", "jpeg"):
        img.convert("RGB").save(path, format="JPEG", quality=quality)
    else:
        img.save(path, format=fmt.upper() if fmt else "PNG")


# ---------------------------------------------------------------------------
# Layer grid visualiser
# ---------------------------------------------------------------------------

def make_layer_grid(
    layers: list,          # list of Layer objects
    original: Image.Image,
    cell_size: Tuple[int, int] = (256, 256),
    cols: int = 4,
    bg_color: Tuple[int, int, int] = (30, 30, 30),
    label_height: int = 24,
    font_size: int = 14,
) -> Image.Image:
    """
    Render all layers in a grid for quick inspection.
    Checkerboard background shows transparency.

    Parameters
    ----------
    layers      : List[Layer]
    original    : PIL.Image.Image — shown in top-left cell
    cell_size   : (width, height) of each thumbnail
    cols        : number of grid columns
    bg_color    : RGB background colour
    label_height: height (px) of label strip below each cell
    """
    from .compositor import Layer  # local import to avoid circular

    cw, ch = cell_size
    n_layers = len(layers)
    n_cells  = 1 + n_layers          # original + layers
    rows = math.ceil(n_cells / cols)
    grid_w = cols * cw
    grid_h = rows * (ch + label_height)

    grid = Image.new("RGB", (grid_w, grid_h), bg_color)
    checker = _make_checker(cw, ch)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    def draw_cell(img_rgba: Image.Image, label: str, idx: int) -> None:
        col = idx % cols
        row = idx // cols
        x = col * cw
        y = row * (ch + label_height)

        # Checkerboard + paste
        cell = checker.copy()
        thumb = img_rgba.resize(cell_size, Image.LANCZOS)
        if thumb.mode == "RGBA":
            cell.paste(thumb, (0, 0), mask=thumb)
        else:
            cell.paste(thumb, (0, 0))
        grid.paste(cell, (x, y))

        # Label strip
        label_strip = Image.new("RGB", (cw, label_height), (20, 20, 20))
        draw = ImageDraw.Draw(label_strip)
        draw.text((4, 4), label, fill=(220, 220, 220), font=font)
        grid.paste(label_strip, (x, y + ch))

    draw_cell(original, "ORIGINAL", 0)
    for i, layer in enumerate(layers):
        label = f"[{layer.meta.rank}] {layer.meta.group}  d={layer.meta.median_depth:.2f}"
        draw_cell(layer.rgba, label, i + 1)

    return grid


def _make_checker(w: int, h: int, tile: int = 8) -> Image.Image:
    """Create a checkerboard background image (light/dark grey)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    light, dark = 200, 150
    for r in range(0, h, tile):
        for c in range(0, w, tile):
            col = light if (r // tile + c // tile) % 2 == 0 else dark
            arr[r:r+tile, c:c+tile] = col
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio between two [0,255] uint8 images."""
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 11,
    C1: float = 6.5025,
    C2: float = 58.5225,
) -> float:
    """
    Structural Similarity Index (SSIM) — single-channel grayscale.
    Inputs: H×W uint8 or float [0,1].
    """
    if img1.dtype == np.uint8:
        i1 = img1.astype(np.float64)
        i2 = img2.astype(np.float64)
    else:
        i1 = img1.astype(np.float64) * 255
        i2 = img2.astype(np.float64) * 255

    mu1 = cv2.GaussianBlur(i1, (window_size, window_size), 1.5)
    mu2 = cv2.GaussianBlur(i2, (window_size, window_size), 1.5)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(i1**2, (window_size, window_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(i2**2, (window_size, window_size), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(i1 * i2, (window_size, window_size), 1.5) - mu12

    num   = (2 * mu12 + C1) * (2 * sigma12 + C2)
    denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(np.mean(num / (denom + 1e-10)))


def layer_overlap_score(layers: list) -> float:
    """
    Compute mean pairwise mask overlap (IoU) between layers.
    Ideal = 0 (no overlap). Returns value in [0,1].
    """
    masks = [np.array(l.rgba)[:, :, 3] > 127 for l in layers]
    n = len(masks)
    if n < 2:
        return 0.0
    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = (masks[i] & masks[j]).sum()
            union = (masks[i] | masks[j]).sum()
            iou = inter / (union + 1e-6)
            scores.append(float(iou))
    return float(np.mean(scores))


def coverage_score(layers: list) -> float:
    """
    Fraction of pixels covered by at least one layer's alpha.
    Should be ~1.0 for a complete decomposition.
    """
    if not layers:
        return 0.0
    first_alpha = np.array(layers[0].rgba)[:, :, 3]
    union_mask = first_alpha > 0
    for l in layers[1:]:
        union_mask |= (np.array(l.rgba)[:, :, 3] > 0)
    return float(union_mask.sum() / union_mask.size)
