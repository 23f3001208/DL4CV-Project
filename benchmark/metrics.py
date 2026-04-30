"""
Benchmark Metrics
==================
Evaluation metrics for the layered representation system.

Metrics are grouped into three categories:

A) Reconstruction Quality
   - Composite PSNR / SSIM  (does recomposing the layers recover the original?)

B) Layer Decomposition Quality
   - Coverage         : fraction of pixels covered by ≥1 layer
   - Overlap IoU      : mean pairwise alpha-mask overlap (lower = better separation)
   - Depth Consistency: do depth ranks correlate with expected near/far ordering?

C) Semantic Quality (if ground-truth seg map available)
   - Segmentation mIoU against a reference segmentation map.

D) Intrinsic Quality (if GT intrinsic images available — MIT dataset)
   - Albedo MSE / SSIM
   - Shading MSE / SSIM
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from src.utils import psnr, ssim, layer_overlap_score, coverage_score


@dataclass
class BenchmarkResult:
    """Container for all evaluation metrics."""

    # Reconstruction
    composite_psnr: float = 0.0
    composite_ssim: float = 0.0

    # Decomposition
    coverage: float = 0.0
    mean_overlap_iou: float = 0.0
    num_layers: int = 0

    # Depth ordering
    depth_spearman: float = 0.0   # Spearman ρ between depth rank and semantic priority

    # Semantic (optional — requires GT)
    seg_miou: Optional[float] = None

    # Intrinsic (optional — requires GT)
    albedo_mse: Optional[float] = None
    albedo_ssim: Optional[float] = None
    shading_mse: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "  BENCHMARK RESULTS",
            "=" * 50,
            f"  Composite PSNR : {self.composite_psnr:6.2f} dB",
            f"  Composite SSIM : {self.composite_ssim:6.4f}",
            f"  Coverage       : {self.coverage*100:6.2f}%",
            f"  Mean Overlap   : {self.mean_overlap_iou*100:6.2f}% (lower=better)",
            f"  Num Layers     : {self.num_layers}",
            f"  Depth Spearman : {self.depth_spearman:6.4f}",
        ]
        if self.seg_miou is not None:
            lines.append(f"  Seg mIoU       : {self.seg_miou*100:6.2f}%")
        if self.albedo_mse is not None:
            lines.append(f"  Albedo MSE     : {self.albedo_mse:.5f}")
            lines.append(f"  Albedo SSIM    : {self.albedo_ssim:.4f}")
            lines.append(f"  Shading MSE    : {self.shading_mse:.5f}")
        lines.append("=" * 50)
        return "\n".join(lines)


def evaluate(
    representation,                        # LayeredRepresentation
    original: Image.Image,
    gt_seg_map: Optional[np.ndarray] = None,   # H×W int32 with class ids
    gt_albedo: Optional[np.ndarray]  = None,   # H×W×3 float32 [0,1]
    gt_shading: Optional[np.ndarray] = None,   # H×W float32 [0,1]
) -> BenchmarkResult:
    """
    Run all applicable metrics on a LayeredRepresentation.

    Parameters
    ----------
    representation : LayeredRepresentation
    original       : PIL.Image.Image  — ground-truth original image
    gt_seg_map     : optional H×W int array (ground-truth labels)
    gt_albedo      : optional H×W×3 float array
    gt_shading     : optional H×W float array
    """
    result = BenchmarkResult()
    layers = representation.layers

    # ------------------------------------------------------------------
    # A) Reconstruction quality
    # ------------------------------------------------------------------
    orig_arr = np.array(original.convert("RGB"))
    comp_arr = np.array(representation.composite.convert("RGB"))

    result.composite_psnr = psnr(orig_arr, comp_arr)
    orig_gray = np.array(original.convert("L"))
    comp_gray = np.array(representation.composite.convert("L"))
    result.composite_ssim = ssim(orig_gray, comp_gray)

    # ------------------------------------------------------------------
    # B) Decomposition quality
    # ------------------------------------------------------------------
    result.num_layers    = len(layers)
    result.coverage      = coverage_score(layers)
    result.mean_overlap_iou = layer_overlap_score(layers)

    # ------------------------------------------------------------------
    # C) Depth ordering consistency
    # ------------------------------------------------------------------
    result.depth_spearman = _depth_spearman(layers)

    # ------------------------------------------------------------------
    # D) Semantic mIoU (optional)
    # ------------------------------------------------------------------
    if gt_seg_map is not None:
        result.seg_miou = _seg_miou(representation, gt_seg_map)

    # ------------------------------------------------------------------
    # E) Intrinsic quality (optional)
    # ------------------------------------------------------------------
    if gt_albedo is not None and layers:
        pred_albedos = [l.albedo for l in layers if l.albedo is not None]
        if pred_albedos:
            # Flatten albedos from all layers into a single composite albedo
            h, w = orig_arr.shape[:2]
            pred_alb = np.zeros((h, w, 3), dtype=np.float32)
            for layer in layers:
                if layer.albedo is None:
                    continue
                alpha = np.array(layer.rgba)[:, :, 3].astype(np.float32) / 255.0
                alb_np = np.array(layer.albedo).astype(np.float32) / 255.0
                pred_alb += alb_np * alpha[..., np.newaxis]
            pred_alb = np.clip(pred_alb, 0, 1)
            result.albedo_mse  = float(np.mean((pred_alb - gt_albedo) ** 2))
            result.albedo_ssim = ssim(
                (pred_alb * 255).astype(np.uint8).mean(-1),
                (gt_albedo * 255).astype(np.uint8).mean(-1)
            )

    if gt_shading is not None and layers:
        pred_shadings = [l.shading for l in layers if l.shading is not None]
        if pred_shadings:
            h, w = orig_arr.shape[:2]
            pred_sh = np.zeros((h, w), dtype=np.float32)
            for layer in layers:
                if layer.shading is None:
                    continue
                alpha = np.array(layer.rgba)[:, :, 3].astype(np.float32) / 255.0
                sh_np = np.array(layer.shading).astype(np.float32) / 255.0
                pred_sh += sh_np * alpha
            pred_sh = np.clip(pred_sh, 0, 1)
            result.shading_mse = float(np.mean((pred_sh - gt_shading) ** 2))

    return result


def _depth_spearman(layers: list) -> float:
    """
    Spearman rank correlation between the layer's assigned depth rank
    and its median depth value. 
    Perfect positive correlation (ρ=1) means depth ordering is consistent.
    """
    if len(layers) < 2:
        return 1.0
    ranks  = np.array([l.meta.rank for l in layers], dtype=float)
    depths = np.array([l.meta.median_depth for l in layers], dtype=float)
    # Spearman: rank both arrays, then Pearson on ranks
    r_ranks  = _rank_array(ranks)
    r_depths = _rank_array(depths)
    n = len(r_ranks)
    cov = np.sum((r_ranks - r_ranks.mean()) * (r_depths - r_depths.mean())) / n
    std = r_ranks.std() * r_depths.std()
    return float(cov / (std + 1e-10))


def _rank_array(arr: np.ndarray) -> np.ndarray:
    """Return ranks (1-indexed) for array."""
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


def _seg_miou(representation, gt_seg_map: np.ndarray) -> float:
    """
    Compute mean-IoU between predicted semantic groups and GT label map.
    Maps each predicted group mask to the GT class it overlaps most.
    """
    from src.segmentation import SEMANTIC_GROUPS
    layers = representation.layers
    ious = []
    for layer in layers:
        alpha = np.array(layer.rgba)[:, :, 3] > 127
        ade_ids = layer.meta.ade20k_indices
        if not ade_ids:
            continue
        gt_mask = np.isin(gt_seg_map, ade_ids)
        inter = (alpha & gt_mask).sum()
        union = (alpha | gt_mask).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0
