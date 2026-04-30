"""
Layer Compositor
=================
Merges semantic segmentation masks, depth ordering, and (optionally)
intrinsic decomposition into a clean, re-composable stack of RGBA layers.

Output spec (per layer)
-----------------------
  - mode        : RGBA
  - RGB channels: original pixel colour (or albedo if intrinsic requested)
  - A channel   : binary/soft mask derived from semantic segmentation
  - metadata    : group name, mean depth, depth rank, pixel count

Composition law: bottom-to-top alpha compositing recovers the original.
  I = Σ_{k=1}^{K}  α_k · C_k · Π_{j=k+1}^{K} (1 - α_j)
(Porter–Duff "over" operator, layers ordered far → near)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .depth import DepthEstimator, DepthResult
from .intrinsic import IntrinsicDecomposer, IntrinsicResult
from .segmentation import SegmentationResult

logger = logging.getLogger(__name__)


@dataclass
class LayerMeta:
    """Metadata attached to each output layer."""
    rank: int                   # 0 = nearest foreground, K-1 = farthest
    group: str                  # semantic group name
    mean_depth: float           # average depth [0,1]  0=near
    median_depth: float
    pixel_count: int
    pixel_fraction: float       # fraction of total image pixels
    ade20k_indices: List[int]   # constituent ADE20K class ids


@dataclass
class Layer:
    """A single RGBA layer."""
    meta: LayerMeta
    rgba: Image.Image           # PIL RGBA image
    albedo: Optional[Image.Image] = None   # RGB albedo (if intrinsic)
    shading: Optional[Image.Image] = None  # L  shading  (if intrinsic)


@dataclass
class LayeredRepresentation:
    """Complete layered output for one input image."""
    layers: List[Layer]         # ordered near → far
    original: Image.Image
    depth_vis: Image.Image      # depth map colourised for display
    seg_vis: Image.Image        # segmentation colormap
    composite: Image.Image      # reconstructed from layers (sanity check)
    metadata: Dict              # full JSON-serialisable metadata


# ---------------------------------------------------------------------------
# Soft-edge mask utilities
# ---------------------------------------------------------------------------

def _soften_mask(
    mask: np.ndarray,
    blur_radius: int = 3,
    erode_radius: int = 1,
) -> np.ndarray:
    """
    Apply slight erosion + Gaussian blur to a binary mask to remove
    hard jagged edges. Returns float32 [0,1].
    """
    m = mask.astype(np.uint8) * 255
    if erode_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erode_radius + 1, 2 * erode_radius + 1)
        )
        m = cv2.erode(m, k, iterations=1)
    if blur_radius > 0:
        m = cv2.GaussianBlur(m, (2 * blur_radius + 1, 2 * blur_radius + 1), 0)
    return m.astype(np.float32) / 255.0


def _depth_to_colormap(depth: np.ndarray) -> Image.Image:
    """Convert normalised depth [0,1] to a TURBO colormap RGB image."""
    d = (depth * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))


def _seg_to_colormap(
    seg: SegmentationResult,
    image_shape: Tuple[int, int],
) -> Image.Image:
    """Render semantic group segmentation as a coloured overlay."""
    h, w = image_shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    palette = {
        "person":       (255,  87,  34),
        "animal":       (233, 196,  56),
        "vehicle":      ( 33, 150, 243),
        "furniture":    (156,  39, 176),
        "architecture": (121,  85,  72),
        "nature":       ( 76, 175,  80),
        "sky":          (  3, 169, 244),
        "background":   (158, 158, 158),
    }
    for group, mask in seg.group_masks.items():
        c = palette.get(group, (200, 200, 200))
        vis[mask] = c
    return Image.fromarray(vis)


# ---------------------------------------------------------------------------
# Compositor
# ---------------------------------------------------------------------------

class LayerCompositor:
    """
    Builds a stack of RGBA layers from pre-computed segmentation and depth.

    Parameters
    ----------
    soft_edge : bool
        Apply soft-edge feathering to alpha masks.
    blur_radius : int
        Gaussian blur kernel radius for soft edges.
    erode_radius : int
        Erosion radius (pixels) before blurring.
    do_intrinsic : bool
        Run intrinsic decomposition per layer (stretch goal).
    intrinsic_backend : str
        "retinex" | "sparse" | "deep"
    depth_sort_metric : str
        "median" | "mean" | "percentile25"  — how to rank layers by depth.
    """

    def __init__(
        self,
        soft_edge: bool = True,
        blur_radius: int = 3,
        erode_radius: int = 1,
        do_intrinsic: bool = False,
        intrinsic_backend: str = "sparse",
        depth_sort_metric: str = "median",
    ) -> None:
        self.soft_edge = soft_edge
        self.blur_radius = blur_radius
        self.erode_radius = erode_radius
        self.do_intrinsic = do_intrinsic
        self.depth_sort_metric = depth_sort_metric

        if do_intrinsic:
            self._intrinsic = IntrinsicDecomposer(backend=intrinsic_backend)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mask_to_alpha(self, mask: np.ndarray) -> np.ndarray:
        """Convert boolean mask → uint8 alpha channel [0,255]."""
        if self.soft_edge:
            soft = _soften_mask(mask, self.blur_radius, self.erode_radius)
            return (soft * 255).clip(0, 255).astype(np.uint8)
        return (mask.astype(np.uint8) * 255)

    def _group_depth_score(
        self, depth: DepthResult, mask: np.ndarray
    ) -> float:
        vals = depth.depth_map[mask]
        if vals.size == 0:
            return 1.0
        if self.depth_sort_metric == "mean":
            return float(vals.mean())
        if self.depth_sort_metric == "percentile25":
            return float(np.percentile(vals, 25))
        return float(np.median(vals))  # default: median

    # ------------------------------------------------------------------
    # Composite reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def recompose(layers: List[Layer]) -> Image.Image:
        """
        Porter–Duff alpha composite all layers (far → near) to reconstruct
        the original image.
        """
        # Sort far → near (reverse rank order)
        sorted_layers = sorted(layers, key=lambda l: l.meta.rank, reverse=True)
        w, h = sorted_layers[0].rgba.size
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        for layer in sorted_layers:
            canvas = Image.alpha_composite(canvas, layer.rgba)
        return canvas.convert("RGB")

    # ------------------------------------------------------------------
    # Main compose
    # ------------------------------------------------------------------

    def compose(
        self,
        image: Image.Image,
        seg: SegmentationResult,
        depth: DepthResult,
    ) -> LayeredRepresentation:
        """
        Compose a full layered representation.

        Parameters
        ----------
        image   : PIL.Image.Image   — original RGB input
        seg     : SegmentationResult
        depth   : DepthResult
        """
        orig_rgb = np.array(image.convert("RGB"))
        h, w = orig_rgb.shape[:2]
        total_pixels = h * w

        layers: List[Layer] = []

        # --- sort groups by depth score (near → far) ------------------
        group_scores: Dict[str, float] = {}
        for group, mask in seg.group_masks.items():
            group_scores[group] = self._group_depth_score(depth, mask)

        sorted_groups = sorted(group_scores.items(), key=lambda x: x[1])

        # --- build each layer ----------------------------------------
        for rank, (group, score) in enumerate(sorted_groups):
            mask = seg.group_masks[group]
            alpha = self._mask_to_alpha(mask)

            # RGBA layer
            rgba_np = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_np[..., :3] = orig_rgb
            rgba_np[..., 3]  = alpha
            # Zero out pixels outside the mask (clean layers)
            alpha_bool = alpha > 0
            rgba_np[~alpha_bool, :3] = 0
            rgba_img = Image.fromarray(rgba_np, mode="RGBA")

            # Depth stats
            vals = depth.depth_map[mask]
            mean_d   = float(vals.mean())   if vals.size > 0 else 1.0
            median_d = float(np.median(vals)) if vals.size > 0 else 1.0
            pix_count = int(mask.sum())

            from .segmentation import SEMANTIC_GROUPS
            ade_indices = SEMANTIC_GROUPS.get(group, [])

            meta = LayerMeta(
                rank=rank,
                group=group,
                mean_depth=round(mean_d, 4),
                median_depth=round(median_d, 4),
                pixel_count=pix_count,
                pixel_fraction=round(pix_count / total_pixels, 4),
                ade20k_indices=ade_indices,
            )

            layer = Layer(meta=meta, rgba=rgba_img)

            # Optional intrinsic decomposition
            if self.do_intrinsic:
                try:
                    ir: IntrinsicResult = self._intrinsic.decompose(image, mask)
                    albedo_img = Image.fromarray(
                        (ir.albedo * 255).clip(0, 255).astype(np.uint8), mode="RGB"
                    )
                    shading_np = (ir.shading * 255).clip(0, 255).astype(np.uint8)
                    shading_img = Image.fromarray(shading_np, mode="L")
                    layer.albedo  = albedo_img
                    layer.shading = shading_img
                except Exception as exc:
                    logger.warning("Intrinsic decomposition failed for '%s': %s", group, exc)

            layers.append(layer)

        # --- visualisations -----------------------------------------
        depth_vis = _depth_to_colormap(depth.depth_map)
        seg_vis   = _seg_to_colormap(seg, (h, w))
        composite = self.recompose(layers)

        # --- metadata dict ------------------------------------------
        metadata = {
            "num_layers": len(layers),
            "image_size": {"width": w, "height": h},
            "depth_model": depth.model_name,
            "depth_sort_metric": self.depth_sort_metric,
            "soft_edge": self.soft_edge,
            "intrinsic": self.do_intrinsic,
            "layers": [
                {
                    "rank": l.meta.rank,
                    "group": l.meta.group,
                    "mean_depth": l.meta.mean_depth,
                    "median_depth": l.meta.median_depth,
                    "pixel_count": l.meta.pixel_count,
                    "pixel_fraction": l.meta.pixel_fraction,
                }
                for l in layers
            ],
        }

        return LayeredRepresentation(
            layers=layers,
            original=image,
            depth_vis=depth_vis,
            seg_vis=seg_vis,
            composite=composite,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Saving utilities
    # ------------------------------------------------------------------

    def save(
        self,
        representation: LayeredRepresentation,
        output_dir: str | Path,
        prefix: str = "output",
    ) -> List[Path]:
        """
        Save all layers + visualisations + metadata to output_dir.

        File layout
        -----------
        output_dir/
          {prefix}_layer_00_person.png
          {prefix}_layer_01_vehicle.png
          ...
          {prefix}_depth.png
          {prefix}_segmentation.png
          {prefix}_composite.png
          {prefix}_original.png
          {prefix}_metadata.json
          {prefix}_layer_00_person_albedo.png    (if intrinsic)
          {prefix}_layer_00_person_shading.png   (if intrinsic)
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        # Layers
        for layer in representation.layers:
            name = f"{prefix}_layer_{layer.meta.rank:02d}_{layer.meta.group}.png"
            p = out / name
            layer.rgba.save(p, format="PNG", optimize=False)
            saved.append(p)
            if layer.albedo is not None:
                pa = out / f"{prefix}_layer_{layer.meta.rank:02d}_{layer.meta.group}_albedo.png"
                layer.albedo.save(pa)
                saved.append(pa)
            if layer.shading is not None:
                ps = out / f"{prefix}_layer_{layer.meta.rank:02d}_{layer.meta.group}_shading.png"
                layer.shading.save(ps)
                saved.append(ps)

        # Visualisations
        for tag, img in [
            ("original",     representation.original),
            ("depth",        representation.depth_vis),
            ("segmentation", representation.seg_vis),
            ("composite",    representation.composite),
        ]:
            p = out / f"{prefix}_{tag}.png"
            img.save(p)
            saved.append(p)

        # Metadata JSON
        meta_path = out / f"{prefix}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(representation.metadata, f, indent=2)
        saved.append(meta_path)

        logger.info("Saved %d files to %s", len(saved), out)
        return saved
