"""
Monocular Depth Estimation Module
==================================
Provides metric-agnostic relative depth maps using one of several
state-of-the-art monocular depth estimators. Depth maps are normalized
to [0, 1] where **0 = nearest** and **1 = farthest**.

Supported backends
------------------
- "depth_anything_v2"  : Depth Anything V2 (LiheYoung/depth-anything-large-hf)
- "dpt"                : DPT-Large / DPT-Hybrid (Intel/dpt-large)
- "midas"              : MiDaS v3 (intel-isl/MiDaS via torch.hub)

Default: depth_anything_v2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Output of the depth estimation stage."""
    depth_map: np.ndarray     # H×W float32, 0=near 1=far
    raw_inverse: np.ndarray   # H×W float32 — raw model output (disparity)
    model_name: str


class DepthEstimator:
    """
    Unified monocular depth estimator.

    Parameters
    ----------
    backend : str
        One of "depth_anything_v2", "dpt", "midas".
    model_variant : str | None
        Override the default model id / variant.
    device : str | None
        "cuda", "mps", "cpu" or None (auto-detect).
    """

    _DEFAULTS = {
        "depth_anything_v2": "LiheYoung/depth-anything-large-hf",
        "dpt": "Intel/dpt-large",
        "midas": "DPT_Large",   # torch.hub variant
    }

    def __init__(
        self,
        backend: str = "depth_anything_v2",
        model_variant: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_variant = model_variant or self._DEFAULTS.get(backend, "")

        logger.info("Loading depth backend: %s (%s) on %s",
                    backend, self.model_variant, self.device)
        self._load_model()
        logger.info("Depth model loaded.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self.backend == "depth_anything_v2":
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self._processor = AutoImageProcessor.from_pretrained(self.model_variant)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.model_variant)
            self._model.eval().to(self.device)

        elif self.backend == "dpt":
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            self._processor = DPTImageProcessor.from_pretrained(self.model_variant)
            self._model = DPTForDepthEstimation.from_pretrained(self.model_variant)
            self._model.eval().to(self.device)

        elif self.backend == "midas":
            self._model = torch.hub.load(
                "intel-isl/MiDaS", self.model_variant,
                pretrained=True, trust_repo=True
            )
            self._model.eval().to(self.device)
            transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            self._midas_transform = (
                transforms.dpt_transform
                if "DPT" in self.model_variant
                else transforms.small_transform
            )
        else:
            raise ValueError(f"Unknown depth backend: {self.backend!r}")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_hf(self, image: Image.Image) -> np.ndarray:
        """Run HuggingFace-based model (depth_anything_v2 or dpt)."""
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        pred = outputs.predicted_depth  # B×H'×W'
        # Upsample to original size
        orig_w, orig_h = image.size
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return pred.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _run_midas(self, image: Image.Image) -> np.ndarray:
        """Run MiDaS (torch.hub) model."""
        import cv2
        img_np = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        inp = self._midas_transform(img_bgr).to(self.device)
        pred = self._model(inp)
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=(img_np.shape[0], img_np.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return pred.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, image: Image.Image) -> DepthResult:
        """
        Estimate depth for a PIL image.

        Returns DepthResult with depth_map in [0,1] where 0=near, 1=far.
        Note: models typically output *inverse depth* (disparity); we invert
        to get depth ordering where larger value = farther.
        """
        if self.backend == "midas":
            raw = self._run_midas(image)
        else:
            raw = self._run_hf(image)

        raw_inverse = raw.copy()

        # All these models output inverse depth / disparity.
        # Invert: large disparity → near → small depth value.
        # Normalize to [0, 1] where 0 = near, 1 = far.
        d_min, d_max = raw.min(), raw.max()
        if d_max > d_min:
            # depth ∝ 1 / disparity  →  near pixel has HIGH disparity, LOW depth
            depth_map = 1.0 - (raw - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(raw)

        return DepthResult(
            depth_map=depth_map,
            raw_inverse=raw_inverse,
            model_name=self.model_variant,
        )

    def median_depth_in_mask(
        self, depth_result: DepthResult, mask: np.ndarray
    ) -> float:
        """Return median depth value within a boolean mask."""
        vals = depth_result.depth_map[mask]
        return float(np.median(vals)) if vals.size > 0 else 1.0

    def percentile_depth_in_mask(
        self, depth_result: DepthResult, mask: np.ndarray, pct: float = 25.0
    ) -> float:
        """Return p-th percentile depth within a mask (lower = nearer)."""
        vals = depth_result.depth_map[mask]
        return float(np.percentile(vals, pct)) if vals.size > 0 else 1.0
