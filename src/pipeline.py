"""
LayeredReprPipeline — Main Orchestrator
=========================================
Wires segmentation → depth → (intrinsic) → compositor into a single
callable interface.

Quick start
-----------
>>> from src.pipeline import LayeredReprPipeline
>>> pipe = LayeredReprPipeline()
>>> result = pipe.run("my_photo.jpg")
>>> pipe.compositor.save(result, "outputs/", prefix="my_photo")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .compositor import LayerCompositor, LayeredRepresentation
from .depth import DepthEstimator
from .segmentation import SemanticSegmenter

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Full configuration for the pipeline."""

    # --- Segmentation ---
    seg_model: str = "nvidia/segformer-b2-finetuned-ade-512-512"
    seg_min_mask_fraction: float = 0.001

    # --- Depth ---
    depth_backend: str = "depth_anything_v2"
    depth_model: Optional[str] = None  # None = use backend default

    # --- Compositor ---
    soft_edge: bool = True
    blur_radius: int = 3
    erode_radius: int = 1
    depth_sort_metric: str = "median"   # "median" | "mean" | "percentile25"

    # --- Intrinsic (stretch) ---
    do_intrinsic: bool = False
    intrinsic_backend: str = "sparse"   # "retinex" | "sparse" | "deep"

    # --- Runtime ---
    device: Optional[str] = None        # None = auto
    max_image_size: int = 1024          # resize longest edge to this before inference


class LayeredReprPipeline:
    """
    End-to-end pipeline: single image → layered RGBA representation.

    Parameters
    ----------
    config : PipelineConfig | None
        Pipeline configuration. Defaults to PipelineConfig() if None.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._seg_model: Optional[SemanticSegmenter] = None
        self._depth_model: Optional[DepthEstimator] = None
        self.compositor = LayerCompositor(
            soft_edge=self.config.soft_edge,
            blur_radius=self.config.blur_radius,
            erode_radius=self.config.erode_radius,
            do_intrinsic=self.config.do_intrinsic,
            intrinsic_backend=self.config.intrinsic_backend,
            depth_sort_metric=self.config.depth_sort_metric,
        )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def seg(self) -> SemanticSegmenter:
        if self._seg_model is None:
            self._seg_model = SemanticSegmenter(
                model_name=self.config.seg_model,
                device=self.config.device,
                min_mask_fraction=self.config.seg_min_mask_fraction,
            )
        return self._seg_model

    @property
    def depth(self) -> DepthEstimator:
        if self._depth_model is None:
            self._depth_model = DepthEstimator(
                backend=self.config.depth_backend,
                model_variant=self.config.depth_model,
                device=self.config.device,
            )
        return self._depth_model

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Resize image so its longest edge ≤ max_image_size."""
        w, h = image.size
        max_side = max(w, h)
        if max_side > self.config.max_image_size:
            scale = self.config.max_image_size / max_side
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            logger.debug("Resized image from (%d,%d) to (%d,%d).", w, h, new_w, new_h)
        return image.convert("RGB")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        image_input: Union[str, Path, Image.Image],
        output_dir: Optional[Union[str, Path]] = None,
        prefix: str = "output",
    ) -> LayeredRepresentation:
        """
        Run the full pipeline on a single image.

        Parameters
        ----------
        image_input : str | Path | PIL.Image.Image
            Path to an image file or a PIL Image object.
        output_dir  : str | Path | None
            If provided, saves all outputs there automatically.
        prefix      : str
            Filename prefix for saved files.

        Returns
        -------
        LayeredRepresentation
        """
        t0 = time.perf_counter()

        # Load image
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
            if prefix == "output":
                prefix = Path(image_input).stem
        else:
            image = image_input.convert("RGB")

        image = self._preprocess(image)
        logger.info("Input image: %dx%d", *image.size)

        # Stage 1 — Semantic Segmentation
        t1 = time.perf_counter()
        seg_result = self.seg.segment(image)
        logger.info(
            "[Segmentation] %.2fs — %d groups: %s",
            time.perf_counter() - t1,
            len(seg_result.group_names),
            seg_result.group_names,
        )

        # Stage 2 — Depth Estimation
        t2 = time.perf_counter()
        depth_result = self.depth.estimate(image)
        logger.info("[Depth] %.2fs — model: %s",
                    time.perf_counter() - t2, depth_result.model_name)

        # Stage 3 — Layer Composition (+ optional intrinsic)
        t3 = time.perf_counter()
        representation = self.compositor.compose(image, seg_result, depth_result)
        logger.info(
            "[Compositor] %.2fs — %d layers generated",
            time.perf_counter() - t3,
            len(representation.layers),
        )

        logger.info("Total pipeline time: %.2fs", time.perf_counter() - t0)

        # Optional save
        if output_dir is not None:
            self.compositor.save(representation, output_dir, prefix=prefix)

        return representation

    def run_batch(
        self,
        image_paths: list,
        output_dir: Union[str, Path],
    ) -> list:
        """Process a list of image paths and save to output_dir."""
        results = []
        for p in image_paths:
            try:
                r = self.run(p, output_dir=output_dir)
                results.append(r)
            except Exception as exc:
                logger.error("Failed on %s: %s", p, exc)
        return results
