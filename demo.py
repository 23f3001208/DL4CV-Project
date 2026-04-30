"""
demo.py — Quick demonstration of the pipeline
===============================================
Generates a synthetic test image (or uses a provided path) and runs the
full pipeline, saving all outputs to  results/demo/.

Usage
-----
  python demo.py                        # use synthetic test image
  python demo.py --image path/to/img   # use your own image
  python demo.py --image img.jpg --intrinsic --depth_backend dpt

Output
------
  results/demo/
    demo_layer_00_<group>.png     ← RGBA layers (near → far)
    demo_layer_01_<group>.png
    ...
    demo_depth.png                ← depth map (TURBO colourmap)
    demo_segmentation.png         ← semantic group colouring
    demo_composite.png            ← reconstruction from layers
    demo_original.png             ← preprocessed input
    demo_metadata.json            ← full metadata
    demo_grid.png                 ← layer thumbnail grid
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import LayeredReprPipeline, PipelineConfig
from src.utils import setup_logging, make_layer_grid, save_image


def make_synthetic_image(width: int = 640, height: int = 480):
    """
    Generate a simple synthetic scene:
      - Blue sky gradient (top)
      - Green ground plane (bottom)
      - Grey building rectangle (centre-right)
      - Brown person silhouette (centre-left)
      - Yellow vehicle rectangle (right)
    Returns a PIL RGB Image.
    """
    from PIL import Image as PILImage, ImageDraw
    import numpy as np

    img = PILImage.new("RGB", (width, height), (135, 206, 235))
    draw = ImageDraw.Draw(img)

    # Sky gradient (blue → lighter blue)
    for y in range(height // 2):
        t = y / (height // 2)
        r = int(135 + (200 - 135) * t)
        g = int(206 + (230 - 206) * t)
        b = int(235)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Ground (green)
    draw.rectangle([0, height // 2, width, height], fill=(90, 140, 60))

    # Building (grey, far)
    draw.rectangle([400, height // 4, 580, height // 2 + 20], fill=(160, 160, 170))
    # Windows
    for wy in range(height // 4 + 15, height // 2 + 10, 30):
        for wx in range(415, 580, 35):
            draw.rectangle([wx, wy, wx + 20, wy + 18], fill=(100, 130, 200))

    # Vehicle / car (yellow, mid distance)
    draw.rectangle([430, height // 2 - 10, 560, height // 2 + 40], fill=(230, 180, 30))
    # Wheels
    draw.ellipse([440, height // 2 + 30, 470, height // 2 + 55], fill=(30, 30, 30))
    draw.ellipse([520, height // 2 + 30, 550, height // 2 + 55], fill=(30, 30, 30))

    # Person silhouette (near)
    cx, cy = 200, height // 2 - 10
    draw.ellipse([cx - 18, cy - 60, cx + 18, cy], fill=(80, 60, 50))   # head
    draw.rectangle([cx - 20, cy, cx + 20, cy + 80], fill=(70, 100, 160))  # body
    draw.rectangle([cx - 30, cy + 80, cx - 10, cy + 140], fill=(50, 50, 80))  # leg L
    draw.rectangle([cx + 10, cy + 80, cx + 30, cy + 140], fill=(50, 50, 80))  # leg R

    # Tree / plant
    draw.rectangle([80, height // 2 - 10, 100, height // 2 + 80], fill=(100, 70, 40))
    draw.ellipse([40, height // 4 + 10, 140, height // 2 + 10], fill=(40, 120, 40))

    return img


def main(args) -> None:
    setup_logging(logging.INFO)
    logger = logging.getLogger("demo")

    # ── Build config ──────────────────────────────────────────────────
    config = PipelineConfig(
        seg_model=args.seg_model,
        depth_backend=args.depth_backend,
        do_intrinsic=args.intrinsic,
        intrinsic_backend=args.intrinsic_backend,
        soft_edge=not args.hard_edge,
        depth_sort_metric=args.depth_metric,
        max_image_size=args.max_size,
    )
    pipe = LayeredReprPipeline(config=config)

    # ── Load / create image ───────────────────────────────────────────
    if args.image:
        from PIL import Image
        img = Image.open(args.image).convert("RGB")
        logger.info("Loaded image: %s  (%dx%d)", args.image, *img.size)
    else:
        logger.info("No image provided — using synthetic test scene.")
        img = make_synthetic_image(640, 480)

    # ── Run pipeline ──────────────────────────────────────────────────
    out_dir = Path(args.output)
    repr_out = pipe.run(img, output_dir=out_dir, prefix="demo")

    # ── Layer grid ────────────────────────────────────────────────────
    grid = make_layer_grid(repr_out.layers, repr_out.original, cols=4)
    grid_path = out_dir / "demo_grid.png"
    save_image(grid, grid_path)
    logger.info("Layer grid saved: %s", grid_path)

    # ── Print metadata ────────────────────────────────────────────────
    import json
    logger.info("\n%s", json.dumps(repr_out.metadata, indent=2))

    # ── Summary table ────────────────────────────────────────────────
    logger.info("\n  Rank  Group           Depth(med)  Pixels")
    logger.info("  " + "-" * 50)
    for layer in repr_out.layers:
        m = layer.meta
        logger.info(
            "  %-5d %-15s  %.3f       %d (%.1f%%)",
            m.rank, m.group, m.median_depth, m.pixel_count,
            m.pixel_fraction * 100,
        )

    logger.info("\nAll outputs saved to: %s", out_dir.resolve())


def parse_args():
    p = argparse.ArgumentParser(description="Layered Repr pipeline demo")
    p.add_argument("--image",   default=None,
                   help="Path to an input image (default: synthetic scene)")
    p.add_argument("--output",  default="results/demo",
                   help="Output directory")
    p.add_argument("--seg_model",
                   default="nvidia/segformer-b2-finetuned-ade-512-512")
    p.add_argument("--depth_backend", default="depth_anything_v2",
                   choices=["depth_anything_v2", "dpt", "midas"])
    p.add_argument("--depth_metric",  default="median",
                   choices=["median", "mean", "percentile25"])
    p.add_argument("--intrinsic_backend", default="sparse",
                   choices=["retinex", "sparse", "deep"])
    p.add_argument("--intrinsic",  action="store_true")
    p.add_argument("--hard_edge", action="store_true")
    p.add_argument("--max_size",  type=int, default=1024)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
