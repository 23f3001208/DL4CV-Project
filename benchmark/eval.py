"""
Benchmark Evaluation Script
============================
Runs the pipeline on a set of images (or an entire directory) and
produces a comprehensive evaluation report.

Usage
-----
  python -m benchmark.eval \
      --input  data/test_images/ \
      --output results/benchmark/ \
      --depth_backend depth_anything_v2 \
      --seg_model nvidia/segformer-b2-finetuned-ade-512-512 \
      --intrinsic \
      --n_images 20

The script outputs:
  - Per-image JSON with all metrics
  - Aggregate CSV summary
  - Layer-grid PNG for each image
  - Markdown report (results/benchmark/report.md)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import LayeredReprPipeline, PipelineConfig
from src.utils import setup_logging, make_layer_grid
from benchmark.metrics import evaluate, BenchmarkResult

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def find_images(path: Path, limit: int = 0) -> List[Path]:
    if path.is_file():
        return [path]
    imgs = sorted(
        p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit:
        imgs = imgs[:limit]
    return imgs


def run_benchmark(args) -> None:
    setup_logging(logging.INFO)

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

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
    images = find_images(input_path, limit=args.n_images)

    if not images:
        logger.error("No images found at: %s", input_path)
        sys.exit(1)

    logger.info("Found %d images. Starting benchmark...", len(images))

    all_results: List[dict] = []
    aggregate = BenchmarkResult()
    n = 0

    for img_path in images:
        logger.info("Processing: %s", img_path.name)
        t_start = time.perf_counter()
        try:
            from PIL import Image
            original = Image.open(img_path).convert("RGB")
            repr_out = pipe.run(original, output_dir=output_path / "layers", prefix=img_path.stem)

            metrics = evaluate(repr_out, original)
            elapsed = time.perf_counter() - t_start

            # Layer grid
            grid = make_layer_grid(repr_out.layers, original, cols=4)
            grid.save(output_path / f"{img_path.stem}_grid.png")

            row = {
                "image": img_path.name,
                "elapsed_s": round(elapsed, 2),
                "num_layers": metrics.num_layers,
                "composite_psnr": round(metrics.composite_psnr, 3),
                "composite_ssim": round(metrics.composite_ssim, 4),
                "coverage": round(metrics.coverage, 4),
                "mean_overlap_iou": round(metrics.mean_overlap_iou, 4),
                "depth_spearman": round(metrics.depth_spearman, 4),
            }
            all_results.append(row)

            # Running aggregate
            aggregate.composite_psnr += metrics.composite_psnr
            aggregate.composite_ssim += metrics.composite_ssim
            aggregate.coverage       += metrics.coverage
            aggregate.mean_overlap_iou += metrics.mean_overlap_iou
            aggregate.depth_spearman += metrics.depth_spearman
            n += 1

        except Exception as exc:
            logger.error("Error on %s: %s", img_path.name, exc)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    if n > 0:
        aggregate.composite_psnr   /= n
        aggregate.composite_ssim   /= n
        aggregate.coverage         /= n
        aggregate.mean_overlap_iou /= n
        aggregate.depth_spearman   /= n
        aggregate.num_layers = n

    logger.info("\n%s", aggregate.summary())

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = output_path / "benchmark_results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_results)
        logger.info("CSV saved: %s", csv_path)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    summary_json = {
        "config": {
            "seg_model": config.seg_model,
            "depth_backend": config.depth_backend,
            "do_intrinsic": config.do_intrinsic,
            "depth_sort_metric": config.depth_sort_metric,
        },
        "n_images": n,
        "aggregate": {
            "composite_psnr":   round(aggregate.composite_psnr, 3),
            "composite_ssim":   round(aggregate.composite_ssim, 4),
            "coverage":         round(aggregate.coverage, 4),
            "mean_overlap_iou": round(aggregate.mean_overlap_iou, 4),
            "depth_spearman":   round(aggregate.depth_spearman, 4),
        },
        "per_image": all_results,
    }
    json_path = output_path / "benchmark_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    logger.info("JSON summary saved: %s", json_path)

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    _write_md_report(output_path, summary_json, all_results)
    logger.info("Markdown report saved: %s", output_path / "report.md")


def _write_md_report(output_path: Path, summary: dict, rows: list) -> None:
    cfg = summary["config"]
    agg = summary["aggregate"]
    lines = [
        "# Layered Representations — Benchmark Report\n",
        "## Configuration\n",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Segmentation model | `{cfg['seg_model']}` |",
        f"| Depth backend | `{cfg['depth_backend']}` |",
        f"| Intrinsic decomposition | `{cfg['do_intrinsic']}` |",
        f"| Depth sort metric | `{cfg['depth_sort_metric']}` |",
        "",
        f"Evaluated on **{summary['n_images']} images**.\n",
        "## Aggregate Results\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Composite PSNR | **{agg['composite_psnr']:.2f} dB** |",
        f"| Composite SSIM | **{agg['composite_ssim']:.4f}** |",
        f"| Coverage | **{agg['coverage']*100:.1f}%** |",
        f"| Mean Overlap IoU | **{agg['mean_overlap_iou']*100:.1f}%** (lower is better) |",
        f"| Depth Spearman ρ | **{agg['depth_spearman']:.4f}** |",
        "",
        "## Per-Image Results\n",
        "| Image | Layers | PSNR | SSIM | Coverage | Overlap | Spearman | Time(s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['image']} | {r['num_layers']} | {r['composite_psnr']:.2f} |"
            f" {r['composite_ssim']:.4f} | {r['coverage']*100:.1f}% |"
            f" {r['mean_overlap_iou']*100:.1f}% | {r['depth_spearman']:.3f} | {r['elapsed_s']} |"
        )

    lines += [
        "",
        "## Metric Definitions\n",
        "- **Composite PSNR / SSIM**: quality of the reconstructed image obtained by "
          "Porter–Duff compositing all RGBA layers back together. Higher is better.",
        "- **Coverage**: fraction of pixels covered by at least one layer. Should be ~100%.",
        "- **Mean Overlap IoU**: average pairwise intersection-over-union of layer alpha "
          "masks. Lower means cleaner layer separation.",
        "- **Depth Spearman ρ**: Spearman rank-correlation between the layer's depth rank "
          "and its actual median depth value. ρ=1 means perfect consistency.",
    ]

    with open(output_path / "report.md", "w") as f:
        f.write("\n".join(lines))


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark layered representation pipeline")
    p.add_argument("--input",   required=True, help="Path to image or directory")
    p.add_argument("--output",  default="results/benchmark", help="Output directory")
    p.add_argument("--seg_model",
                   default="nvidia/segformer-b2-finetuned-ade-512-512")
    p.add_argument("--depth_backend",   default="depth_anything_v2",
                   choices=["depth_anything_v2", "dpt", "midas"])
    p.add_argument("--depth_metric",    default="median",
                   choices=["median", "mean", "percentile25"])
    p.add_argument("--intrinsic_backend", default="sparse",
                   choices=["retinex", "sparse", "deep"])
    p.add_argument("--intrinsic",  action="store_true",
                   help="Enable intrinsic decomposition per layer")
    p.add_argument("--hard_edge", action="store_true",
                   help="Use hard binary masks (no soft-edge feathering)")
    p.add_argument("--n_images",  type=int, default=0,
                   help="Limit to first N images (0 = all)")
    p.add_argument("--max_size",  type=int, default=1024,
                   help="Resize longest image edge to this before inference")
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
