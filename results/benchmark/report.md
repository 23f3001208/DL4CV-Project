# Layered Representations — Benchmark Report

## Configuration

| Parameter | Value |
|---|---|
| Segmentation model | `nvidia/segformer-b2-finetuned-ade-512-512` |
| Depth backend | `depth_anything_v2` |
| Intrinsic decomposition | `False` |
| Depth sort metric | `median` |

Evaluated on **5 images**.

## Aggregate Results

| Metric | Value |
|---|---|
| Composite PSNR | **29.73 dB** |
| Composite SSIM | **0.8619** |
| Coverage | **100.0%** |
| Mean Overlap IoU | **0.0%** (lower is better) |
| Depth Spearman ρ | **1.0000** |

## Per-Image Results

| Image | Layers | PSNR | SSIM | Coverage | Overlap | Spearman | Time(s) |
|---|---|---|---|---|---|---|---|
| photo-1.jpg | 4 | 25.72 | 0.8382 | 100.0% | 0.0% | 1.000 | 9.4 |
| photo-2.jpg | 4 | 39.93 | 0.9519 | 100.0% | 0.0% | 1.000 | 1.97 |
| photo-3.jpg | 6 | 25.10 | 0.7924 | 100.0% | 0.0% | 1.000 | 2.05 |
| photo-4.jpg | 3 | 30.83 | 0.8868 | 100.0% | 0.0% | 1.000 | 2.03 |
| photo-5.jpg | 3 | 27.05 | 0.8401 | 100.0% | 0.0% | 1.000 | 2.09 |

## Metric Definitions

- **Composite PSNR / SSIM**: quality of the reconstructed image obtained by Porter–Duff compositing all RGBA layers back together. Higher is better.
- **Coverage**: fraction of pixels covered by at least one layer. Should be ~100%.
- **Mean Overlap IoU**: average pairwise intersection-over-union of layer alpha masks. Lower means cleaner layer separation.
- **Depth Spearman ρ**: Spearman rank-correlation between the layer's depth rank and its actual median depth value. ρ=1 means perfect consistency.