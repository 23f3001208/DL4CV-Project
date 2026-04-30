"""
layered_repr.src
================
Layered Representations from a Single Image.

Imports are lazy to avoid torch requirement at import time.
"""

__all__ = [
    "LayeredReprPipeline",
    "PipelineConfig",
    "LayeredRepresentation",
    "Layer",
    "LayerMeta",
    "SemanticSegmenter",
    "SegmentationResult",
    "DepthEstimator",
    "DepthResult",
    "IntrinsicDecomposer",
    "IntrinsicResult",
]


def __getattr__(name):
    if name in ("LayeredReprPipeline", "PipelineConfig"):
        from .pipeline import LayeredReprPipeline, PipelineConfig
        return locals()[name]
    if name in ("LayeredRepresentation", "Layer", "LayerMeta"):
        from .compositor import LayeredRepresentation, Layer, LayerMeta
        return locals()[name]
    if name in ("SemanticSegmenter", "SegmentationResult"):
        from .segmentation import SemanticSegmenter, SegmentationResult
        return locals()[name]
    if name in ("DepthEstimator", "DepthResult"):
        from .depth import DepthEstimator, DepthResult
        return locals()[name]
    if name in ("IntrinsicDecomposer", "IntrinsicResult"):
        from .intrinsic import IntrinsicDecomposer, IntrinsicResult
        return locals()[name]
    raise AttributeError(f"module 'src' has no attribute {name!r}")
