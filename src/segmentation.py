"""
Semantic Segmentation Module
============================
Uses SegFormer (B2/B5) fine-tuned on ADE20K-150 to produce
per-pixel semantic labels that are then merged into high-level
semantic groups (person, vehicle, animal, furniture, architecture,
nature, sky, background).

Model: nvidia/segformer-b2-finetuned-ade-512-512
       nvidia/segformer-b5-finetuned-ade-640-640   (higher quality)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from PIL import Image
try:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
except ImportError:
    SegformerForSemanticSegmentation = None
    SegformerImageProcessor = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ADE20K-150 → semantic-group mapping
# ---------------------------------------------------------------------------
# Maps ADE20K class index (0-149) to a high-level group label.
# Classes not listed fall into "background".

ADE20K_CLASS_NAMES: List[str] = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "computer",
    "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "television", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster",
    "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall",
    "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step",
    "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake",
    "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase",
    "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate",
    "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag",
]

SEMANTIC_GROUPS: Dict[str, List[int]] = {
    "person":       [12, 92],                           # person, apparel
    "animal":       [126],                               # animal (catch-all)
    "vehicle":      [20, 76, 80, 83, 90, 102, 103,
                     116, 127],                          # car, boat, bus, truck, airplane, van, ship, minibike, bicycle
    "furniture":    [7, 10, 14, 15, 18, 19, 22, 23,
                     24, 27, 28, 30, 31, 33, 35, 37,
                     39, 44, 45, 47, 49, 50, 55, 56,
                     57, 62, 63, 64, 65, 67, 69, 70,
                     71, 73, 74, 75, 77, 81, 85, 88,
                     89, 97, 98, 99, 100, 107, 108,
                     110, 111, 112, 117, 118, 119, 120,
                     124, 125, 129, 130, 131, 132, 134,
                     135, 137, 138, 139, 141, 142, 143,
                     144, 145, 146, 147, 148],
    "architecture": [0, 1, 3, 5, 8, 11, 13, 25, 32,
                     34, 38, 40, 42, 43, 48, 52, 53,
                     54, 58, 59, 61, 82, 84, 86, 87,
                     93, 95, 96, 101, 104, 106, 121,
                     123, 133, 136, 140, 149],
    "nature":       [4, 6, 9, 16, 17, 21, 26, 29,
                     34, 46, 60, 66, 68, 72, 94,
                     105, 109, 113, 114, 128],
    "sky":          [2],
}

# Depth priority for ordering layers (lower = nearer to camera)
GROUP_DEPTH_PRIORITY: Dict[str, int] = {
    "person":       1,
    "animal":       2,
    "vehicle":      3,
    "furniture":    4,
    "architecture": 5,
    "nature":       6,
    "sky":          7,
    "background":   8,
}

# Build reverse map: ADE20K class index → group name
_IDX_TO_GROUP: Dict[int, str] = {}
for _grp, _idxs in SEMANTIC_GROUPS.items():
    for _i in _idxs:
        _IDX_TO_GROUP[_i] = _grp


def ade_idx_to_group(idx: int) -> str:
    """Map ADE20K class index to semantic group name."""
    return _IDX_TO_GROUP.get(idx, "background")


@dataclass
class SegmentationResult:
    """Output of the segmentation stage."""
    raw_labels: np.ndarray           # H×W  uint8  — raw ADE20K class indices
    group_masks: Dict[str, np.ndarray]  # group_name → H×W bool mask
    group_names: List[str]           # ordered list of groups present
    confidence: np.ndarray           # H×W float32  — max softmax confidence


class SemanticSegmenter:
    """
    Wraps HuggingFace SegFormer for ADE20K-150 semantic segmentation.

    Parameters
    ----------
    model_name : str
        HuggingFace model id. Recommended variants:
        - "nvidia/segformer-b2-finetuned-ade-512-512"   (fast, ~25M params)
        - "nvidia/segformer-b5-finetuned-ade-640-640"   (accurate, ~85M params)
    device : str | None
        "cuda", "mps", "cpu" or None (auto-detect).
    min_mask_fraction : float
        Minimum fraction of image pixels a group must cover to be kept.
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        device: Optional[str] = None,
        min_mask_fraction: float = 0.001,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_mask_fraction = min_mask_fraction

        logger.info("Loading segmentation model: %s on %s", model_name, self.device)
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval().to(self.device)
        logger.info("Segmentation model loaded.")

    @torch.no_grad()
    def segment(self, image: Image.Image) -> SegmentationResult:
        """
        Run semantic segmentation on a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input RGB image (any resolution).

        Returns
        -------
        SegmentationResult
        """
        orig_w, orig_h = image.size

        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)

        # Upsample logits to original image size
        logits = outputs.logits  # B×C×H'×W'
        upsampled = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )  # B×C×H×W

        # Softmax → confidence & labels
        probs = upsampled.softmax(dim=1)[0]  # C×H×W
        confidence, raw_labels = probs.max(dim=0)
        raw_labels = raw_labels.cpu().numpy().astype(np.uint8)
        confidence = confidence.cpu().numpy().astype(np.float32)

        # Build group masks
        total_pixels = orig_h * orig_w
        group_masks: Dict[str, np.ndarray] = {}

        for group in list(SEMANTIC_GROUPS.keys()) + ["background"]:
            if group == "background":
                # Background = all pixels not claimed by any other group
                mask = np.ones((orig_h, orig_w), dtype=bool)
                for g, m in group_masks.items():
                    mask &= ~m
            else:
                indices = SEMANTIC_GROUPS[group]
                mask = np.isin(raw_labels, indices)

            if mask.sum() / total_pixels >= self.min_mask_fraction:
                group_masks[group] = mask

        # Sort groups by depth priority (nearest first)
        group_names = sorted(
            group_masks.keys(),
            key=lambda g: GROUP_DEPTH_PRIORITY.get(g, 99),
        )

        return SegmentationResult(
            raw_labels=raw_labels,
            group_masks=group_masks,
            group_names=group_names,
            confidence=confidence,
        )

    def get_colormap(self) -> Dict[str, Tuple[int, int, int]]:
        """Return a consistent RGB color per group for visualization."""
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
        return palette
