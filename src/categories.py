"""
ADE20K category mappings — no torch dependency.
Imported by both segmentation.py and test suites.
"""
from __future__ import annotations
from typing import Dict, List

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
    "person":       [12, 92],
    "animal":       [126],
    "vehicle":      [20, 76, 80, 83, 90, 102, 103, 116, 127],
    "furniture":    [7, 10, 14, 15, 18, 19, 22, 23, 24, 27, 28, 30, 31, 33,
                     35, 37, 39, 44, 45, 47, 49, 50, 55, 56, 57, 62, 63, 64,
                     65, 67, 69, 70, 71, 73, 74, 75, 77, 81, 85, 88, 89, 97,
                     98, 99, 100, 107, 108, 110, 111, 112, 117, 118, 119, 120,
                     124, 125, 129, 130, 131, 132, 134, 135, 137, 138, 139,
                     141, 142, 143, 144, 145, 146, 147, 148],
    "architecture": [0, 1, 3, 5, 8, 11, 13, 25, 32, 34, 38, 40, 42, 43, 48,
                     52, 53, 54, 58, 59, 61, 82, 84, 86, 87, 93, 95, 96, 101,
                     104, 106, 121, 123, 133, 136, 140, 149],
    "nature":       [4, 6, 9, 16, 17, 21, 26, 29, 34, 46, 60, 66, 68, 72,
                     94, 105, 109, 113, 114, 128],
    "sky":          [2],
}

GROUP_DEPTH_PRIORITY: Dict[str, int] = {
    "person": 1, "animal": 2, "vehicle": 3, "furniture": 4,
    "architecture": 5, "nature": 6, "sky": 7, "background": 8,
}

_IDX_TO_GROUP: Dict[int, str] = {}
for _grp, _idxs in SEMANTIC_GROUPS.items():
    for _i in _idxs:
        _IDX_TO_GROUP[_i] = _grp


def ade_idx_to_group(idx: int) -> str:
    return _IDX_TO_GROUP.get(idx, "background")
