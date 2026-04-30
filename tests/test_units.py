"""
Unit Tests
===========
Tests for utility functions and metric computations.
Runs without downloading any large models.

  python -m pytest tests/test_units.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from PIL import Image

# ── Utility tests ────────────────────────────────────────────────────

class TestMetrics:
    def _dummy_image(self, h=64, w=64, c=3):
        return (np.random.rand(h, w, c) * 255).astype(np.uint8)

    def test_psnr_identical(self):
        from src.utils import psnr
        img = self._dummy_image()
        score = psnr(img, img)
        assert score == float("inf"), "PSNR of identical images should be inf"

    def test_psnr_range(self):
        from src.utils import psnr
        img1 = self._dummy_image()
        img2 = self._dummy_image()
        score = psnr(img1, img2)
        assert 0 < score < 60

    def test_ssim_identical(self):
        from src.utils import ssim
        img = self._dummy_image(c=1)[..., 0]
        score = ssim(img, img)
        assert score > 0.99, f"SSIM of identical images should be ~1, got {score}"

    def test_ssim_range(self):
        from src.utils import ssim
        img1 = self._dummy_image(c=1)[..., 0]
        img2 = self._dummy_image(c=1)[..., 0]
        score = ssim(img1, img2)
        assert -1.0 <= score <= 1.0


class TestCheckerboard:
    def test_checker_shape(self):
        from src.utils import _make_checker
        c = _make_checker(128, 96)
        assert c.size == (128, 96)
        assert c.mode == "RGB"


class TestIntrinsicRetinex:
    def _rgb_image(self, h=64, w=64):
        arr = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def test_retinex_shapes(self):
        from src.intrinsic import IntrinsicDecomposer
        dec = IntrinsicDecomposer(backend="retinex")
        img = self._rgb_image()
        result = dec.decompose(img)
        assert result.albedo.shape == (64, 64, 3), "Albedo shape mismatch"
        assert result.shading.shape == (64, 64), "Shading shape mismatch"

    def test_retinex_range(self):
        from src.intrinsic import IntrinsicDecomposer
        dec = IntrinsicDecomposer(backend="retinex")
        img = self._rgb_image()
        result = dec.decompose(img)
        assert result.albedo.min() >= 0.0
        assert result.albedo.max() <= 1.001
        assert result.shading.min() >= 0.0
        assert result.shading.max() <= 1.001

    def test_sparse_shapes(self):
        from src.intrinsic import IntrinsicDecomposer
        dec = IntrinsicDecomposer(backend="sparse")
        img = self._rgb_image()
        result = dec.decompose(img)
        assert result.albedo.shape == (64, 64, 3)
        assert result.shading.shape == (64, 64)

    def test_masked_decompose(self):
        from src.intrinsic import IntrinsicDecomposer
        dec = IntrinsicDecomposer(backend="retinex")
        img = self._rgb_image()
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:50, 10:50] = True
        result = dec.decompose(img, mask)
        # Outside mask should be 0
        assert result.albedo[~mask].max() == 0.0


class TestSegmentationGroupMapping:
    def test_known_indices(self):
        from src.categories import ade_idx_to_group, SEMANTIC_GROUPS
        # person = index 12
        assert ade_idx_to_group(12) == "person"
        # sky = index 2
        assert ade_idx_to_group(2) == "sky"
        # car = index 20
        assert ade_idx_to_group(20) == "vehicle"

    def test_background_fallback(self):
        from src.categories import ade_idx_to_group
        # Some indices not in any group → background
        assert ade_idx_to_group(999) == "background"

    def test_all_groups_have_indices(self):
        from src.categories import SEMANTIC_GROUPS
        for group, indices in SEMANTIC_GROUPS.items():
            assert isinstance(indices, list), f"Group {group} should have a list"


class TestDepthSpearman:
    def test_perfect_ordering(self):
        from benchmark.metrics import _depth_spearman

        class FakeMeta:
            def __init__(self, rank, depth):
                self.rank = rank
                self.median_depth = depth

        class FakeLayer:
            def __init__(self, rank, depth):
                self.meta = FakeMeta(rank, depth)

        layers = [FakeLayer(r, r * 0.1) for r in range(5)]
        rho = _depth_spearman(layers)
        assert rho > 0.99, f"Perfect ordering should give ρ≈1, got {rho}"

    def test_reversed_ordering(self):
        from benchmark.metrics import _depth_spearman

        class FakeMeta:
            def __init__(self, rank, depth):
                self.rank = rank
                self.median_depth = depth

        class FakeLayer:
            def __init__(self, rank, depth):
                self.meta = FakeMeta(rank, depth)

        # Reversed: rank 0 has highest depth — bad ordering
        layers = [FakeLayer(r, (4 - r) * 0.1) for r in range(5)]
        rho = _depth_spearman(layers)
        assert rho < -0.9, f"Reversed ordering should give ρ≈-1, got {rho}"


class TestCoverage:
    def test_full_coverage(self):
        from src.utils import coverage_score

        class FakeLayer:
            def __init__(self, alpha_mask):
                arr = np.zeros((10, 10, 4), dtype=np.uint8)
                arr[:, :, 3] = (alpha_mask * 255).astype(np.uint8)
                from PIL import Image
                self.rgba = Image.fromarray(arr, mode="RGBA")

        mask1 = np.ones((10, 10), dtype=bool)
        layers = [FakeLayer(mask1)]
        assert coverage_score(layers) == pytest.approx(1.0, abs=0.01)

    def test_partial_coverage(self):
        from src.utils import coverage_score

        class FakeLayer:
            def __init__(self, alpha_mask):
                arr = np.zeros((10, 10, 4), dtype=np.uint8)
                arr[:, :, 3] = (alpha_mask * 255).astype(np.uint8)
                from PIL import Image
                self.rgba = Image.fromarray(arr, mode="RGBA")

        mask_half = np.zeros((10, 10), dtype=bool)
        mask_half[:, :5] = True
        layers = [FakeLayer(mask_half)]
        score = coverage_score(layers)
        assert 0.45 < score < 0.55


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
