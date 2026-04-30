"""
Intrinsic Image Decomposition Module
======================================
Decomposes a layer (or full image) into:
  - Albedo   (reflectance): the intrinsic colour, illumination-independent.
  - Shading  (illumination): the low-frequency illumination field.

Image model: I = A ⊙ S   (element-wise multiplication in linear light)

Two backends are provided:
  1. "retinex"   — Multi-Scale Retinex (MSR) — fast, training-free.
  2. "sparse"    — Weighted Least Squares smoothing (WLS) — better quality.
  3. "deep"      — Neural network (MIT Intrinsic Images / IID-Net style)
                   — best quality if available. Falls back to "sparse" if
                   the model cannot be downloaded.

Reference
---------
- Land & McCann (1971) "Lightness and Retinex Theory."
- Garces et al. (2012) "Intrinsic images by clustering."
- Grosse et al. (2009) MIT Intrinsic Images dataset.
- Bell et al. (2014) "Intrinsic Images in the Wild."
- Liu et al. (2020) "Unsupervised Intrinsic Image Decomposition (USI3D)."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class IntrinsicResult:
    """Output of intrinsic decomposition."""
    albedo: np.ndarray    # H×W×3 float32 in [0, 1]
    shading: np.ndarray   # H×W    float32 in [0, 1]  (grayscale luminance)
    reconstructed: np.ndarray  # A ⊙ S — should equal input


class IntrinsicDecomposer:
    """
    Intrinsic image decomposition into albedo and shading.

    Parameters
    ----------
    backend : str
        "retinex" | "sparse" | "deep"
    sigma_list : list[float]
        Gaussian sigmas used by Multi-Scale Retinex.
    wls_lambda : float
        Smoothness weight for WLS shading.
    wls_alpha : float
        Gradient sensitivity for WLS.
    """

    def __init__(
        self,
        backend: str = "sparse",
        sigma_list: Optional[list] = None,
        wls_lambda: float = 1.0,
        wls_alpha: float = 1.2,
    ) -> None:
        self.backend = backend
        self.sigma_list = sigma_list or [15.0, 80.0, 250.0]
        self.wls_lambda = wls_lambda
        self.wls_alpha = wls_alpha

        if backend == "deep":
            try:
                self._load_deep_model()
            except Exception as exc:
                logger.warning(
                    "Could not load deep intrinsic model (%s). "
                    "Falling back to 'sparse'.", exc
                )
                self.backend = "sparse"

    # ------------------------------------------------------------------
    # Deep model loader (optional)
    # ------------------------------------------------------------------

    def _load_deep_model(self) -> None:
        """
        Attempt to load a pretrained neural intrinsic model.
        Here we use a UNet-style encoder-decoder pretrained on Sintel/MIT.
        In production, replace with IID-Net or USI3D weights.
        """
        import torch
        from transformers import AutoModelForImageSegmentation  # placeholder
        raise NotImplementedError(
            "Deep backend not yet configured — set model weights path."
        )

    # ------------------------------------------------------------------
    # Multi-Scale Retinex
    # ------------------------------------------------------------------

    @staticmethod
    def _single_scale_retinex(img: np.ndarray, sigma: float) -> np.ndarray:
        """SSR in log domain: log(I) - log(GaussianBlur(I))."""
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        ssr = np.log1p(img) - np.log1p(blur)
        return ssr

    def _multi_scale_retinex(self, img: np.ndarray) -> np.ndarray:
        """MSR: equal-weight average of SSR over multiple scales."""
        msr = np.zeros_like(img, dtype=np.float32)
        for sigma in self.sigma_list:
            msr += self._single_scale_retinex(img.astype(np.float32), sigma)
        msr /= len(self.sigma_list)
        return msr

    def _retinex_decompose(self, img_linear: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retinex-based albedo/shading decomposition.
        img_linear: H×W×3 float32 in [0,1] (linear light).
        """
        eps = 1e-6
        img = img_linear.astype(np.float32) + eps

        # Compute log shading (illumination) from luminance channel
        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        log_shading = cv2.GaussianBlur(
            np.log(lum + eps), (0, 0), self.sigma_list[-1]
        )
        shading = np.exp(log_shading)
        shading = np.clip(shading, eps, None)

        # Albedo = Image / Shading (broadcast over channels)
        albedo = img / shading[..., np.newaxis]

        # Normalise
        shading = (shading - shading.min()) / (shading.max() - shading.min() + eps)
        albedo  = np.clip(albedo, 0, 1)
        albedo  = albedo / (albedo.max() + eps)

        return albedo, shading

    # ------------------------------------------------------------------
    # Weighted Least Squares (sparse) decomposition
    # ------------------------------------------------------------------

    @staticmethod
    def _wls_filter(
        guide: np.ndarray,
        src: np.ndarray,
        lam: float,
        alpha: float,
    ) -> np.ndarray:
        """
        Fast WLS smoothing via OpenCV's ximgproc (edge-preserving filter).
        If ximgproc is unavailable, falls back to Gaussian blur.
        """
        try:
            import cv2.ximgproc as xip
            wls = xip.createFastGlobalSmootherFilter(
                guide, lam, alpha
            )
            return wls.filter(src)
        except (ImportError, AttributeError):
            logger.debug("cv2.ximgproc not available — using GaussianBlur fallback.")
            ksize = int(6 * alpha) | 1  # ensure odd
            return cv2.GaussianBlur(src, (ksize, ksize), 0)

    def _sparse_decompose(self, img_linear: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        WLS-based intrinsic decomposition.
        Shading = WLS-smoothed log-luminance.
        Albedo  = Image / Shading.
        """
        eps = 1e-6
        img = img_linear.astype(np.float32) + eps

        lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        log_lum = np.log(lum)

        # WLS smoothed log luminance → log shading
        guide = (lum * 255).astype(np.uint8)
        log_shading = self._wls_filter(
            guide, log_lum, self.wls_lambda, self.wls_alpha
        )
        shading = np.exp(log_shading)
        shading = np.clip(shading, eps, None)

        albedo = img / shading[..., np.newaxis]
        albedo = np.clip(albedo, 0, 1)

        sh_min, sh_max = shading.min(), shading.max()
        shading = (shading - sh_min) / (sh_max - sh_min + eps)

        return albedo.astype(np.float32), shading.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, image: Image.Image, mask: Optional[np.ndarray] = None) -> IntrinsicResult:
        """
        Decompose a PIL image (or masked region) into albedo and shading.

        Parameters
        ----------
        image : PIL.Image.Image
            Input RGB image.
        mask : np.ndarray | None
            Boolean H×W mask. If provided, decomposition is performed only
            inside the mask region; outside pixels are left as zeros.

        Returns
        -------
        IntrinsicResult
        """
        # Convert to float32 linear light (approximate gamma decode)
        img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        img_linear = np.power(img_np, 2.2)  # sRGB → linear approx

        if mask is not None:
            # Decompose only in masked region
            img_to_decompose = img_linear * mask[..., np.newaxis]
        else:
            img_to_decompose = img_linear

        if self.backend == "retinex":
            albedo, shading = self._retinex_decompose(img_to_decompose)
        else:  # sparse or fallback
            albedo, shading = self._sparse_decompose(img_to_decompose)

        if mask is not None:
            albedo  = albedo  * mask[..., np.newaxis]
            shading = shading * mask

        # Reconstruct for quality check
        reconstructed = albedo * shading[..., np.newaxis]
        reconstructed = np.clip(reconstructed, 0, 1)

        return IntrinsicResult(
            albedo=albedo,
            shading=shading,
            reconstructed=reconstructed,
        )

    @staticmethod
    def reconstruction_error(result: IntrinsicResult, original: np.ndarray) -> float:
        """
        Mean absolute error between A⊙S and the original image.
        original: H×W×3 float32 in [0,1].
        """
        diff = np.abs(result.reconstructed - original)
        return float(diff.mean())
