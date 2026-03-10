"""Perceptual masking for embedding strength adaptation.

Computes a per-pixel mask that scales watermark embedding strength based on
local image complexity. Textured/noisy regions can hide more watermark energy
without visible artifacts than flat/smooth regions.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from sigil_watermark.config import SigilConfig, DEFAULT_CONFIG


def compute_perceptual_mask(
    image: np.ndarray,
    config: SigilConfig = DEFAULT_CONFIG,
    block_size: int = 8,
) -> np.ndarray:
    """Compute a perceptual masking map based on local image activity.

    Uses local variance (in a sliding window) as a proxy for visual complexity.
    High-variance regions can tolerate stronger watermark embedding.

    Args:
        image: 2D grayscale image (float64, 0-255)
        config: Sigil configuration
        block_size: Size of the local analysis window

    Returns:
        2D mask array (same shape as image) with values in [mask_floor, 1.0].
    """
    # Compute local mean and variance
    local_mean = uniform_filter(image, size=block_size)
    local_sq_mean = uniform_filter(image ** 2, size=block_size)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)
    local_std = np.sqrt(local_var)

    # Normalize: map local_std to [0, 1] range
    # JND threshold: below this std, the region is "flat"
    # Above it, linearly scale to 1.0
    jnd = config.jnd_threshold
    normalized = np.clip((local_std - jnd) / (50.0 - jnd), 0, 1)

    # Apply floor and scale to [mask_floor, 1.0]
    mask = config.mask_floor + (1.0 - config.mask_floor) * normalized

    return mask
