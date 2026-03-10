"""Color space conversion for RGB/YCbCr watermark embedding.

The watermark is embedded in the Y (luminance) channel only, which:
- Survives JPEG compression (Cb/Cr are 4:2:0 subsampled)
- Survives print-scan (luminance is preserved better than chroma)
- Survives color shifts (hue, saturation changes don't affect Y much)
- Is backward-compatible with grayscale input
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ColorMetadata:
    """Metadata for reconstructing the original color format after embedding."""

    is_color: bool
    original_shape: tuple[int, ...]
    cb_channel: np.ndarray | None = None
    cr_channel: np.ndarray | None = None


def rgb_to_ycbcr(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to YCbCr color space.

    Uses ITU-R BT.601 conversion matrix (same as JPEG).

    Args:
        image_rgb: (H, W, 3) float64 array in [0, 255].

    Returns:
        (H, W, 3) float64 array: Y in [0, 255], Cb/Cr in [0, 255].
    """
    r, g, b = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def ycbcr_to_rgb(image_ycbcr: np.ndarray) -> np.ndarray:
    """Convert YCbCr image back to RGB color space.

    Args:
        image_ycbcr: (H, W, 3) float64 array.

    Returns:
        (H, W, 3) float64 array clipped to [0, 255].
    """
    y, cb, cr = image_ycbcr[:, :, 0], image_ycbcr[:, :, 1], image_ycbcr[:, :, 2]
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    return np.clip(np.stack([r, g, b], axis=-1), 0, 255)


def prepare_for_embedding(image: np.ndarray) -> tuple[np.ndarray, ColorMetadata]:
    """Extract the Y (luminance) channel for watermark embedding.

    Handles both grayscale (H, W) and RGB (H, W, 3) inputs.

    Args:
        image: Grayscale (H, W) or RGB (H, W, 3) float64 array.

    Returns:
        (y_channel, metadata) where y_channel is 2D float64 and metadata
        allows reconstruction via reconstruct_from_embedding().
    """
    if image.ndim == 2:
        return image, ColorMetadata(
            is_color=False,
            original_shape=image.shape,
        )

    if image.ndim == 3 and image.shape[2] == 3:
        ycbcr = rgb_to_ycbcr(image)
        return ycbcr[:, :, 0], ColorMetadata(
            is_color=True,
            original_shape=image.shape,
            cb_channel=ycbcr[:, :, 1],
            cr_channel=ycbcr[:, :, 2],
        )

    raise ValueError(f"Unsupported image shape: {image.shape}")


def reconstruct_from_embedding(y_watermarked: np.ndarray, metadata: ColorMetadata) -> np.ndarray:
    """Recombine watermarked Y channel with original Cb/Cr.

    Args:
        y_watermarked: 2D watermarked luminance channel.
        metadata: ColorMetadata from prepare_for_embedding().

    Returns:
        Reconstructed image in the same format as the original input.
    """
    if not metadata.is_color:
        return y_watermarked

    assert metadata.cb_channel is not None and metadata.cr_channel is not None
    ycbcr = np.stack([y_watermarked, metadata.cb_channel, metadata.cr_channel], axis=-1)
    return ycbcr_to_rgb(ycbcr)


def extract_y_channel(image: np.ndarray) -> np.ndarray:
    """Extract Y channel for detection. Handles both grayscale and RGB."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return rgb_to_ycbcr(image)[:, :, 0]
    raise ValueError(f"Unsupported image shape: {image.shape}")
