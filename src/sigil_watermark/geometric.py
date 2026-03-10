"""Geometric auto-correction using Fourier-Mellin Transform.

Estimates rotation and scale from the DFT magnitude spectrum using
log-polar phase correlation. This replaces the stub in transforms.py.
"""

from __future__ import annotations

import numpy as np
import cv2


def _highpass_filter(image: np.ndarray) -> np.ndarray:
    """Apply a highpass filter to suppress DC and low-frequency energy."""
    h, w = image.shape
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # Cosine highpass: suppress DC, transition from 0 to 1
    filt = 1.0 - np.cos(np.pi * np.minimum(r / (min(h, w) / 4), 1.0))
    return filt


def _log_polar_transform(magnitude: np.ndarray) -> np.ndarray:
    """Convert DFT magnitude to log-polar coordinates.

    Rotation in spatial domain → shift in angle axis.
    Scale in spatial domain → shift in log-radius axis.
    """
    h, w = magnitude.shape
    center = (w / 2, h / 2)
    max_radius = min(h, w) / 2

    # Use OpenCV's log-polar transform
    # flags: WARP_FILL_OUTLIERS + INTER_LINEAR
    log_polar = cv2.logPolar(
        magnitude.astype(np.float32),
        center,
        max_radius / np.log(max_radius),
        cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR,
    )
    return log_polar.astype(np.float64)


def _phase_correlation(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float, float]:
    """Phase correlation between two images.

    Returns:
        (shift_x, shift_y, peak_value)
    """
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)

    # Cross-power spectrum
    cross = f1 * np.conj(f2)
    cross_mag = np.abs(cross)
    cross_mag[cross_mag < 1e-10] = 1e-10
    cross_norm = cross / cross_mag

    correlation = np.real(np.fft.ifft2(cross_norm))

    # Find the peak
    peak_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak_val = correlation[peak_idx]

    # Convert to shift (handle wrap-around)
    h, w = correlation.shape
    sy, sx = peak_idx
    if sy > h / 2:
        sy -= h
    if sx > w / 2:
        sx -= w

    return float(sx), float(sy), float(peak_val)


def estimate_rotation_scale(
    image: np.ndarray,
    reference: np.ndarray | None = None,
) -> tuple[float, float]:
    """Estimate rotation and scale between image and reference.

    Uses the Fourier-Mellin Transform:
    1. Compute DFT magnitude of both images
    2. Apply highpass filter to suppress DC
    3. Convert to log-polar coordinates
    4. Phase correlate → (rotation, scale)

    Args:
        image: Test image (possibly rotated/scaled).
        reference: Reference image. If None, uses the DFT ring pattern
                   as an implicit reference (self-referencing via symmetry).

    Returns:
        (angle_degrees, scale_factor)
    """
    h, w = image.shape[:2]

    # Compute DFT magnitude
    mag_test = np.abs(np.fft.fftshift(np.fft.fft2(image)))

    if reference is not None:
        mag_ref = np.abs(np.fft.fftshift(np.fft.fft2(reference)))
    else:
        # Without a reference, we can't estimate. Return identity.
        return 0.0, 1.0

    # Make both the same size
    if mag_test.shape != mag_ref.shape:
        mag_test = cv2.resize(mag_test.astype(np.float32),
                              (mag_ref.shape[1], mag_ref.shape[0])).astype(np.float64)

    # Highpass filter to suppress DC
    hpf = _highpass_filter(mag_ref)
    mag_test_hp = mag_test * hpf
    mag_ref_hp = mag_ref * hpf

    # Log-polar transform
    lp_test = _log_polar_transform(mag_test_hp)
    lp_ref = _log_polar_transform(mag_ref_hp)

    # Phase correlation in log-polar domain
    shift_x, shift_y, peak_val = _phase_correlation(lp_ref, lp_test)

    # Convert shifts to rotation and scale
    lp_h, lp_w = lp_ref.shape
    max_radius = min(mag_ref.shape) / 2

    # Angle: shift_y corresponds to angular shift
    # Log-polar maps angle 0..360 to rows 0..h
    angle = -shift_y * 360.0 / lp_h

    # Scale: shift_x corresponds to log-radius shift
    scale = np.exp(shift_x * np.log(max_radius) / lp_w)

    return angle, scale


def auto_correct(
    image: np.ndarray,
    angle: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Apply inverse geometric correction to an image.

    Args:
        image: Image to correct (2D).
        angle: Estimated rotation angle in degrees.
        scale: Estimated scale factor.

    Returns:
        Corrected image.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    # Apply inverse: rotate back and rescale
    M = cv2.getRotationMatrix2D(center, -angle, 1.0 / scale)
    corrected = cv2.warpAffine(
        image.astype(np.float32), M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return corrected.astype(np.float64)


def try_rotations(
    image: np.ndarray,
    detect_fn,
    angles: list[float] | None = None,
) -> tuple[np.ndarray, float, float]:
    """Try multiple rotation corrections and return the best one.

    Brute-force approach for when we don't have a reference image.
    Tries common rotation angles and picks the one that gives the
    best detection confidence.

    Args:
        image: Image to correct.
        detect_fn: Callable(image) -> confidence_score.
        angles: Angles to try (degrees). Defaults to common rotations.

    Returns:
        (best_image, best_angle, best_confidence)
    """
    if angles is None:
        angles = [0, 90, 180, 270, 1, -1, 2, -2, 5, -5]

    best_img = image
    best_angle = 0.0
    best_conf = detect_fn(image)

    for angle in angles:
        if angle == 0:
            continue
        corrected = auto_correct(image, angle=angle)
        conf = detect_fn(corrected)
        if conf > best_conf:
            best_conf = conf
            best_img = corrected
            best_angle = angle

    return best_img, best_angle, best_conf
