"""Signal processing transforms for the Sigil watermark system.

DFT ring template embedding/detection, DWT decomposition, spread-spectrum
modulation, and geometric correction utilities.
"""

from __future__ import annotations

import cv2
import numpy as np
import pywt

# --- DFT Ring Template (Layer 1: Geometric Compass) ---


def embed_dft_rings(
    image: np.ndarray,
    radii: np.ndarray,
    strength: float = 50.0,
    ring_width: float = 0.04,
    phase_offsets: np.ndarray | None = None,
    target_psnr: float | None = None,
    min_alpha_fraction: float = 0.05,
) -> np.ndarray:
    """Embed concentric ring peaks into the DFT spectrum.

    Uses multiplicative magnitude embedding with optional phase modulation.
    Multiplicative scaling ties the watermark to local spectrum energy,
    making removal via simple notch filtering destroy image content.
    Phase offsets add key-dependent information that pure magnitude
    analysis cannot recover.

    Args:
        image: 2D grayscale image (float64, 0-255)
        radii: Array of ring radii as fractions of image_size/2
        strength: Embedding strength (controls the multiplicative boost)
        ring_width: Gaussian sigma of ring profiles (fraction of Nyquist)
        phase_offsets: Optional per-ring phase offsets in radians.
            If provided, must have same length as radii.
        target_psnr: If set, adaptively scale alpha so the ring layer's
            estimated PSNR stays above this target (dB).  Uses Parseval's
            theorem for exact MSE prediction.  None disables adaptation.
        min_alpha_fraction: Floor for adaptive scaling — alpha never drops
            below this fraction of the nominal value.

    Returns:
        Watermarked image with rings in its Fourier spectrum.
    """
    h, w = image.shape
    f = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f)

    magnitude = np.abs(f_shifted)
    phase = np.angle(f_shifted)

    cy, cx = h // 2, w // 2
    half = min(h, w) // 2

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / half

    # Scale strength relative to the median magnitude in mid-frequency ring region
    mid_mask = (dist > 0.05) & (dist < 0.45)
    if mid_mask.any():
        median_mag = np.median(magnitude[mid_mask])
    else:
        median_mag = np.median(magnitude)

    # Multiplicative with soft ceiling: scales with local energy (makes notch
    # filtering harder) but caps at 2× median to prevent runaway distortion
    # on high-energy spectrum peaks.
    # Scale per-ring alpha by 4/num_rings so total energy stays constant
    # regardless of how many rings are embedded.
    num_rings = len(radii)
    base_alpha = (strength / 50.0) * (4.0 / max(num_rings, 1))
    capped_mag = np.minimum(magnitude, median_mag * 2.0)

    # Adaptive ring strength: predict pixel-domain MSE via Parseval's theorem
    # and scale alpha down if it would exceed the target PSNR.
    alpha = base_alpha
    if target_psnr is not None:
        # Compute total energy the rings would add at base_alpha.
        # delta_F = alpha * ring_mask * capped_mag for each ring.
        # By Parseval: MSE = sum(|delta_F|^2) / (h * w).
        sum_sq = 0.0
        for r in radii:
            ring_mask = np.exp(-((dist - r) ** 2) / (2 * ring_width**2))
            delta = ring_mask * capped_mag
            sum_sq += np.sum(delta**2)

        if sum_sq > 0:
            estimated_mse = (base_alpha**2) * sum_sq / (h * w)
            if estimated_mse > 0:
                estimated_psnr = 10.0 * np.log10(255.0**2 / estimated_mse)
                if estimated_psnr < target_psnr:
                    target_mse = 255.0**2 / (10.0 ** (target_psnr / 10.0))
                    max_alpha = np.sqrt(target_mse * h * w / sum_sq)
                    alpha = max(max_alpha, base_alpha * min_alpha_fraction)

    for i, r in enumerate(radii):
        ring_mask = np.exp(-((dist - r) ** 2) / (2 * ring_width**2))
        magnitude += alpha * ring_mask * capped_mag

        # Phase modulation: small rotation at ring frequencies by key-derived offset.
        # Limited to ±0.3 radians (~17°) to avoid destructive interference
        # with existing signal while still being detectable.
        if phase_offsets is not None and i < len(phase_offsets):
            # Map [0, 2π) → [-0.3, 0.3] radians
            scaled_offset = (phase_offsets[i] / np.pi - 1.0) * 0.3
            phase += scaled_offset * ring_mask

    f_modified = magnitude * np.exp(1j * phase)
    f_unshifted = np.fft.ifftshift(f_modified)
    result = np.real(np.fft.ifft2(f_unshifted))

    return result


def detect_dft_rings(
    image: np.ndarray,
    expected_radii: np.ndarray,
    tolerance: float = 0.02,
    ring_width: float = 0.04,
    phase_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Detect ring peaks in the DFT spectrum using radial profile analysis.

    Computes a dense radial magnitude profile (mean magnitude per thin
    annular bin), fits a smooth polynomial baseline through bins far from
    expected ring radii, then measures the excess at ring locations. This
    approach is robust to natural 1/f spectral slope in real photographs.

    Args:
        image: 2D grayscale image
        expected_radii: Array of expected ring radii (fractions of size/2)
        tolerance: Radius matching tolerance
        ring_width: Gaussian sigma of ring profiles (must match embedding)
        phase_offsets: Optional expected phase offsets per ring (for verification)

    Returns:
        (detected_radii, confidence) where confidence is 0-1.
    """
    h, w = image.shape
    f = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f)
    magnitude = np.abs(f_shifted)

    cy, cx = h // 2, w // 2
    half = min(h, w) // 2

    y_coords = np.arange(h).reshape(-1, 1)
    x_coords = np.arange(w).reshape(1, -1)
    dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / half

    # Spectral whitening: divide the 2D magnitude by a smooth radial
    # model to remove the natural 1/f slope.  The ring signal (a ~20%
    # multiplicative boost at specific radii) becomes detectable via NCC
    # on the whitened spectrum.
    #
    # The radial model is fit to ALL radial bins (including ring locations)
    # using a low-degree polynomial in log-log space.  Since the polynomial
    # can't fit the localized ring bumps, it averages through them — the
    # excess at ring radii is preserved in the whitened spectrum.

    num_bins = max(200, half)
    bin_edges = np.linspace(0.02, 0.95, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    profile = np.zeros(num_bins)
    valid = np.zeros(num_bins, dtype=bool)
    for i in range(num_bins):
        mask = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1])
        if mask.sum() > 0:
            profile[i] = magnitude[mask].mean()
            valid[i] = True

    if valid.sum() < 6:
        mag_confidence = 0.0
    else:
        # Fit degree-2 polynomial in log-log to ALL valid bins.
        # Low degree ensures the polynomial smooths through ring peaks.
        log_r = np.log(bin_centers[valid])
        log_m = np.log(profile[valid] + 1.0)
        coeffs = np.polyfit(log_r, log_m, 2)
        log_dist = np.log(np.maximum(dist, 0.01))
        log_model = np.clip(np.polyval(coeffs, log_dist), -20.0, 40.0)
        radial_model = np.maximum(np.exp(log_model), 1.0)

        whitened = magnitude / radial_model

        # NCC between whitened spectrum and ring template
        expected_template = np.zeros_like(dist)
        for r in expected_radii:
            expected_template += np.exp(-((dist - r) ** 2) / (2 * ring_width**2))

        region_mask = (
            (dist > expected_radii.min() - 0.05) & (dist < expected_radii.max() + 0.05)
        ).flatten()

        if region_mask.sum() < 10:
            mag_confidence = 0.0
        else:
            t_region = expected_template.flatten()[region_mask]
            w_region = whitened.flatten()[region_mask]

            t_centered = t_region - t_region.mean()
            w_centered = w_region - w_region.mean()

            t_norm = np.linalg.norm(t_centered)
            w_norm = np.linalg.norm(w_centered)

            if t_norm < 1e-10 or w_norm < 1e-10:
                mag_confidence = 0.0
            else:
                ncc = np.dot(t_centered, w_centered) / (t_norm * w_norm)
                # Calibrated threshold: unwatermarked images typically
                # produce NCC < 0.05.  Watermarked images produce 0.06-0.28.
                # Map so 0.06 → 0.5, 0.12 → 1.0.
                mag_confidence = min(1.0, max(0.0, (ncc - 0.03) / 0.09))

    # Phase modulation serves a security purpose (key-dependent embedding)
    # but phase coherence at ring locations is too noisy to contribute to
    # detection confidence — measured ~0.01 even on watermarked images,
    # which dilutes the magnitude-based NCC by ~30%.  Use magnitude only.
    confidence = mag_confidence

    return np.array(expected_radii), confidence


def estimate_rotation_from_rings(
    image: np.ndarray,
    expected_radii: np.ndarray,
) -> float:
    """Estimate rotation angle from ring template angular analysis.

    For now, returns 0.0 — ring templates are rotationally symmetric,
    so rotation estimation requires additional angular markers.
    This is a placeholder for the angular analysis component.
    """
    # Rings alone are rotation-invariant (that's the point).
    # Rotation estimation requires either:
    # - Angular markers added to the rings (future)
    # - Cross-correlation with known angular pattern
    # For now, handle 90° multiples only via brute force in the detector.
    return 0.0


# --- DWT Decomposition (Layer 2 foundation) ---


def dwt_decompose(
    image: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> list:
    """Decompose image using multi-level 2D DWT.

    Returns pywt coefficient list: [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
    """
    return pywt.wavedec2(image, wavelet=wavelet, level=level)


def dwt_reconstruct(
    coeffs: list,
    wavelet: str = "db4",
) -> np.ndarray:
    """Reconstruct image from DWT coefficients."""
    return pywt.waverec2(coeffs, wavelet=wavelet)


# --- Spread-Spectrum Embedding (Layer 2: Payload) ---


def embed_spread_spectrum(
    coeffs_2d: np.ndarray,
    pn_sequence: np.ndarray,
    payload_bits: list[int],
    strength: float = 5.0,
    spreading_factor: int = 256,
) -> np.ndarray:
    """Embed payload bits into 2D coefficient array using CDMA-style spread spectrum.

    Each payload bit is spread across `spreading_factor` chips of the PN sequence,
    then added to the coefficient array.

    Args:
        coeffs_2d: 2D array of DWT coefficients to embed into
        pn_sequence: Bipolar (+1/-1) PN sequence, length >= coeffs_2d.size
        payload_bits: List of payload bits (0 or 1)
        strength: Embedding strength multiplier
        spreading_factor: Number of PN chips per payload bit

    Returns:
        Modified coefficient array with payload embedded.
    """
    flat = coeffs_2d.flatten().copy()
    pn = pn_sequence[: len(flat)]

    num_bits = len(payload_bits)
    total_chips = num_bits * spreading_factor

    if total_chips > len(flat):
        # Reduce spreading factor to fit
        spreading_factor = len(flat) // num_bits
        total_chips = num_bits * spreading_factor

    if spreading_factor < 1:
        raise ValueError("Coefficient array too small for payload")

    # Build the spread signal: each bit modulates `spreading_factor` chips
    spread_signal = np.zeros(total_chips)
    for i, bit in enumerate(payload_bits):
        bipolar_bit = 2.0 * bit - 1.0  # 0 -> -1, 1 -> +1
        start = i * spreading_factor
        end = start + spreading_factor
        spread_signal[start:end] = bipolar_bit * pn[start:end]

    # Add to coefficients
    flat[:total_chips] += strength * spread_signal

    return flat.reshape(coeffs_2d.shape)


def extract_spread_spectrum(
    coeffs_2d: np.ndarray,
    pn_sequence: np.ndarray,
    num_bits: int,
    spreading_factor: int = 256,
) -> list[int]:
    """Extract payload bits from coefficient array using spread-spectrum correlation.

    Args:
        coeffs_2d: 2D array of (watermarked) DWT coefficients
        pn_sequence: Same bipolar PN sequence used for embedding
        num_bits: Number of payload bits to extract
        spreading_factor: Same spreading factor used for embedding

    Returns:
        List of extracted payload bits (0 or 1).
    """
    flat = coeffs_2d.flatten()
    pn = pn_sequence[: len(flat)]

    total_chips = num_bits * spreading_factor
    if total_chips > len(flat):
        spreading_factor = len(flat) // num_bits
        total_chips = num_bits * spreading_factor

    bits = []
    for i in range(num_bits):
        start = i * spreading_factor
        end = start + spreading_factor
        # Correlate: dot product of coefficients with PN chips for this bit
        correlation = np.dot(flat[start:end], pn[start:end])
        bits.append(1 if correlation > 0 else 0)

    return bits


# --- Geometric Correction ---


def apply_geometric_correction(
    image: np.ndarray,
    angle: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Apply rotation and scale correction to an image.

    Args:
        image: 2D grayscale image
        angle: Rotation angle in degrees (counter-clockwise positive)
        scale: Scale factor

    Returns:
        Corrected image.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    corrected = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return corrected
