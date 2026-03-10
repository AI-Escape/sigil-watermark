"""Ghost signature spectral analysis.

Analyzes images for traces of an author's spectral fingerprint —
the "ghost" that propagates through AI model training.

The ghost signal uses multiplicative magnitude modulation at specific
frequency bands. Extraction uses spectral whitening (normalizing by
local magnitude) to detect the ±modulation pattern regardless of
image content. This is robust to natural images with complex spectra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigil_watermark.config import DEFAULT_CONFIG, SigilConfig
from sigil_watermark.keygen import build_ghost_composite_pn, get_ghost_hash_pns


@dataclass
class GhostAnalysisResult:
    """Result of ghost signature spectral analysis.

    Attributes:
        ghost_detected: ``True`` if a statistically significant ghost signal
            was found (correlation > 0.01 and p-value < 0.05).
        correlation: Normalized correlation between whitened spectrum and
            expected ghost PN pattern. Higher values indicate stronger signal.
        band_energies: Average spectral magnitude at each ghost frequency
            band, keyed by normalized frequency.
        p_value: Statistical significance under the null hypothesis of no
            watermark. Combined across channels via Fisher's method for
            RGB inputs.
        ghost_hash: Extracted ghost hash bits (blind, no author key needed),
            or ``None`` if extraction failed. Used for O(1) author lookup.
    """

    ghost_detected: bool
    correlation: float
    band_energies: dict[float, float]
    p_value: float
    ghost_hash: list[int] | None = None


def _compute_ghost_band_mask(
    h: int,
    w: int,
    config: SigilConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the combined ghost band mask for all bands."""
    cy, cx = h // 2, w // 2
    max_freq = min(h, w) // 2
    y, x = np.ogrid[:h, :w]
    freq_dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_freq

    combined = np.zeros((h, w))
    for band_freq in config.ghost_bands:
        combined += np.exp(-((freq_dist - band_freq) ** 2) / (2 * config.ghost_bandwidth**2))
    return combined, freq_dist


def _whiten_spectrum(f_shifted: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Whiten image spectrum by normalizing by locally-smoothed magnitude.

    Removes the image content's spectral shape (1/f law), leaving
    the multiplicative modulation pattern detectable.
    """
    from scipy.ndimage import uniform_filter

    magnitude = np.abs(f_shifted)
    mag_smooth = uniform_filter(magnitude, size=kernel_size)
    mag_smooth = np.maximum(mag_smooth, 1e-10)
    return magnitude / mag_smooth


def _correlate_pn_in_bands(
    f_shifted: np.ndarray,
    pn_1d: np.ndarray,
    band_mask: np.ndarray,
) -> float:
    """Correlate PN sign pattern with whitened magnitude in ghost bands.

    The ghost signal is embedded as multiplicative modulation:
    |F_embedded| = |F_original| * (1 + depth * pn_sign * band_mask).

    After whitening (dividing by local average magnitude), the modulation
    pattern becomes visible: whitened ≈ 1 + depth * pn_sign * band_mask.
    Correlating (whitened - 1) with pn_sign * band_mask recovers the sign.
    """
    h, w = f_shifted.shape

    # Whiten: normalize magnitude by local average
    whitened = _whiten_spectrum(f_shifted)

    # The modulation is in (whitened - 1). Extract within ghost bands.
    modulation = (whitened - 1.0) * band_mask

    # PN sign pattern (spatial domain PN → sign only)
    pn_2d = pn_1d[: h * w].reshape(h, w)
    pn_sign = np.sign(pn_2d)
    pn_band = pn_sign * band_mask

    # Correlation: how well does the modulation pattern match the PN sign?
    pn_energy = np.sum(pn_band**2)
    if pn_energy > 1e-10:
        return float(np.sum(modulation * pn_band) / pn_energy)
    return 0.0


def extract_ghost_hash(
    image: np.ndarray,
    config: SigilConfig = DEFAULT_CONFIG,
) -> tuple[list[int], list[float]]:
    """Extract ghost hash bits from an image (blind, no key needed).

    Detects multiplicative magnitude modulation at ghost frequency bands.
    Each hash bit's PN sign pattern produces a detectable modulation.
    Uses spectral whitening for robustness to natural image content.

    Args:
        image: Grayscale (H,W) or RGB (H,W,3) image.
        config: Sigil configuration.

    Returns:
        (hash_bits, confidences) where hash_bits is a list of 0/1 values
        and confidences is a list of absolute correlation magnitudes.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        # Multi-channel extraction with majority vote
        all_channel_bits = []
        all_channel_confs = []
        for ch in range(3):
            bits_ch, confs_ch = _extract_ghost_hash_single(image[:, :, ch], config)
            all_channel_bits.append(bits_ch)
            all_channel_confs.append(confs_ch)

        # Majority vote across channels
        bits = []
        confidences = []
        for i in range(config.ghost_hash_bits):
            votes = [cb[i] for cb in all_channel_bits]
            bits.append(1 if sum(votes) >= 2 else 0)
            confidences.append(max(cc[i] for cc in all_channel_confs))
        return bits, confidences

    return _extract_ghost_hash_single(image, config)


def _extract_ghost_hash_single(
    image: np.ndarray,
    config: SigilConfig,
) -> tuple[list[int], list[float]]:
    """Extract ghost hash from a single 2D channel."""
    h, w = image.shape
    f = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f)

    combined_mask, _ = _compute_ghost_band_mask(h, w, config)
    hash_pns = get_ghost_hash_pns(config.ghost_hash_bits, h * w, config)

    bits = []
    confidences = []
    for pn in hash_pns:
        corr = _correlate_pn_in_bands(f_shifted, pn, combined_mask)
        bits.append(1 if corr > 0 else 0)
        confidences.append(abs(corr))

    return bits, confidences


def analyze_ghost_signature(
    image: np.ndarray,
    public_key: bytes,
    config: SigilConfig = DEFAULT_CONFIG,
) -> GhostAnalysisResult:
    """Analyze a single image for ghost signature traces.

    Uses the composite ghost PN (encoding the author's ghost hash bits)
    to measure correlation. Also extracts the ghost hash bits blindly.
    Supports both grayscale (H,W) and RGB (H,W,3) input — RGB channels
    are analyzed independently and averaged for sqrt(3) SNR improvement.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        channel_results = [
            _analyze_ghost_single_channel(image[:, :, ch], public_key, config) for ch in range(3)
        ]
        avg_corr = np.mean([r.correlation for r in channel_results])
        avg_band_energies = {}
        for band in config.ghost_bands:
            avg_band_energies[band] = np.mean(
                [r.band_energies.get(band, 0) for r in channel_results]
            )
        p_values = [r.p_value for r in channel_results if r.p_value > 0]
        if p_values:
            from scipy.stats import chi2

            chi2_stat = -2 * sum(np.log(max(p, 1e-300)) for p in p_values)
            combined_p = float(1.0 - chi2.cdf(chi2_stat, df=2 * len(p_values)))
        else:
            combined_p = 1.0

        # Majority-vote ghost hash across channels
        all_channel_hashes = [r.ghost_hash for r in channel_results if r.ghost_hash]
        if all_channel_hashes:
            ghost_hash = []
            for bit_idx in range(config.ghost_hash_bits):
                votes = [h[bit_idx] for h in all_channel_hashes]
                ghost_hash.append(1 if sum(votes) > len(votes) / 2 else 0)
        else:
            ghost_hash = channel_results[0].ghost_hash

        ghost_detected = bool(avg_corr > 0.01 and combined_p < 0.05)
        return GhostAnalysisResult(
            ghost_detected=ghost_detected,
            correlation=float(avg_corr),
            band_energies=avg_band_energies,
            p_value=combined_p,
            ghost_hash=ghost_hash,
        )

    return _analyze_ghost_single_channel(image, public_key, config)


def _analyze_ghost_single_channel(
    image: np.ndarray,
    public_key: bytes,
    config: SigilConfig = DEFAULT_CONFIG,
) -> GhostAnalysisResult:
    """Analyze a single 2D channel for ghost signature traces."""
    h, w = image.shape
    f = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f)
    magnitude = np.abs(f_shifted)

    combined_mask, freq_dist = _compute_ghost_band_mask(h, w, config)

    # Correlate with the author's composite ghost PN
    composite_pn = build_ghost_composite_pn(public_key, h * w, config)
    composite_corr = _correlate_pn_in_bands(f_shifted, composite_pn, combined_mask)

    # Per-band energies
    band_energies = {}
    for band_freq in config.ghost_bands:
        band_mask = np.exp(-((freq_dist - band_freq) ** 2) / (2 * config.ghost_bandwidth**2))
        band_energy = np.sum(magnitude * band_mask) / max(np.sum(band_mask), 1e-10)
        band_energies[band_freq] = float(band_energy)

    # Extract ghost hash bits (blind)
    hash_pns = get_ghost_hash_pns(config.ghost_hash_bits, h * w, config)
    ghost_hash_bits = []
    for pn in hash_pns:
        corr = _correlate_pn_in_bands(f_shifted, pn, combined_mask)
        ghost_hash_bits.append(1 if corr > 0 else 0)

    # p-value for composite correlation
    n_bins = h * w
    null_std = 1.0 / np.sqrt(n_bins)
    if null_std > 0:
        z_score = composite_corr / null_std
        from scipy.stats import norm

        p_value = float(1.0 - norm.cdf(z_score))
    else:
        p_value = 1.0

    ghost_detected = composite_corr > 0.01 and p_value < 0.05

    return GhostAnalysisResult(
        ghost_detected=ghost_detected,
        correlation=float(composite_corr),
        band_energies=band_energies,
        p_value=p_value,
        ghost_hash=ghost_hash_bits,
    )


def batch_analyze_ghost(
    images: list[np.ndarray],
    public_key: bytes,
    config: SigilConfig = DEFAULT_CONFIG,
) -> GhostAnalysisResult:
    """Analyze multiple images for collective ghost signature.

    More reliable than single-image analysis because the ghost signal
    is consistent across all images from the same author, while noise
    averages out.
    """
    if not images:
        return GhostAnalysisResult(
            ghost_detected=False,
            correlation=0.0,
            band_energies={},
            p_value=1.0,
        )

    results = [analyze_ghost_signature(img, public_key, config) for img in images]

    # Average correlations and band energies
    avg_corr = np.mean([r.correlation for r in results])
    avg_band_energies = {}
    for band in config.ghost_bands:
        avg_band_energies[band] = np.mean([r.band_energies.get(band, 0) for r in results])

    # Fisher's method to combine p-values
    p_values = [r.p_value for r in results if r.p_value > 0]
    if p_values:
        from scipy.stats import chi2

        chi2_stat = -2 * sum(np.log(max(p, 1e-300)) for p in p_values)
        combined_p = float(1.0 - chi2.cdf(chi2_stat, df=2 * len(p_values)))
    else:
        combined_p = 1.0

    ghost_detected = bool(avg_corr > 0.005 and combined_p < 0.01)

    return GhostAnalysisResult(
        ghost_detected=ghost_detected,
        correlation=float(avg_corr),
        band_energies=avg_band_energies,
        p_value=combined_p,
    )
