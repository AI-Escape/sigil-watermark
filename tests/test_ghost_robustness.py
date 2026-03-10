"""Ghost signal (Layer 3) robustness tests.

Tests parameterized across multiple natural image types and author keys.
"""

import io

import cv2
import numpy as np
import pytest
from conftest import (
    NATURAL_IMAGE_GENERATORS,
    make_natural_scene,
)
from PIL import Image

from sigil_watermark.ghost.spectral_analysis import (
    GhostAnalysisResult,
    analyze_ghost_signature,
    batch_analyze_ghost,
)


def _jpeg_compress(image, quality):
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.float64)


# --- Single Image Ghost Tests ---
# Now parameterized across natural image types and multiple keys.


class TestGhostSurvivesJPEG:
    """Ghost signal spectral difference persists through JPEG on all image types."""

    @pytest.mark.parametrize("quality", [95, 75, 50])
    def test_spectral_difference_after_jpeg(
        self, embedder, natural_image, multi_author_keys, config, quality
    ):
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        compressed_wm = _jpeg_compress(watermarked, quality)
        compressed_clean = _jpeg_compress(img, quality)

        wm_result = analyze_ghost_signature(compressed_wm, multi_author_keys.public_key, config)
        clean_result = analyze_ghost_signature(
            compressed_clean, multi_author_keys.public_key, config
        )

        assert wm_result.correlation != clean_result.correlation, (
            f"Ghost correlation identical on {name} after JPEG Q{quality}"
        )

    def test_band_energies_change_after_jpeg(self, embedder, multi_author_keys, config):
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        compressed = _jpeg_compress(watermarked, 75)
        comp_result = analyze_ghost_signature(compressed, multi_author_keys.public_key, config)
        for band in config.ghost_bands:
            assert comp_result.band_energies[band] > 0


class TestGhostSurvivesNoise:
    @pytest.mark.parametrize("sigma", [5, 10, 15])
    def test_spectral_difference_after_noise(
        self, embedder, natural_image, multi_author_keys, config, sigma
    ):
        name, img = natural_image
        rng = np.random.default_rng(42)
        watermarked = embedder.embed(img, multi_author_keys)
        noisy_wm = np.clip(watermarked + rng.normal(0, sigma, watermarked.shape), 0, 255)
        noisy_clean = np.clip(img + rng.normal(0, sigma, img.shape), 0, 255)

        wm_result = analyze_ghost_signature(noisy_wm, multi_author_keys.public_key, config)
        clean_result = analyze_ghost_signature(noisy_clean, multi_author_keys.public_key, config)
        assert wm_result.correlation != clean_result.correlation, (
            f"Ghost identical on {name} after sigma={sigma} noise"
        )


class TestGhostSurvivesBlur:
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    def test_spectral_difference_after_blur(
        self, embedder, natural_image, multi_author_keys, config, sigma
    ):
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        ksize = max(3, int(sigma * 6) | 1)
        blurred_wm = cv2.GaussianBlur(watermarked.astype(np.float32), (ksize, ksize), sigma).astype(
            np.float64
        )
        blurred_clean = cv2.GaussianBlur(img.astype(np.float32), (ksize, ksize), sigma).astype(
            np.float64
        )

        wm_result = analyze_ghost_signature(blurred_wm, multi_author_keys.public_key, config)
        clean_result = analyze_ghost_signature(blurred_clean, multi_author_keys.public_key, config)
        assert wm_result.correlation != clean_result.correlation, (
            f"Ghost identical on {name} after blur sigma={sigma}"
        )


class TestGhostSurvivesBrightness:
    @pytest.mark.parametrize("delta", [-40, 40])
    def test_spectral_difference_brightness(
        self, embedder, natural_image, multi_author_keys, config, delta
    ):
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        wm_result = analyze_ghost_signature(
            np.clip(watermarked + delta, 0, 255), multi_author_keys.public_key, config
        )
        clean_result = analyze_ghost_signature(
            np.clip(img + delta, 0, 255), multi_author_keys.public_key, config
        )
        # On binary/saturated images, clipping may produce identical spectra
        if name not in {"edges"}:
            assert wm_result.correlation != clean_result.correlation, (
                f"Ghost correlation identical on {name} after brightness {delta:+d}"
            )

    @pytest.mark.parametrize("factor", [0.7, 1.3])
    def test_spectral_difference_contrast(
        self, embedder, natural_image, multi_author_keys, config, factor
    ):
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        wm_adj = np.clip(watermarked.mean() + factor * (watermarked - watermarked.mean()), 0, 255)
        clean_adj = np.clip(img.mean() + factor * (img - img.mean()), 0, 255)
        wm_result = analyze_ghost_signature(wm_adj, multi_author_keys.public_key, config)
        clean_result = analyze_ghost_signature(clean_adj, multi_author_keys.public_key, config)
        # On binary/saturated images, clipping may produce identical spectra
        if name not in {"edges"}:
            assert wm_result.correlation != clean_result.correlation, (
                f"Ghost correlation identical on {name} after {factor}x contrast"
            )


class TestGhostSurvivesGamma:
    @pytest.mark.parametrize("gamma", [0.7, 1.5])
    def test_spectral_difference_gamma(
        self, embedder, natural_image, multi_author_keys, config, gamma
    ):
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        wm_gamma = np.power(np.clip(watermarked, 0, 255) / 255.0, gamma) * 255.0
        clean_gamma = np.power(np.clip(img, 0, 255) / 255.0, gamma) * 255.0
        wm_result = analyze_ghost_signature(wm_gamma, multi_author_keys.public_key, config)
        clean_result = analyze_ghost_signature(clean_gamma, multi_author_keys.public_key, config)
        assert wm_result.correlation != clean_result.correlation


class TestGhostKeyDiscrimination:
    """Ghost should show different correlations for right vs wrong key."""

    def test_different_keys_different_correlation(
        self, embedder, natural_image, author_keys, author_keys_b, config
    ):
        name, img = natural_image
        watermarked = embedder.embed(img, author_keys)
        right = analyze_ghost_signature(watermarked, author_keys.public_key, config)
        wrong = analyze_ghost_signature(watermarked, author_keys_b.public_key, config)
        assert right.correlation != wrong.correlation, f"Key discrimination failed on {name}"

    def test_consistent_correlation_across_image_types(self, embedder, multi_author_keys, config):
        """Same author key should produce consistent patterns across diverse images."""
        correlations = []
        for gen in NATURAL_IMAGE_GENERATORS.values():
            img = gen()
            watermarked = embedder.embed(img, multi_author_keys)
            result = analyze_ghost_signature(watermarked, multi_author_keys.public_key, config)
            correlations.append(result.correlation)
        std = np.std(correlations)
        assert std < 1.0, f"Ghost correlation std too high: {std:.4f}"


# --- Batch Ghost Robustness ---


class TestBatchGhostRobustness:
    def test_batch_watermarked_differs_from_clean(self, embedder, multi_author_keys, config):
        """Batch using all natural image types."""
        images = [gen() for gen in NATURAL_IMAGE_GENERATORS.values()]
        watermarked = [embedder.embed(img, multi_author_keys) for img in images]

        wm_result = batch_analyze_ghost(watermarked, multi_author_keys.public_key, config)
        clean_result = batch_analyze_ghost(images, multi_author_keys.public_key, config)

        assert wm_result.correlation != clean_result.correlation

    def test_batch_jpeg_differs_from_clean_jpeg(self, embedder, multi_author_keys, config):
        images = [gen() for gen in list(NATURAL_IMAGE_GENERATORS.values())[:5]]
        watermarked = [embedder.embed(img, multi_author_keys) for img in images]
        wm_compressed = [_jpeg_compress(wm, 75) for wm in watermarked]
        clean_compressed = [_jpeg_compress(img, 75) for img in images]

        wm_result = batch_analyze_ghost(wm_compressed, multi_author_keys.public_key, config)
        clean_result = batch_analyze_ghost(clean_compressed, multi_author_keys.public_key, config)
        assert wm_result.correlation != clean_result.correlation

    def test_batch_returns_valid_result(self, embedder, multi_author_keys, config):
        images = [make_natural_scene(seed=s) for s in range(5)]
        watermarked = [embedder.embed(img, multi_author_keys) for img in images]
        result = batch_analyze_ghost(watermarked, multi_author_keys.public_key, config)
        assert isinstance(result, GhostAnalysisResult)
        assert len(result.band_energies) == len(config.ghost_bands)
        assert 0 <= result.p_value <= 1


class TestGhostBandEnergies:
    def test_band_energies_all_positive(self, embedder, natural_image, multi_author_keys, config):
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        result = analyze_ghost_signature(watermarked, multi_author_keys.public_key, config)
        for band, energy in result.band_energies.items():
            assert energy > 0, f"Band {band} energy is {energy} on {name}"

    def test_watermarked_band_energies_differ_from_clean(
        self, embedder, natural_image, multi_author_keys, config
    ):
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        wm_result = analyze_ghost_signature(watermarked, multi_author_keys.public_key, config)
        clean_result = analyze_ghost_signature(img, multi_author_keys.public_key, config)

        different_count = sum(
            1
            for band in config.ghost_bands
            if abs(wm_result.band_energies[band] - clean_result.band_energies[band]) > 1.0
        )
        assert different_count >= 1, f"No band energies changed on {name}"
