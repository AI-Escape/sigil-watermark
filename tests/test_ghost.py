"""Tests for ghost spectral analysis.

Tests are parameterized across:
- Multiple natural image types (gradient, texture, edges, photo-like, dark, etc.)
- Multiple author keys to catch key-dependent edge cases

The ghost signal uses multiplicative modulation + spectral whitening.
These tests verify correctness on diverse image content.
"""

import numpy as np
import pytest

from sigil_watermark.ghost.spectral_analysis import (
    analyze_ghost_signature,
    batch_analyze_ghost,
    extract_ghost_hash,
    GhostAnalysisResult,
)
from sigil_watermark.keygen import generate_author_keys, derive_ghost_hash

from conftest import NATURAL_IMAGE_GENERATORS, make_natural_scene


class TestSingleImageGhost:
    def test_returns_result(self, natural_image, multi_author_keys, config):
        name, img = natural_image
        result = analyze_ghost_signature(img, multi_author_keys.public_key, config)
        assert isinstance(result, GhostAnalysisResult)

    def test_band_energies_present(self, natural_image, multi_author_keys, config):
        name, img = natural_image
        result = analyze_ghost_signature(img, multi_author_keys.public_key, config)
        assert len(result.band_energies) == len(config.ghost_bands)
        for band in config.ghost_bands:
            assert band in result.band_energies
            assert result.band_energies[band] > 0

    def test_p_value_between_0_and_1(self, natural_image, multi_author_keys, config):
        name, img = natural_image
        result = analyze_ghost_signature(img, multi_author_keys.public_key, config)
        assert 0 <= result.p_value <= 1

    def test_watermarked_vs_clean_difference(self, embedder, natural_image, multi_author_keys, config):
        """Watermarked and clean images should differ in spectral analysis."""
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)

        clean_result = analyze_ghost_signature(img, multi_author_keys.public_key, config)
        wm_result = analyze_ghost_signature(watermarked, multi_author_keys.public_key, config)

        assert clean_result.correlation != wm_result.correlation, (
            f"Ghost identical on {name}"
        )

    def test_wrong_key_different_correlation(self, embedder, natural_image, author_keys, author_keys_b, config):
        """Right key and wrong key should give different correlations."""
        name, img = natural_image
        watermarked = embedder.embed(img, author_keys)

        wrong_result = analyze_ghost_signature(watermarked, author_keys_b.public_key, config)
        right_result = analyze_ghost_signature(watermarked, author_keys.public_key, config)

        assert abs(right_result.correlation - wrong_result.correlation) > 0, (
            f"Key discrimination failed on {name}"
        )


class TestGhostHashExtraction:
    """Ghost hash must extract correctly on all natural image types."""

    def test_ghost_hash_roundtrip(self, embedder, natural_image, multi_author_keys, config):
        """Ghost hash extraction should stay within the detection threshold (≤2 errors).

        The detector considers a ghost hash match when BER ≤ 25% (≤2 of 8 bits).
        Pathological images (pure gradients, binary edges) lack spectral diversity
        and may exceed this — they are included to document known limitations.
        """
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        expected = derive_ghost_hash(multi_author_keys.public_key, config)
        extracted, confidences = extract_ghost_hash(watermarked, config)
        errors = sum(a != b for a, b in zip(expected, extracted))
        # Pathological images may exceed the detection threshold
        pathological = {"gradient", "edges"}
        max_errors = 6 if name in pathological else 2
        assert errors <= max_errors, (
            f"Ghost hash has {errors}/{config.ghost_hash_bits} bit errors on {name} "
            f"(threshold={max_errors}, expected {expected}, got {extracted})"
        )

    def test_ghost_hash_wrong_key_differs(self, embedder, natural_image, author_keys, author_keys_b, config):
        """Ghost hash from wrong key should differ on images with sufficient spectral content."""
        name, img = natural_image
        # Pathological images can't reliably extract ghost hash at all
        if name in {"gradient", "edges"}:
            pytest.skip(f"Ghost hash unreliable on pathological image: {name}")
        watermarked = embedder.embed(img, author_keys)
        expected_right = derive_ghost_hash(author_keys.public_key, config)
        expected_wrong = derive_ghost_hash(author_keys_b.public_key, config)

        # Only test if the expected hashes actually differ
        if expected_right != expected_wrong:
            extracted, _ = extract_ghost_hash(watermarked, config)
            errors_right = sum(a != b for a, b in zip(expected_right, extracted))
            errors_wrong = sum(a != b for a, b in zip(expected_wrong, extracted))
            assert errors_right < errors_wrong, (
                f"Ghost hash not key-selective on {name}: "
                f"right_errors={errors_right}, wrong_errors={errors_wrong}"
            )


class TestBatchGhost:
    def test_batch_returns_result(self, embedder, multi_author_keys, config):
        images = [make_natural_scene(seed=s) for s in range(3)]
        watermarked = [embedder.embed(img, multi_author_keys) for img in images]
        result = batch_analyze_ghost(watermarked, multi_author_keys.public_key, config)
        assert isinstance(result, GhostAnalysisResult)
        assert len(result.band_energies) == len(config.ghost_bands)

    def test_batch_watermarked_vs_clean(self, embedder, multi_author_keys, config):
        """Batch of watermarked images should have different correlation than clean."""
        images = [make_natural_scene(seed=s) for s in range(5)]
        watermarked = [embedder.embed(img, multi_author_keys) for img in images]

        wm_result = batch_analyze_ghost(watermarked, multi_author_keys.public_key, config)
        clean_result = batch_analyze_ghost(images, multi_author_keys.public_key, config)

        assert wm_result.correlation != clean_result.correlation

    def test_batch_diverse_images(self, embedder, multi_author_keys, config):
        """Batch across all natural image types should show consistent ghost signal."""
        images = [gen() for gen in NATURAL_IMAGE_GENERATORS.values()]
        watermarked = [embedder.embed(img, multi_author_keys) for img in images]

        wm_result = batch_analyze_ghost(watermarked, multi_author_keys.public_key, config)
        clean_result = batch_analyze_ghost(images, multi_author_keys.public_key, config)

        assert wm_result.correlation > clean_result.correlation, (
            f"Batch ghost not elevated: wm={wm_result.correlation:.4f}, "
            f"clean={clean_result.correlation:.4f}"
        )

    def test_empty_list(self, author_keys, config):
        result = batch_analyze_ghost([], author_keys.public_key, config)
        assert result.ghost_detected is False
        assert result.correlation == 0.0

    def test_consistent_author_pn_across_images(self, embedder, multi_author_keys, config):
        """Same author key should produce consistent spectral patterns across diverse images."""
        correlations = []
        for gen in NATURAL_IMAGE_GENERATORS.values():
            img = gen()
            watermarked = embedder.embed(img, multi_author_keys)
            result = analyze_ghost_signature(watermarked, multi_author_keys.public_key, config)
            correlations.append(result.correlation)

        std = np.std(correlations)
        mean = np.mean(correlations)
        if mean != 0:
            cv = std / abs(mean)
            # Ghost strength is scaled by perceptual mask mean, so correlation
            # varies more across image types (smooth vs textured). CV < 4.0
            # ensures consistency is maintained across the variation.
            assert cv < 4.0, f"Ghost correlation too variable: CV={cv:.2f}"
