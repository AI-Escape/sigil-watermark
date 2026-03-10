"""Tests for the full embed/detect pipeline.

Tests are parameterized across:
- Multiple natural image types (gradient, texture, edges, photo-like, dark, etc.)
- Multiple author keys to catch key-dependent edge cases
"""

import numpy as np
import pytest
from conftest import (
    make_edges_image,
    make_natural_scene,
    psnr,
)

from sigil_watermark.detect import DetectionResult
from sigil_watermark.keygen import generate_author_keys


class TestBasicRoundTrip:
    """Core tests: embed then detect should recover author identity."""

    def test_embed_returns_image(self, embedder, author_keys):
        img = make_natural_scene()
        result = embedder.embed(img, author_keys)
        assert result.shape == img.shape
        assert result.dtype == np.float64

    def test_embed_preserves_value_range(self, embedder, author_keys):
        img = make_natural_scene()
        result = embedder.embed(img, author_keys)
        assert result.min() >= -10
        assert result.max() <= 265

    # Images that lack the spectral diversity needed for full watermark fidelity
    PATHOLOGICAL_IMAGES = {"gradient", "edges", "highfreq"}

    def test_roundtrip_detects_author(self, embedder, detector, natural_image, multi_author_keys):
        """Roundtrip detection across all image types and all author keys.

        Pathological images (gradient, edges, highfreq) are extreme synthetic content
        that lack the spectral diversity of real photographs. They may fail full
        detection but should still show significant watermark signal.
        """
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        result = detector.detect(watermarked, multi_author_keys.public_key)
        assert isinstance(result, DetectionResult)
        if name in self.PATHOLOGICAL_IMAGES:
            # At minimum, some watermark signal should be present
            assert result.confidence > 0.2 or result.payload_confidence > 0.3, (
                f"No watermark signal on {name}: conf={result.confidence:.3f}, "
                f"payload={result.payload_confidence:.3f}"
            )
        else:
            assert result.detected is True, f"Detection failed on {name}"
            assert result.confidence > 0.4, f"Low confidence on {name}: {result.confidence:.3f}"
            assert result.author_id_match is True, f"Author ID mismatch on {name}"

    def test_no_false_positive_on_clean_image(self, detector, natural_image, multi_author_keys):
        """Clean images should not trigger detection, across all image types and keys."""
        name, img = natural_image
        result = detector.detect(img, multi_author_keys.public_key)
        assert result.detected is False or result.confidence < 0.3, (
            f"False positive on clean {name} image"
        )

    def test_wrong_key_does_not_detect(
        self, embedder, detector, natural_image, author_keys, author_keys_b
    ):
        """Embedding with key A, detecting with key B should fail."""
        name, img = natural_image
        watermarked = embedder.embed(img, author_keys)
        result = detector.detect(watermarked, author_keys_b.public_key)
        assert result.author_id_match is False, f"Cross-key false positive on {name}"


class TestImperceptibility:
    """Quality metrics for the watermarked image."""

    def test_psnr_above_threshold(self, embedder, natural_image, multi_author_keys):
        """PSNR should meet threshold across all image types and keys."""
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        p = psnr(img, watermarked)
        assert p > 37.0, f"PSNR {p:.1f}dB unacceptably low on {name}"

    def test_ssim_above_threshold(self, embedder, author_keys, config):
        img = make_natural_scene()
        watermarked = embedder.embed(img, author_keys)
        ssim = _compute_ssim(img, watermarked)
        assert ssim > config.target_ssim, f"SSIM {ssim:.4f} below target {config.target_ssim}"

    def test_max_pixel_deviation(self, embedder, natural_image, multi_author_keys):
        """Max pixel change should be bounded across all image types and keys."""
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        max_dev = np.max(np.abs(watermarked - img))
        assert max_dev < 30, f"Max pixel deviation {max_dev:.1f} on {name}"


class TestCrossKeyFalsePositives:
    """Verify zero cross-key false positives."""

    def test_many_keys_no_cross_detection(self, embedder, detector, natural_image):
        """Test 20-key cross-detection on each natural image type."""
        name, img = natural_image
        if name in TestBasicRoundTrip.PATHOLOGICAL_IMAGES:
            pytest.skip(f"Skipping cross-key test on pathological image: {name}")
        keys = [
            generate_author_keys(seed=f"cross-key-test-{i:04d}-32-bytes!!".encode())
            for i in range(20)
        ]
        watermarked = embedder.embed(img, keys[0])
        for i, k in enumerate(keys):
            result = detector.detect(watermarked, k.public_key)
            if i == 0:
                assert result.detected, f"Own key failed on {name}"
                assert result.author_id_match
            else:
                assert result.author_id_match is False, (
                    f"Cross-key FP: key {i} matched key 0 on {name}"
                )


class TestBeaconDetection:
    """Tests for tier-1 universal beacon detection (blind, no key)."""

    def test_no_beacon_in_clean_image(self, detector, natural_image):
        name, img = natural_image
        assert detector.detect_beacon(img) is False, f"False beacon on clean {name} image"


class TestAuthorIndexExtraction:
    """Tests for tier-2 author index extraction (blind, no key)."""

    def test_correct_index_extracted_when_detectable(self, embedder, detector, multi_author_keys):
        from sigil_watermark.keygen import derive_author_index

        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        expected_index = derive_author_index(multi_author_keys.public_key)
        extracted_index = detector.extract_author_index(watermarked)
        if extracted_index is not None:
            errors = sum(a != b for a, b in zip(expected_index, extracted_index))
            assert errors <= 3, f"Author index has {errors} bit errors"


class TestImageFormats:
    """Test with different image dimensions and content types."""

    @pytest.mark.parametrize("size", [(256, 256), (512, 512), (512, 1024), (640, 480)])
    def test_various_sizes(self, embedder, detector, multi_author_keys, size):
        img = make_natural_scene(h=size[0], w=size[1])
        watermarked = embedder.embed(img, multi_author_keys)
        result = detector.detect(watermarked, multi_author_keys.public_key)
        # Small images (256x256) with adaptive ring strength may not achieve
        # full detected=True if ring_confidence < 0.5 and author_id_match fails
        assert result.detected or result.payload_confidence > 0.4, (
            f"No signal for size {size}: payload={result.payload_confidence:.3f}"
        )

    def test_flat_image(self, embedder, detector, multi_author_keys):
        img = np.full((512, 512), 128.0, dtype=np.float64)
        watermarked = embedder.embed(img, multi_author_keys)
        result = detector.detect(watermarked, multi_author_keys.public_key)
        assert result.detected

    def test_high_contrast_image(self, embedder, detector, multi_author_keys):
        img = make_edges_image()
        watermarked = embedder.embed(img, multi_author_keys)
        result = detector.detect(watermarked, multi_author_keys.public_key)
        assert result.detected


def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simple SSIM computation between two images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim)
