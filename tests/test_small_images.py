"""Edge case tests for small and unusual image dimensions.

Tests watermark embedding/detection on:
- Very small images (64x64, 128x128)
- Non-square images
- Odd-dimension images
- Very large images (1024x1024)
- Rectangular images with extreme aspect ratios
"""

import numpy as np
import pytest

from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.detect import SigilDetector
from sigil_watermark.keygen import generate_author_keys
from sigil_watermark.config import SigilConfig
from sigil_watermark.tiling import best_tile_size


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def author_keys():
    return generate_author_keys(seed=b"small-image-test-author32bytes!!")


@pytest.fixture
def embedder(config):
    return SigilEmbedder(config=config)


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


def _make_image(rng, size):
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            img[i, j] = (
                128
                + 40 * np.sin(i / max(h / 25, 1))
                + 30 * np.cos(j / max(w / 30, 1))
            )
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255)


def _make_rgb_image(rng, size):
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        for i in range(h):
            for j in range(w):
                img[i, j, c] = (
                    128
                    + 40 * np.sin((i + c * 20) / max(h / 25, 1))
                    + 30 * np.cos((j + c * 15) / max(w / 30, 1))
                )
    img += rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255)


# --- Small Images ---


class TestSmallImages:
    """Watermarking on small images."""

    @pytest.mark.parametrize("size", [(128, 128), (256, 256)])
    def test_small_roundtrip(self, embedder, detector, author_keys, size):
        rng = np.random.default_rng(42)
        img = _make_image(rng, size)
        watermarked = embedder.embed(img, author_keys)
        assert watermarked.shape == img.shape
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Size {size}: conf={result.payload_confidence:.2f}"
        )

    def test_64x64_embeds(self, embedder, author_keys):
        """64x64 should at least embed without crashing."""
        rng = np.random.default_rng(42)
        img = _make_image(rng, (64, 64))
        watermarked = embedder.embed(img, author_keys)
        assert watermarked.shape == (64, 64)

    def test_128x128_quality(self, embedder, author_keys):
        """PSNR on small image should still be reasonable."""
        rng = np.random.default_rng(42)
        img = _make_image(rng, (128, 128))
        watermarked = embedder.embed(img, author_keys)
        mse = np.mean((watermarked - img) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255.0**2 / mse)
            assert psnr > 25, f"128x128 PSNR {psnr:.1f} dB too low"


# --- Non-Square Images ---


class TestNonSquareImages:
    """Various non-square dimensions."""

    @pytest.mark.parametrize(
        "size",
        [
            (256, 512),
            (512, 256),
            (384, 512),
            (512, 384),
            (256, 1024),
            (1024, 256),
        ],
    )
    def test_non_square_roundtrip(self, embedder, detector, author_keys, size):
        rng = np.random.default_rng(42)
        img = _make_image(rng, size)
        watermarked = embedder.embed(img, author_keys)
        assert watermarked.shape == size
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Size {size}: conf={result.payload_confidence:.2f}"
        )

    def test_non_square_psnr(self, embedder, author_keys):
        rng = np.random.default_rng(42)
        for size in [(256, 512), (512, 256), (384, 512)]:
            img = _make_image(rng, size)
            watermarked = embedder.embed(img, author_keys)
            mse = np.mean((watermarked - img) ** 2)
            if mse > 0:
                psnr = 10 * np.log10(255.0**2 / mse)
                assert psnr > 30, f"Size {size}: PSNR {psnr:.1f} dB too low"


# --- Odd Dimensions ---


class TestOddDimensions:
    """Odd-numbered dimensions require careful handling."""

    @pytest.mark.parametrize(
        "size",
        [
            (511, 511),
            (513, 513),
            (255, 257),
            (500, 500),
            (333, 444),
        ],
    )
    def test_odd_dimensions(self, embedder, detector, author_keys, size):
        rng = np.random.default_rng(42)
        img = _make_image(rng, size)
        watermarked = embedder.embed(img, author_keys)
        # Output should match input shape
        assert watermarked.shape == size
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, (
            f"Size {size}: conf={result.payload_confidence:.2f}"
        )


# --- Large Images ---


class TestLargeImages:
    """Large image handling."""

    def test_1024x1024(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng, (1024, 1024))
        watermarked = embedder.embed(img, author_keys)
        assert watermarked.shape == (1024, 1024)
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected

    def test_1024x1024_psnr(self, embedder, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng, (1024, 1024))
        watermarked = embedder.embed(img, author_keys)
        mse = np.mean((watermarked - img) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255.0**2 / mse)
            assert psnr > 35, f"1024x1024 PSNR {psnr:.1f} dB too low"


# --- RGB Small/Non-Square ---


class TestRGBSizeVariations:
    """Color images at various sizes."""

    @pytest.mark.parametrize(
        "size",
        [(256, 256), (512, 256), (256, 512), (384, 512)],
    )
    def test_rgb_sizes(self, embedder, detector, author_keys, size):
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng, size)
        watermarked = embedder.embed(img, author_keys)
        assert watermarked.shape == (*size, 3)
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4


# --- Tile Size Selection Edge Cases ---


class TestTileSizeEdgeCases:
    """Tile size selection for various subband shapes."""

    def test_subband_from_small_image(self):
        """A 64x64 image after 3-level DWT has ~8x8 subbands."""
        ts = best_tile_size((8, 8), (32, 64, 128, 256), num_bits=100)
        assert ts == 32  # Fallback

    def test_subband_from_128_image(self):
        """128x128 image after 3-level DWT has ~16x16 subbands."""
        ts = best_tile_size((16, 16), (32, 64, 128, 256), num_bits=100)
        assert ts == 32  # Fallback

    def test_subband_from_256_image(self):
        """256x256 image after 3-level DWT has ~32x32 subbands."""
        ts = best_tile_size((32, 32), (32, 64, 128, 256), num_bits=100)
        assert ts == 32

    def test_subband_from_512_image(self):
        """512x512 image after 3-level DWT has ~64x64 subbands."""
        ts = best_tile_size((64, 64), (32, 64, 128, 256), num_bits=100)
        assert ts in (32, 64)

    def test_subband_from_1024_image(self):
        """1024x1024 image after 3-level DWT has ~128x128 subbands."""
        ts = best_tile_size((128, 128), (32, 64, 128, 256), num_bits=100)
        assert ts in (32, 64, 128)


# --- Consistency Across Sizes ---


class TestCrossSizeConsistency:
    """Same author key should work across different image sizes."""

    def test_same_key_different_sizes(self, embedder, detector, author_keys):
        sizes = [(256, 256), (512, 512), (384, 512)]
        for size in sizes:
            rng = np.random.default_rng(42)
            img = _make_image(rng, size)
            watermarked = embedder.embed(img, author_keys)
            result = detector.detect(watermarked, author_keys.public_key)
            assert result.detected or result.payload_confidence > 0.4, (
                f"Failed on size {size}: conf={result.payload_confidence:.2f}"
            )

    def test_no_false_positive_across_sizes(self, detector, author_keys):
        sizes = [(128, 128), (256, 256), (512, 512), (256, 512)]
        for size in sizes:
            rng = np.random.default_rng(42)
            img = _make_image(rng, size)
            result = detector.detect(img, author_keys.public_key)
            assert not result.detected, f"False positive on clean {size}"
