"""Tests for color (RGB/YCbCr) watermark support."""

import io

import numpy as np
import pytest
from PIL import Image

from sigil_watermark.color import (
    extract_y_channel,
    prepare_for_embedding,
    reconstruct_from_embedding,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)
from sigil_watermark.config import SigilConfig
from sigil_watermark.detect import SigilDetector
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.keygen import generate_author_keys


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def author_keys():
    return generate_author_keys(seed=b"color-test-author-32bytes-long!!")


@pytest.fixture
def embedder(config):
    return SigilEmbedder(config=config)


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


def _make_rgb_image(rng, size=(512, 512)):
    """Create a synthetic RGB test image."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        for i in range(h):
            for j in range(w):
                img[i, j, c] = 128 + 40 * np.sin((i + c * 30) / 25) + 30 * np.cos((j + c * 20) / 20)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255)


def _make_gray_image(rng, size=(512, 512)):
    h, w = size
    img = np.zeros(size, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            img[i, j] = 128 + 30 * np.sin(i / 30) + 20 * np.cos(j / 20)
    img += rng.normal(0, 5, size)
    return np.clip(img, 0, 255)


# --- Unit Tests: Color Conversion ---


class TestColorConversion:
    def test_rgb_ycbcr_roundtrip(self):
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng, (64, 64))
        ycbcr = rgb_to_ycbcr(img)
        recovered = ycbcr_to_rgb(ycbcr)
        # Should be very close (clipping may cause minor differences at edges)
        assert np.allclose(img, recovered, atol=0.5)

    def test_y_channel_range(self):
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng, (64, 64))
        ycbcr = rgb_to_ycbcr(img)
        assert ycbcr[:, :, 0].min() >= -1.0
        assert ycbcr[:, :, 0].max() <= 256.0

    def test_pure_white(self):
        img = np.full((10, 10, 3), 255.0)
        ycbcr = rgb_to_ycbcr(img)
        assert np.allclose(ycbcr[:, :, 0], 255.0, atol=0.5)
        assert np.allclose(ycbcr[:, :, 1], 128.0, atol=0.5)
        assert np.allclose(ycbcr[:, :, 2], 128.0, atol=0.5)

    def test_pure_black(self):
        img = np.full((10, 10, 3), 0.0)
        ycbcr = rgb_to_ycbcr(img)
        assert np.allclose(ycbcr[:, :, 0], 0.0, atol=0.5)
        assert np.allclose(ycbcr[:, :, 1], 128.0, atol=0.5)
        assert np.allclose(ycbcr[:, :, 2], 128.0, atol=0.5)


class TestPrepareReconstruct:
    def test_grayscale_passthrough(self):
        rng = np.random.default_rng(42)
        img = _make_gray_image(rng, (64, 64))
        y, meta = prepare_for_embedding(img)
        assert not meta.is_color
        assert np.array_equal(y, img)
        reconstructed = reconstruct_from_embedding(y, meta)
        assert np.array_equal(reconstructed, img)

    def test_rgb_extracts_y(self):
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng, (64, 64))
        y, meta = prepare_for_embedding(img)
        assert meta.is_color
        assert y.ndim == 2
        assert y.shape == (64, 64)
        assert meta.cb_channel is not None
        assert meta.cr_channel is not None

    def test_rgb_roundtrip(self):
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng, (64, 64))
        y, meta = prepare_for_embedding(img)
        reconstructed = reconstruct_from_embedding(y, meta)
        assert reconstructed.shape == img.shape
        assert np.allclose(img, reconstructed, atol=1.0)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            prepare_for_embedding(np.zeros((10, 10, 4)))


class TestExtractYChannel:
    def test_grayscale(self):
        img = np.full((64, 64), 100.0)
        assert np.array_equal(extract_y_channel(img), img)

    def test_rgb(self):
        img = np.full((64, 64, 3), 100.0)
        y = extract_y_channel(img)
        assert y.ndim == 2
        assert y.shape == (64, 64)


# --- Integration Tests: Full Pipeline with Color ---


class TestColorPipeline:
    def test_rgb_embed_detect_roundtrip(self, embedder, detector, author_keys):
        """Embed in RGB, detect from RGB."""
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng)
        watermarked = embedder.embed(img, author_keys)
        assert watermarked.shape == img.shape  # Should be RGB
        assert watermarked.ndim == 3
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected, f"RGB detection failed: {result}"
        assert result.author_id_match

    def test_rgb_output_range(self, embedder, author_keys):
        """Watermarked RGB image should stay in valid range."""
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng)
        watermarked = embedder.embed(img, author_keys)
        assert watermarked.min() >= -1.0  # May have minor overshoot
        assert watermarked.max() <= 256.0

    def test_rgb_psnr(self, embedder, author_keys):
        """PSNR should be reasonable for color images."""
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng)
        watermarked = embedder.embed(img, author_keys)
        mse = np.mean((watermarked - img) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255.0**2 / mse)
            assert psnr > 35.0, f"Color PSNR {psnr:.1f}dB too low"

    def test_grayscale_still_works(self, embedder, detector, author_keys):
        """Grayscale backward compatibility."""
        rng = np.random.default_rng(42)
        img = _make_gray_image(rng)
        watermarked = embedder.embed(img, author_keys)
        assert watermarked.ndim == 2
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected

    def test_no_false_positive_rgb(self, detector, author_keys):
        """Clean RGB image should not detect."""
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng)
        result = detector.detect(img, author_keys.public_key)
        assert not result.detected

    def test_rgb_jpeg_survives(self, embedder, detector, author_keys):
        """Color JPEG compression should preserve watermark."""
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng)
        watermarked = embedder.embed(img, author_keys)

        # JPEG compress as RGB
        img_uint8 = np.clip(watermarked, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        jpeg_img = np.array(Image.open(buf).convert("RGB"), dtype=np.float64)

        result = detector.detect(jpeg_img, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Color JPEG failed: conf={result.payload_confidence:.2f}"
        )

    def test_rgb_different_sizes(self, embedder, detector, author_keys):
        """Color images of different sizes."""
        rng = np.random.default_rng(42)
        for size in [(256, 256), (512, 512)]:
            img = _make_rgb_image(rng, size=size)
            watermarked = embedder.embed(img, author_keys)
            result = detector.detect(watermarked, author_keys.public_key)
            assert result.detected, f"RGB failed for size {size}"


class TestColorAttacks:
    """Color-specific attack tests."""

    def test_saturation_change(self, embedder, detector, author_keys):
        """Changing saturation shouldn't affect Y-channel watermark."""
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng)
        watermarked = embedder.embed(img, author_keys)

        # Desaturate: move all channels toward the Y channel
        y = (
            0.299 * watermarked[:, :, 0]
            + 0.587 * watermarked[:, :, 1]
            + 0.114 * watermarked[:, :, 2]
        )
        desaturated = np.stack([watermarked[:, :, c] * 0.5 + y * 0.5 for c in range(3)], axis=-1)

        result = detector.detect(desaturated, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4

    def test_brightness_shift(self, embedder, detector, author_keys):
        """Brightness shift on RGB image."""
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng)
        watermarked = embedder.embed(img, author_keys)
        brightened = np.clip(watermarked + 20, 0, 255)
        result = detector.detect(brightened, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4

    def test_grayscale_conversion(self, embedder, detector, author_keys):
        """Converting watermarked RGB to grayscale should preserve watermark."""
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng)
        watermarked = embedder.embed(img, author_keys)
        # Convert to grayscale (same as Y channel extraction)
        gray = extract_y_channel(watermarked)
        result = detector.detect(gray, author_keys.public_key)
        assert result.detected
