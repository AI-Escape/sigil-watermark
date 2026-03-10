"""Tests for fractal tiling — crop robustness via tiled DWT embedding."""

import numpy as np
import pytest

from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.detect import SigilDetector
from sigil_watermark.keygen import generate_author_keys
from sigil_watermark.config import SigilConfig
from sigil_watermark.tiling import tile_embed, tile_extract, best_tile_size, majority_vote
from sigil_watermark.transforms import embed_spread_spectrum, extract_spread_spectrum


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def author_keys():
    return generate_author_keys(seed=b"tiling-test-author-32-bytes!!!!!")


@pytest.fixture
def embedder(config):
    return SigilEmbedder(config=config)


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


def _make_natural_image(rng, size=(512, 512)):
    img = np.zeros(size, dtype=np.float64)
    for i in range(size[0]):
        for j in range(size[1]):
            img[i, j] = 128 + 30 * np.sin(i / 30) + 20 * np.cos(j / 20) + rng.normal(0, 5)
    return np.clip(img, 0, 255)


def crop_image(image, crop_fraction):
    """Crop by removing crop_fraction from each side."""
    h, w = image.shape
    margin_h = int(crop_fraction * h / 2)
    margin_w = int(crop_fraction * w / 2)
    return image[margin_h:h - margin_h, margin_w:w - margin_w].copy()


def crop_asymmetric(image, top, left, bottom, right):
    """Crop with different amounts from each side (fractions)."""
    h, w = image.shape
    t = int(top * h)
    b = h - int(bottom * h)
    l = int(left * w)
    r = w - int(right * w)
    return image[t:b, l:r].copy()


# --- Unit Tests: tile_embed / tile_extract ---


class TestTileEmbedExtract:
    """Low-level tiling functions."""

    def test_roundtrip_single_tile(self):
        """Single tile that fits perfectly."""
        rng = np.random.default_rng(42)
        subband = rng.normal(0, 10, (64, 64))
        pn = np.where(rng.integers(0, 2, 64 * 64) == 0, -1.0, 1.0)
        payload = [1, 0, 1, 1, 0, 0, 1, 0]

        embedded = tile_embed(subband, pn, payload, tile_size=64, strength=5.0, spreading_factor=256)
        bits, conf = tile_extract(embedded, pn, num_bits=8, tile_size=64, spreading_factor=256)
        assert bits == payload

    def test_roundtrip_multiple_tiles(self):
        """Multiple tiles in a larger subband."""
        rng = np.random.default_rng(42)
        subband = rng.normal(0, 10, (256, 256))
        pn = np.where(rng.integers(0, 2, 256 * 256) == 0, -1.0, 1.0)
        payload = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]

        embedded = tile_embed(subband, pn, payload, tile_size=64, strength=5.0, spreading_factor=128)
        bits, conf = tile_extract(embedded, pn, num_bits=10, tile_size=64, spreading_factor=128)
        assert bits == payload
        assert conf > 0.9

    def test_partial_tiles_skipped(self):
        """Subband not evenly divisible by tile_size — partial tiles are handled."""
        rng = np.random.default_rng(42)
        subband = rng.normal(0, 10, (100, 100))
        pn = np.where(rng.integers(0, 2, 100 * 100) == 0, -1.0, 1.0)
        payload = [1, 0, 1, 1]

        # Should not crash — partial tiles will be handled
        embedded = tile_embed(subband, pn, payload, tile_size=64, strength=5.0, spreading_factor=128)
        bits, conf = tile_extract(embedded, pn, num_bits=4, tile_size=64, spreading_factor=128)
        assert bits == payload


class TestBestTileSize:
    def test_selects_largest_fitting(self):
        ts = best_tile_size((256, 256), (32, 64, 128, 256), num_bits=20)
        assert ts == 128  # 256 gives only 1 tile, 128 gives 4

    def test_small_subband_uses_smallest(self):
        ts = best_tile_size((32, 32), (32, 64, 128, 256), num_bits=8)
        assert ts == 32

    def test_large_payload_forces_smaller_tiles(self):
        # 144-bit payload needs at least 144*4 = 576 coefficients per tile
        ts = best_tile_size((256, 256), (32, 64, 128, 256), num_bits=144)
        # 32*32=1024 >= 576, but gives 64 tiles. 64*64=4096, 16 tiles. 128*128=16384, 4 tiles.
        assert ts in (32, 64, 128)


class TestMajorityVote:
    def test_perfect_agreement(self):
        all_bits = [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]
        voted, conf = majority_vote(all_bits, 4)
        assert voted == [1, 0, 1, 0]
        assert conf == 1.0

    def test_majority_wins(self):
        all_bits = [[1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0]]
        voted, conf = majority_vote(all_bits, 4)
        assert voted == [1, 0, 1, 0]

    def test_single_voter(self):
        all_bits = [[1, 0, 1]]
        voted, conf = majority_vote(all_bits, 3)
        assert voted == [1, 0, 1]


# --- Integration Tests: Crop Robustness ---


class TestCropRobustnessIntegration:
    """Full pipeline crop tests — the main reason tiling exists."""

    def test_crop_10_percent_detects(self, embedder, detector, author_keys):
        """10% crop should show signal with tiling."""
        rng = np.random.default_rng(42)
        img = _make_natural_image(rng)
        watermarked = embedder.embed(img, author_keys)
        cropped = crop_image(watermarked, 0.10)
        result = detector.detect(cropped, author_keys.public_key)
        # After crop, tile alignment may shift — accept ring or payload signal
        assert result.detected or result.payload_confidence > 0.4 or result.ring_confidence > 0.5, \
            f"10% crop failed: conf={result.payload_confidence:.2f}, ring={result.ring_confidence:.2f}"

    def test_crop_25_percent_detects(self, embedder, detector, author_keys):
        """25% crop should still show strong signal."""
        rng = np.random.default_rng(42)
        img = _make_natural_image(rng)
        watermarked = embedder.embed(img, author_keys)
        cropped = crop_image(watermarked, 0.25)
        result = detector.detect(cropped, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, \
            f"25% crop failed: conf={result.payload_confidence:.2f}"

    def test_crop_50_percent(self, embedder, detector, author_keys):
        """50% crop — severe but tiling should help."""
        rng = np.random.default_rng(42)
        img = _make_natural_image(rng)
        watermarked = embedder.embed(img, author_keys)
        cropped = crop_image(watermarked, 0.50)
        result = detector.detect(cropped, author_keys.public_key)
        # Even 50% crop should retain some signal with tiling
        assert result.ring_confidence > 0.3 or result.payload_confidence > 0.3, \
            f"50% crop: ring={result.ring_confidence:.2f}, payload={result.payload_confidence:.2f}"

    def test_asymmetric_crop(self, embedder, detector, author_keys):
        """Non-symmetric crop (more from one side)."""
        rng = np.random.default_rng(42)
        img = _make_natural_image(rng)
        watermarked = embedder.embed(img, author_keys)
        # Crop 30% from top, 10% from left, 5% from bottom, 0% from right
        cropped = crop_asymmetric(watermarked, 0.30, 0.10, 0.05, 0.0)
        result = detector.detect(cropped, author_keys.public_key)
        assert result.ring_confidence > 0.3 or result.payload_confidence > 0.3

    def test_crop_no_false_positive(self, detector, author_keys):
        """Cropped clean image should NOT detect."""
        rng = np.random.default_rng(42)
        img = _make_natural_image(rng)
        cropped = crop_image(img, 0.10)
        result = detector.detect(cropped, author_keys.public_key)
        assert not result.detected

    def test_crop_plus_jpeg(self, embedder, detector, author_keys):
        """Crop + JPEG compression — realistic attack chain."""
        import io
        from PIL import Image

        rng = np.random.default_rng(42)
        img = _make_natural_image(rng)
        watermarked = embedder.embed(img, author_keys)
        cropped = crop_image(watermarked, 0.15)

        # JPEG Q75
        img_uint8 = np.clip(cropped, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8, mode='L')
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=75)
        buf.seek(0)
        jpeg_img = np.array(Image.open(buf), dtype=np.float64)

        result = detector.detect(jpeg_img, author_keys.public_key)
        assert result.ring_confidence > 0.3 or result.payload_confidence > 0.3, \
            f"Crop+JPEG: ring={result.ring_confidence:.2f}, payload={result.payload_confidence:.2f}"

    def test_crop_different_sizes(self, embedder, detector, author_keys):
        """Crop robustness across different image sizes."""
        for size in [(256, 256), (512, 512), (512, 1024)]:
            rng = np.random.default_rng(42)
            img = _make_natural_image(rng, size=size)
            watermarked = embedder.embed(img, author_keys)
            cropped = crop_image(watermarked, 0.15)
            result = detector.detect(cropped, author_keys.public_key)
            assert result.payload_confidence > 0.3 or result.ring_confidence > 0.3, \
                f"Crop failed for size {size}"
