"""Tile destruction and recovery tests.

Tests how many tiles can be corrupted before the payload is lost,
asymmetric cropping, and edge cases in tile sizing.
"""

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from sigil_watermark.config import SigilConfig
from sigil_watermark.detect import SigilDetector
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.keygen import generate_author_keys
from sigil_watermark.tiling import best_tile_size, majority_vote


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def author_keys():
    return generate_author_keys(seed=b"tile-destruction-test-32-bytes!!")


@pytest.fixture
def embedder(config):
    return SigilEmbedder(config=config)


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


def _make_image(rng, size=(512, 512)):
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            img[i, j] = 128 + 40 * np.sin(i / 20.0) + 30 * np.cos(j / 15.0)
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255)


def _jpeg_compress(image, quality):
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.float64)


# --- Asymmetric Cropping ---


class TestAsymmetricCrop:
    """Crops from different sides and with different amounts per axis."""

    def test_crop_left_only(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        # Crop 20% from the left
        h, w = watermarked.shape
        cropped = watermarked[:, int(w * 0.2) :].copy()
        result = detector.detect(cropped, author_keys.public_key)
        assert result.ring_confidence > 0.3 or result.payload_confidence > 0.3, (
            f"Left crop: ring={result.ring_confidence:.2f}, payload={result.payload_confidence:.2f}"
        )

    def test_crop_top_only(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        cropped = watermarked[int(h * 0.2) :, :].copy()
        result = detector.detect(cropped, author_keys.public_key)
        assert result.ring_confidence > 0.3 or result.payload_confidence > 0.3

    def test_crop_bottom_right_corner(self, embedder, detector, author_keys):
        """Keep only the bottom-right 70% x 70%."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        cropped = watermarked[int(h * 0.3) :, int(w * 0.3) :].copy()
        result = detector.detect(cropped, author_keys.public_key)
        assert result.ring_confidence > 0.2 or result.payload_confidence > 0.2

    def test_narrow_horizontal_strip(self, embedder, detector, author_keys):
        """Keep a narrow horizontal strip (full width, 40% height)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        y0 = int(h * 0.3)
        y1 = int(h * 0.7)
        cropped = watermarked[y0:y1, :].copy()
        result = detector.detect(cropped, author_keys.public_key)
        assert result.ring_confidence > 0.2 or result.payload_confidence > 0.2

    def test_narrow_vertical_strip(self, embedder, detector, author_keys):
        """Keep a narrow vertical strip (40% width, full height)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        x0 = int(w * 0.3)
        x1 = int(w * 0.7)
        cropped = watermarked[:, x0:x1].copy()
        result = detector.detect(cropped, author_keys.public_key)
        assert result.ring_confidence > 0.2 or result.payload_confidence > 0.2

    def test_asymmetric_crop_different_ratios(self, embedder, detector, author_keys):
        """Crop different percentages from each side."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        # 5% from top, 15% from bottom, 10% from left, 20% from right
        cropped = watermarked[
            int(h * 0.05) : int(h * 0.85),
            int(w * 0.10) : int(w * 0.80),
        ].copy()
        result = detector.detect(cropped, author_keys.public_key)
        assert result.ring_confidence > 0.2 or result.payload_confidence > 0.2


# --- Tile Region Corruption ---


class TestTileCorruption:
    """Corrupt specific regions to simulate tile destruction."""

    def test_one_quadrant_destroyed(self, embedder, detector, author_keys):
        """Zero out one quadrant — 3/4 of tiles survive."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        attacked = watermarked.copy()
        h, w = attacked.shape
        attacked[: h // 2, : w // 2] = 128  # Destroy top-left quadrant
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"1/4 destroyed: conf={result.payload_confidence:.2f}"
        )

    def test_two_quadrants_destroyed(self, embedder, detector, author_keys):
        """Zero out two quadrants — 1/2 of tiles survive."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        attacked = watermarked.copy()
        h, w = attacked.shape
        attacked[: h // 2, :] = 128  # Destroy top half
        result = detector.detect(attacked, author_keys.public_key)
        assert result.ring_confidence > 0.2 or result.payload_confidence > 0.3, (
            f"2/4 destroyed: ring={result.ring_confidence:.2f}, "
            f"payload={result.payload_confidence:.2f}"
        )

    def test_random_blocks_corrupted(self, embedder, detector, author_keys):
        """Corrupt random 64x64 blocks (simulating partial occlusion)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        attacked = watermarked.copy()
        h, w = attacked.shape
        block_rng = np.random.default_rng(99)
        # Corrupt ~25% of 64x64 blocks
        for y in range(0, h, 64):
            for x in range(0, w, 64):
                if block_rng.random() < 0.25:
                    bh = min(64, h - y)
                    bw = min(64, w - x)
                    attacked[y : y + bh, x : x + bw] = 128
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3

    def test_checkerboard_corruption(self, embedder, detector, author_keys):
        """Corrupt every other 64x64 block (50% tiles destroyed)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        attacked = watermarked.copy()
        h, w = attacked.shape
        for y_idx, y in enumerate(range(0, h, 64)):
            for x_idx, x in enumerate(range(0, w, 64)):
                if (y_idx + x_idx) % 2 == 0:
                    bh = min(64, h - y)
                    bw = min(64, w - x)
                    attacked[y : y + bh, x : x + bw] = 128
        result = detector.detect(attacked, author_keys.public_key)
        assert result.ring_confidence > 0.2 or result.payload_confidence > 0.2


# --- Majority Vote Unit Tests ---


class TestMajorityVote:
    """Direct tests for the majority voting function."""

    def test_unanimous_vote(self):
        all_bits = [[1, 0, 1, 1, 0]] * 5
        voted, conf = majority_vote(all_bits, 5)
        assert voted == [1, 0, 1, 1, 0]
        assert conf == 1.0

    def test_one_dissenter(self):
        all_bits = [
            [1, 0, 1, 1, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 0, 0, 1],  # Dissenter
        ]
        voted, conf = majority_vote(all_bits, 5)
        assert voted == [1, 0, 1, 1, 0]
        assert conf > 0.6

    def test_tie_goes_to_zero(self):
        all_bits = [
            [1, 1],
            [0, 0],
        ]
        voted, conf = majority_vote(all_bits, 2)
        # With exactly half ones, should vote 0 (not > half)
        assert voted == [0, 0]

    def test_empty_input(self):
        voted, conf = majority_vote([], 4)
        assert voted == [0, 0, 0, 0]
        assert conf == 0.0

    def test_single_voter(self):
        all_bits = [[1, 0, 1, 0]]
        voted, conf = majority_vote(all_bits, 4)
        assert voted == [1, 0, 1, 0]
        assert conf == 1.0

    def test_many_voters_noise(self):
        """9 correct + 3 flipped should still recover."""
        correct = [1, 0, 1, 1, 0, 0, 1, 0]
        np.random.default_rng(42)
        all_bits = []
        for i in range(12):
            if i < 9:
                all_bits.append(correct.copy())
            else:
                flipped = [1 - b for b in correct]
                all_bits.append(flipped)
        voted, conf = majority_vote(all_bits, 8)
        assert voted == correct
        assert conf > 0.7


# --- Best Tile Size ---


class TestBestTileSize:
    """Tests for tile size selection edge cases."""

    def test_large_subband(self):
        ts = best_tile_size((256, 256), (32, 64, 128, 256), num_bits=100)
        assert ts in (32, 64, 128, 256)

    def test_small_subband(self):
        ts = best_tile_size((64, 64), (32, 64, 128, 256), num_bits=100)
        assert ts <= 64

    def test_very_small_subband(self):
        ts = best_tile_size((32, 32), (32, 64, 128, 256), num_bits=100)
        assert ts == 32  # Only option that fits

    def test_tiny_subband_fallback(self):
        ts = best_tile_size((16, 16), (32, 64, 128, 256), num_bits=100)
        assert ts == 32  # Falls back to minimum

    def test_wide_subband(self):
        """Wide but short subband."""
        ts = best_tile_size((64, 512), (32, 64, 128, 256), num_bits=100)
        assert ts in (32, 64)

    def test_tall_subband(self):
        """Tall but narrow subband."""
        ts = best_tile_size((512, 64), (32, 64, 128, 256), num_bits=100)
        assert ts in (32, 64)

    def test_many_bits_reduces_tile_size(self):
        """More payload bits should prefer smaller tiles with more spreading."""
        ts_small_payload = best_tile_size((256, 256), (32, 64, 128), num_bits=50)
        ts_large_payload = best_tile_size((256, 256), (32, 64, 128), num_bits=500)
        assert ts_large_payload <= ts_small_payload


# --- Crop + JPEG Combined on Tiles ---


class TestTileSurvivalCombined:
    """Test tile survival under crop + other attacks."""

    def test_crop_10_then_jpeg(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        nh, nw = int(h * 0.9), int(w * 0.9)
        nh = max(nh - nh % 2, 64)
        nw = max(nw - nw % 2, 64)
        cropped = watermarked[
            (h - nh) // 2 : (h - nh) // 2 + nh,
            (w - nw) // 2 : (w - nw) // 2 + nw,
        ].copy()
        attacked = _jpeg_compress(cropped, 75)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.ring_confidence > 0.3 or result.payload_confidence > 0.3

    def test_crop_20_then_noise(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        nh, nw = int(h * 0.8), int(w * 0.8)
        nh = max(nh - nh % 2, 64)
        nw = max(nw - nw % 2, 64)
        cropped = watermarked[
            (h - nh) // 2 : (h - nh) // 2 + nh,
            (w - nw) // 2 : (w - nw) // 2 + nw,
        ].copy()
        attacked = np.clip(cropped + rng.normal(0, 10, cropped.shape), 0, 255)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.ring_confidence > 0.2 or result.payload_confidence > 0.2

    def test_crop_30_then_blur(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        nh, nw = int(h * 0.7), int(w * 0.7)
        nh = max(nh - nh % 2, 64)
        nw = max(nw - nw % 2, 64)
        cropped = watermarked[
            (h - nh) // 2 : (h - nh) // 2 + nh,
            (w - nw) // 2 : (w - nw) // 2 + nw,
        ].copy()
        attacked = cv2.GaussianBlur(cropped.astype(np.float32), (5, 5), 1.0).astype(np.float64)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.ring_confidence > 0.2 or result.payload_confidence > 0.2
