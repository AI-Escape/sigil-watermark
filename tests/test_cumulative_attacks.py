"""Cumulative and chained attack tests.

Tests watermark survival under multiple weak attacks applied sequentially,
simulating real-world scenarios where images go through several processing steps.
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


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def author_keys():
    return generate_author_keys(seed=b"cumulative-attack-test-32bytes!!")


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
            img[i, j] = (
                128 + 40 * np.sin(i / 20.0) + 30 * np.cos(j / 15.0) + 20 * np.sin((i + j) / 25.0)
            )
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255)


def _jpeg(image, quality):
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.float64)


def _blur(image, sigma):
    ksize = max(3, int(sigma * 6) | 1)
    return cv2.GaussianBlur(image.astype(np.float32), (ksize, ksize), sigma).astype(np.float64)


def _noise(image, sigma, rng):
    return np.clip(image + rng.normal(0, sigma, image.shape), 0, 255)


def _brightness(image, delta):
    return np.clip(image + delta, 0, 255)


def _contrast(image, factor):
    mean = image.mean()
    return np.clip(mean + factor * (image - mean), 0, 255)


def _gamma(image, g):
    return np.power(np.clip(image, 0, 255) / 255.0, g) * 255.0


# --- Three-Step Chains ---


class TestThreeStepChains:
    """Chains of three distinct mild attacks."""

    def test_blur_noise_jpeg(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = _blur(wm, 0.5)
        attacked = _noise(attacked, 5, rng)
        attacked = _jpeg(attacked, 85)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected, f"blur+noise+jpeg: conf={result.payload_confidence:.2f}"

    def test_brightness_gamma_jpeg(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = _brightness(wm, 15)
        attacked = _gamma(attacked, 1.1)
        attacked = _jpeg(attacked, 80)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected, f"bright+gamma+jpeg: conf={result.payload_confidence:.2f}"

    def test_contrast_blur_noise(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = _contrast(wm, 1.2)
        attacked = _blur(attacked, 0.8)
        attacked = _noise(attacked, 8, rng)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4

    def test_noise_jpeg_gamma(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = _noise(wm, 5, rng)
        attacked = _jpeg(attacked, 85)
        attacked = _gamma(attacked, 1.2)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4

    def test_gamma_contrast_jpeg(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = _gamma(wm, 0.8)
        attacked = _contrast(attacked, 1.15)
        attacked = _jpeg(attacked, 75)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4


# --- Four-Step Chains ---


class TestFourStepChains:
    """Chains of four attacks."""

    def test_blur_noise_brightness_jpeg(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = _blur(wm, 0.5)
        attacked = _noise(attacked, 5, rng)
        attacked = _brightness(attacked, 10)
        attacked = _jpeg(attacked, 80)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4

    def test_gamma_blur_contrast_jpeg(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = _gamma(wm, 1.1)
        attacked = _blur(attacked, 0.5)
        attacked = _contrast(attacked, 1.1)
        attacked = _jpeg(attacked, 80)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4

    def test_noise_gamma_blur_jpeg(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = _noise(wm, 5, rng)
        attacked = _gamma(attacked, 1.15)
        attacked = _blur(attacked, 0.5)
        attacked = _jpeg(attacked, 75)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3


# --- Five-Step Chains ---


class TestFiveStepChains:
    """Chains of five attacks — stress test."""

    def test_full_processing_pipeline(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = _brightness(wm, 10)
        attacked = _contrast(attacked, 1.1)
        attacked = _blur(attacked, 0.5)
        attacked = _noise(attacked, 3, rng)
        attacked = _jpeg(attacked, 85)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4

    def test_aggressive_five_step(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = _gamma(wm, 1.2)
        attacked = _noise(attacked, 8, rng)
        attacked = _blur(attacked, 0.8)
        attacked = _brightness(attacked, -15)
        attacked = _jpeg(attacked, 70)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3


# --- Repeated Same Attack ---


class TestRepeatedAttacks:
    """Apply the same attack multiple times."""

    @pytest.mark.parametrize("n_times", [2, 3, 5])
    def test_repeated_blur(self, embedder, detector, author_keys, n_times):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = wm
        for _ in range(n_times):
            attacked = _blur(attacked, 0.5)
        result = detector.detect(attacked, author_keys.public_key)
        if n_times <= 3:
            assert result.detected or result.payload_confidence > 0.4
        else:
            assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3

    @pytest.mark.parametrize("n_times", [2, 3, 5])
    def test_repeated_noise(self, embedder, detector, author_keys, n_times):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = wm
        for i in range(n_times):
            noise_rng = np.random.default_rng(42 + i)
            attacked = _noise(attacked, 3, noise_rng)
        result = detector.detect(attacked, author_keys.public_key)
        if n_times <= 3:
            assert result.detected or result.payload_confidence > 0.4
        else:
            assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3

    @pytest.mark.parametrize("n_times", [2, 3])
    def test_repeated_gamma(self, embedder, detector, author_keys, n_times):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = wm
        for _ in range(n_times):
            attacked = _gamma(attacked, 1.1)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3


# --- Attack Order Matters ---


class TestAttackOrdering:
    """Same attacks in different orders may have different effects."""

    def test_jpeg_then_noise_vs_noise_then_jpeg(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        # Order A: JPEG first then noise
        a = _jpeg(wm, 75)
        a = _noise(a, 8, np.random.default_rng(99))

        # Order B: noise first then JPEG
        b = _noise(wm, 8, np.random.default_rng(99))
        b = _jpeg(b, 75)

        result_a = detector.detect(a, author_keys.public_key)
        result_b = detector.detect(b, author_keys.public_key)

        # Both should survive, but may have different confidence
        assert result_a.payload_confidence > 0.3 or result_a.ring_confidence > 0.3, (
            f"Order A failed: conf={result_a.payload_confidence:.2f}"
        )
        assert result_b.payload_confidence > 0.3 or result_b.ring_confidence > 0.3, (
            f"Order B failed: conf={result_b.payload_confidence:.2f}"
        )

    def test_blur_then_gamma_vs_gamma_then_blur(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        a = _blur(wm, 1.0)
        a = _gamma(a, 1.3)

        b = _gamma(wm, 1.3)
        b = _blur(b, 1.0)

        result_a = detector.detect(a, author_keys.public_key)
        result_b = detector.detect(b, author_keys.public_key)

        assert result_a.payload_confidence > 0.2 or result_a.ring_confidence > 0.3
        assert result_b.payload_confidence > 0.2 or result_b.ring_confidence > 0.3


# --- Histogram Attacks in Chain ---


class TestHistogramChainAttacks:
    """Histogram-based attacks combined with other processing."""

    def test_histeq_then_jpeg(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        img_u8 = np.clip(wm, 0, 255).astype(np.uint8)
        attacked = cv2.equalizeHist(img_u8).astype(np.float64)
        attacked = _jpeg(attacked, 80)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3

    def test_clahe_then_blur_then_jpeg(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        img_u8 = np.clip(wm, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        attacked = clahe.apply(img_u8).astype(np.float64)
        attacked = _blur(attacked, 0.5)
        attacked = _jpeg(attacked, 80)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3


# --- Median Filter in Chain ---


class TestMedianFilterChains:
    """Median filter (non-linear) combined with other attacks."""

    def test_median_then_jpeg(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = cv2.medianBlur(wm.astype(np.float32), 3).astype(np.float64)
        attacked = _jpeg(attacked, 80)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3

    def test_jpeg_then_median(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = _jpeg(wm, 80)
        attacked = cv2.medianBlur(attacked.astype(np.float32), 3).astype(np.float64)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3
