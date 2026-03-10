"""Tests for geometric auto-correction (Fourier-Mellin Transform)."""

import cv2
import numpy as np
import pytest

from sigil_watermark.config import SigilConfig
from sigil_watermark.detect import SigilDetector
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.geometric import (
    auto_correct,
    estimate_rotation_scale,
    try_rotations,
)
from sigil_watermark.keygen import generate_author_keys


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def author_keys():
    return generate_author_keys(seed=b"geometric-test-author-32bytes!!")


@pytest.fixture
def embedder(config):
    return SigilEmbedder(config=config)


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


def _make_test_image(rng, size=(512, 512)):
    """Create a synthetic test image with varied frequency content."""
    h, w = size
    img = np.zeros(size, dtype=np.float64)
    for i in range(h):
        for j in range(w):
            img[i, j] = 128 + 30 * np.sin(i / 30) + 20 * np.cos(j / 20) + 10 * np.sin((i + j) / 15)
    img += rng.normal(0, 5, size)
    return np.clip(img, 0, 255)


def _rotate_image(image, angle_degrees):
    """Rotate image by a given angle around the center."""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(
        image.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return rotated.astype(np.float64)


def _scale_image(image, scale_factor):
    """Scale image by a given factor then crop/pad to original size."""
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    scaled = cv2.resize(image.astype(np.float32), (new_w, new_h)).astype(np.float64)

    # Center crop or pad to original size
    result = np.full((h, w), 128.0)
    y_off = (new_h - h) // 2
    x_off = (new_w - w) // 2
    if scale_factor >= 1.0:
        result = scaled[y_off : y_off + h, x_off : x_off + w]
    else:
        dst_y = -y_off
        dst_x = -x_off
        result[dst_y : dst_y + new_h, dst_x : dst_x + new_w] = scaled
    return result


# --- Unit Tests: auto_correct ---


class TestAutoCorrect:
    def test_identity(self):
        """No correction (0 angle, 1.0 scale) should return ~same image."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng, (128, 128))
        corrected = auto_correct(img, angle=0.0, scale=1.0)
        assert np.allclose(img, corrected, atol=0.5)

    def test_90_degree_inverse(self):
        """Rotating 90 degrees then correcting by -90 should recover."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng, (128, 128))
        rotated = _rotate_image(img, 90)
        corrected = auto_correct(rotated, angle=90.0)
        # Should be close to original (border effects will cause differences)
        center = corrected[30:98, 30:98]
        orig_center = img[30:98, 30:98]
        mse = np.mean((center - orig_center) ** 2)
        assert mse < 50, f"MSE after 90-degree correction too high: {mse:.1f}"

    def test_small_angle_correction(self):
        """Small rotation correction."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng, (128, 128))
        rotated = _rotate_image(img, 2)
        corrected = auto_correct(rotated, angle=2.0)
        center = corrected[20:108, 20:108]
        orig_center = img[20:108, 20:108]
        mse = np.mean((center - orig_center) ** 2)
        assert mse < 30, f"MSE after 2-degree correction: {mse:.1f}"


# --- Unit Tests: try_rotations ---


class TestTryRotations:
    def test_finds_best_angle(self):
        """Should find the angle that maximizes the confidence function."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng, (128, 128))

        # Create a confidence function that peaks at a specific rotation
        target_angle = 5.0
        original = _rotate_image(img, -target_angle)  # "un-rotated" version

        def conf_fn(test_img):
            center = test_img[20:108, 20:108]
            orig_center = original[20:108, 20:108]
            mse = np.mean((center - orig_center) ** 2)
            return max(0, 1.0 - mse / 500)

        rotated = _rotate_image(original, target_angle)
        best_img, best_angle, best_conf = try_rotations(rotated, conf_fn)
        assert best_angle == target_angle or best_conf > conf_fn(rotated)

    def test_identity_when_already_best(self):
        """If the original image has highest confidence, return angle=0."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng, (128, 128))

        def conf_fn(test_img):
            mse = np.mean((test_img - img) ** 2)
            return max(0, 1.0 - mse / 500)

        best_img, best_angle, best_conf = try_rotations(img, conf_fn)
        assert best_angle == 0.0

    def test_custom_angles(self):
        """Can provide custom angle list."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng, (128, 128))

        def conf_fn(test_img):
            return 0.5

        _, _, _ = try_rotations(img, conf_fn, angles=[0, 45, -45])
        # Should not raise


# --- Unit Tests: estimate_rotation_scale ---


class TestEstimateRotationScale:
    def test_identity_without_reference(self):
        """Without a reference, should return (0, 1)."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng, (128, 128))
        angle, scale = estimate_rotation_scale(img, reference=None)
        assert angle == 0.0
        assert scale == 1.0

    def test_with_reference(self):
        """With a reference image, should return some estimate."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng, (128, 128))
        ref = _make_test_image(rng, (128, 128))
        angle, scale = estimate_rotation_scale(img, reference=ref)
        # Just check it returns valid numbers
        assert isinstance(angle, float)
        assert isinstance(scale, float)
        assert scale > 0


# --- Integration Tests: Detection with Rotation ---


class TestDetectionWithRotation:
    def test_detect_90_rotation(self, embedder, detector, author_keys):
        """90-degree rotation should be detected via auto-correction."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng)
        watermarked = embedder.embed(img, author_keys)
        rotated = _rotate_image(watermarked, 90)

        result = detector.detect(rotated, author_keys.public_key)
        assert result.detected, f"90-degree rotation failed: {result}"

    def test_detect_180_rotation(self, embedder, detector, author_keys):
        """180-degree rotation."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng)
        watermarked = embedder.embed(img, author_keys)
        rotated = _rotate_image(watermarked, 180)

        result = detector.detect(rotated, author_keys.public_key)
        assert result.detected, f"180-degree rotation failed: {result}"

    def test_detect_270_rotation(self, embedder, detector, author_keys):
        """270-degree rotation."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng)
        watermarked = embedder.embed(img, author_keys)
        rotated = _rotate_image(watermarked, 270)

        result = detector.detect(rotated, author_keys.public_key)
        assert result.detected, f"270-degree rotation failed: {result}"

    def test_detect_small_rotation_1deg(self, embedder, detector, author_keys):
        """1-degree rotation — small angle auto-correction."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng)
        watermarked = embedder.embed(img, author_keys)
        rotated = _rotate_image(watermarked, 1)

        result = detector.detect(rotated, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"1-degree rotation: conf={result.payload_confidence:.2f}"
        )

    def test_detect_small_rotation_2deg(self, embedder, detector, author_keys):
        """2-degree rotation."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng)
        watermarked = embedder.embed(img, author_keys)
        rotated = _rotate_image(watermarked, 2)

        result = detector.detect(rotated, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"2-degree rotation: conf={result.payload_confidence:.2f}"
        )

    def test_detect_5deg_rotation(self, embedder, detector, author_keys):
        """5-degree rotation — moderate angle."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng)
        watermarked = embedder.embed(img, author_keys)
        rotated = _rotate_image(watermarked, 5)

        result = detector.detect(rotated, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, (
            f"5-degree rotation: conf={result.payload_confidence:.2f}"
        )

    def test_unmodified_still_detected(self, embedder, detector, author_keys):
        """Unrotated watermarked image still detected (no regression)."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng)
        watermarked = embedder.embed(img, author_keys)
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected
        assert result.author_id_match

    def test_no_false_positive_rotated(self, detector, author_keys):
        """Rotated clean image should not detect."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng)
        rotated = _rotate_image(img, 90)
        result = detector.detect(rotated, author_keys.public_key)
        assert not result.detected


class TestDetectionWithScale:
    def test_detect_after_downscale_and_back(self, embedder, detector, author_keys):
        """Downscale to 75% then upscale back — should still detect."""
        rng = np.random.default_rng(42)
        img = _make_test_image(rng)
        watermarked = embedder.embed(img, author_keys)

        h, w = watermarked.shape
        small = cv2.resize(watermarked.astype(np.float32), (int(w * 0.75), int(h * 0.75)))
        restored = cv2.resize(small, (w, h)).astype(np.float64)

        result = detector.detect(restored, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Downscale/upscale: conf={result.payload_confidence:.2f}"
        )

    def test_detect_after_rotation_plus_jpeg(self, embedder, detector, author_keys):
        """Rotation + JPEG — combined attack."""
        import io

        from PIL import Image

        rng = np.random.default_rng(42)
        img = _make_test_image(rng)
        watermarked = embedder.embed(img, author_keys)
        rotated = _rotate_image(watermarked, 90)

        # JPEG compress
        img_uint8 = np.clip(rotated, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        jpeg_img = np.array(Image.open(buf).convert("L"), dtype=np.float64)

        result = detector.detect(jpeg_img, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, (
            f"Rotation+JPEG: conf={result.payload_confidence:.2f}"
        )
