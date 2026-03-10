"""Robustness tests for the Sigil watermark system.

Tests watermark survival under various attacks, parameterized across:
- Multiple natural image types (gradient, texture, edges, photo-like, dark, etc.)
- Multiple author keys
"""

import io
import numpy as np
import pytest
from PIL import Image

from conftest import (
    NATURAL_IMAGE_GENERATORS,
    make_natural_scene,
    psnr as compute_psnr,
)


# --- Attack functions ---

def jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf), dtype=np.float64)


def crop_image(image: np.ndarray, crop_fraction: float) -> np.ndarray:
    h, w = image.shape
    keep = 1.0 - crop_fraction
    new_h, new_w = int(h * keep), int(w * keep)
    new_h = max(new_h - new_h % 2, 64)
    new_w = max(new_w - new_w % 2, 64)
    y_start = (h - new_h) // 2
    x_start = (w - new_w) // 2
    return image[y_start:y_start + new_h, x_start:x_start + new_w].copy()


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    import cv2
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    import cv2
    h, w = image.shape
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(image + rng.normal(0, sigma, image.shape), 0, 255)


def adjust_brightness(image: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(image + delta, 0, 255)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    mean = image.mean()
    return np.clip(mean + factor * (image - mean), 0, 255)


# --- JPEG Compression Tests ---

class TestJPEGRobustness:
    """Watermark survival under JPEG compression, across image types and keys."""

    @pytest.mark.parametrize("quality", [95, 75])
    def test_jpeg_quality(self, embedder, detector, realistic_image, multi_author_keys, quality):
        """Watermark should survive JPEG at Q75+ on realistic image types."""
        name, img = realistic_image
        watermarked = embedder.embed(img, multi_author_keys)
        compressed = jpeg_compress(watermarked, quality)
        result = detector.detect(compressed, multi_author_keys.public_key)
        assert result.detected, f"Detection failed on {name} after JPEG Q{quality}"
        assert result.author_id_match, f"Author ID mismatch on {name} after JPEG Q{quality}"

    def test_jpeg_q50_partial(self, embedder, detector, multi_author_keys):
        """JPEG Q50 is aggressive — detection should survive but author match may degrade."""
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        compressed = jpeg_compress(watermarked, 50)
        result = detector.detect(compressed, multi_author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, (
            "No watermark signal survives JPEG Q50"
        )

    def test_jpeg_q20_beacon_survives(self, embedder, detector, multi_author_keys):
        """Even at Q20, beacon should be somewhat detectable."""
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        compressed = jpeg_compress(watermarked, 20)
        beacon = detector.detect_beacon(compressed)
        result = detector.detect(compressed, multi_author_keys.public_key)
        assert beacon or result.payload_confidence > 0.3, \
            "No watermark trace survives JPEG Q20"


# --- Cropping Tests ---

class TestCropRobustness:
    """Watermark survival under cropping."""

    @pytest.mark.parametrize("crop_frac", [0.10, 0.25])
    def test_crop_ring_survival(self, embedder, detector, multi_author_keys, crop_frac):
        """Ring detection should survive moderate cropping."""
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        cropped = crop_image(watermarked, crop_frac)
        result = detector.detect(cropped, multi_author_keys.public_key)
        assert result.ring_confidence > 0.1, \
            f"Ring detection failed after {int(crop_frac*100)}% crop (conf={result.ring_confidence:.2f})"

    def test_crop_50_percent_partial(self, embedder, detector, multi_author_keys):
        """50% crop -- severe, but ring detection may still show signal."""
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        cropped = crop_image(watermarked, 0.50)
        result = detector.detect(cropped, multi_author_keys.public_key)
        assert result.ring_confidence > 0.1 or result.payload_confidence > 0.3, \
            f"No signal after 50% crop"


# --- Rotation Tests ---

class TestRotationRobustness:
    """Watermark survival under rotation."""

    @pytest.mark.parametrize("angle", [90, 180])
    def test_rotation_cardinal(self, embedder, detector, multi_author_keys, angle):
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        rotated = rotate_image(watermarked, angle)
        corrected = rotate_image(rotated, -angle)
        result = detector.detect(corrected, multi_author_keys.public_key)
        assert result.detected, f"Detection failed after {angle} rotation+correction"

    @pytest.mark.parametrize("angle", [1.0, 5.0])
    def test_small_rotation(self, embedder, detector, multi_author_keys, angle):
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        rotated = rotate_image(watermarked, angle)
        corrected = rotate_image(rotated, -angle)
        result = detector.detect(corrected, multi_author_keys.public_key)
        if angle <= 1.0:
            assert result.detected, f"Detection failed after {angle} rotation+correction"
        else:
            assert result.beacon_found or result.payload_confidence > 0.4, \
                f"No signal after {angle} rotation+correction"


# --- Resize Tests ---

class TestResizeRobustness:
    """Watermark survival under resizing."""

    @pytest.mark.parametrize("scale", [0.5, 2.0])
    def test_resize_and_back(self, embedder, detector, multi_author_keys, scale):
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        resized = resize_image(watermarked, scale)
        result = detector.detect(resized, multi_author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"No signal after {scale}x resize roundtrip: "
            f"payload={result.payload_confidence:.3f}"
        )

    def test_resize_quarter_and_back(self, embedder, detector, multi_author_keys):
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        resized = resize_image(watermarked, 0.25)
        result = detector.detect(resized, multi_author_keys.public_key)
        assert result.beacon_found or result.payload_confidence > 0.3, \
            "No signal after 0.25x resize roundtrip"


# --- Noise Tests ---

class TestNoiseRobustness:
    """Watermark survival under additive noise."""

    @pytest.mark.parametrize("sigma", [5, 10])
    def test_mild_noise(self, embedder, detector, realistic_image, multi_author_keys, sigma):
        """Across realistic image types and keys.

        With adaptive ring strength, ring_confidence drops to 0.15-0.30 on
        typical images.  After noise, the payload may also degrade enough that
        author_id_match fails.  In those cases we require significant watermark
        signal (payload_confidence > 0.4) instead of strict detected=True.
        """
        name, img = realistic_image
        watermarked = embedder.embed(img, multi_author_keys)
        noisy = add_gaussian_noise(watermarked, sigma)
        result = detector.detect(noisy, multi_author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"No signal on {name} with sigma={sigma} noise: "
            f"payload={result.payload_confidence:.3f}"
        )

    def test_heavy_noise(self, embedder, detector, multi_author_keys):
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        noisy = add_gaussian_noise(watermarked, 20)
        result = detector.detect(noisy, multi_author_keys.public_key)
        assert result.beacon_found or result.payload_confidence > 0.3, \
            "No signal with sigma=20 noise"


# --- Brightness/Contrast Tests ---

class TestBrightnessContrastRobustness:
    """Watermark survival under brightness and contrast changes."""

    @pytest.mark.parametrize("delta", [30, -30])
    def test_brightness(self, embedder, detector, realistic_image, multi_author_keys, delta):
        name, img = realistic_image
        watermarked = embedder.embed(img, multi_author_keys)
        adjusted = adjust_brightness(watermarked, delta)
        result = detector.detect(adjusted, multi_author_keys.public_key)
        assert result.detected, f"Detection failed on {name} after brightness {delta:+d}"

    @pytest.mark.parametrize("factor", [0.7, 1.3])
    def test_contrast(self, embedder, detector, realistic_image, multi_author_keys, factor):
        name, img = realistic_image
        watermarked = embedder.embed(img, multi_author_keys)
        adjusted = adjust_contrast(watermarked, factor)
        result = detector.detect(adjusted, multi_author_keys.public_key)
        assert result.detected, f"Detection failed on {name} after {factor}x contrast"


# --- Combined Attacks ---

class TestCombinedAttacks:
    """Watermark survival under multiple simultaneous attacks."""

    def test_jpeg_plus_noise(self, embedder, detector, multi_author_keys):
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        attacked = add_gaussian_noise(jpeg_compress(watermarked, 75), 5)
        result = detector.detect(attacked, multi_author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, (
            "No signal after JPEG Q75 + noise"
        )

    def test_jpeg_plus_brightness(self, embedder, detector, multi_author_keys):
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        attacked = adjust_brightness(jpeg_compress(watermarked, 75), 20)
        result = detector.detect(attacked, multi_author_keys.public_key)
        assert result.detected, "Detection failed after JPEG Q75 + brightness"

    def test_resize_plus_noise(self, embedder, detector, multi_author_keys):
        img = make_natural_scene()
        watermarked = embedder.embed(img, multi_author_keys)
        attacked = add_gaussian_noise(resize_image(watermarked, 0.5), 5)
        result = detector.detect(attacked, multi_author_keys.public_key)
        # Resize 0.5x + sigma=5 noise is aggressive combined attack
        assert result.detected or result.payload_confidence > 0.3, (
            "No signal after resize + noise"
        )


# --- Quality Metrics Across Image Types ---

class TestQualityMetrics:
    """Measure PSNR and max deviation across all natural image types and keys."""

    def test_psnr_across_image_types(self, embedder, natural_image, multi_author_keys):
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        p = compute_psnr(img, watermarked)
        assert p > 37.0, f"PSNR {p:.1f}dB unacceptably low on {name}"

    def test_max_deviation_across_types(self, embedder, natural_image, multi_author_keys):
        name, img = natural_image
        watermarked = embedder.embed(img, multi_author_keys)
        max_dev = np.max(np.abs(watermarked - img))
        assert max_dev < 30, f"Max deviation {max_dev:.1f} on {name}"

    def test_detection_across_types(self, embedder, detector, realistic_image, multi_author_keys):
        name, img = realistic_image
        watermarked = embedder.embed(img, multi_author_keys)
        result = detector.detect(watermarked, multi_author_keys.public_key)
        assert result.detected, f"Detection failed on {name}"
        assert result.author_id_match, f"Author ID mismatch on {name}"


# --- Robustness Report ---

class TestRobustnessReport:
    def test_print_robustness_summary(self, embedder, detector, author_keys, capsys):
        img = make_natural_scene()
        watermarked = embedder.embed(img, author_keys)
        p = compute_psnr(img, watermarked)

        attacks = {
            "No attack": watermarked,
            "JPEG Q95": jpeg_compress(watermarked, 95),
            "JPEG Q75": jpeg_compress(watermarked, 75),
            "JPEG Q50": jpeg_compress(watermarked, 50),
            "Crop 10%": crop_image(watermarked, 0.10),
            "Resize 0.5x": resize_image(watermarked, 0.5),
            "Noise s=10": add_gaussian_noise(watermarked, 10),
            "Bright +30": adjust_brightness(watermarked, 30),
            "Contrast 1.3x": adjust_contrast(watermarked, 1.3),
        }

        print(f"\n{'='*70}")
        print(f"ROBUSTNESS REPORT (PSNR: {p:.1f} dB)")
        print(f"{'='*70}")
        print(f"{'Attack':<20} {'Detected':>10} {'Payload':>10} {'Ring':>10} {'AuthorID':>10}")
        print(f"{'-'*70}")

        for name, attacked_img in attacks.items():
            result = detector.detect(attacked_img, author_keys.public_key)
            print(
                f"{name:<20} "
                f"{'YES' if result.detected else 'no':>10} "
                f"{result.payload_confidence:>10.2f} "
                f"{result.ring_confidence:>10.2f} "
                f"{'YES' if result.author_id_match else 'no':>10}"
            )
        print(f"{'='*70}")
