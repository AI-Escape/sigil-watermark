"""Re-encoding robustness tests across JPEG and PNG quality levels.

Tests watermark survival under:
- JPEG compression at quality levels 30-99
- PNG lossless roundtrip (uint8 quantization only)
- Multiple sequential re-encodings
- JPEG → PNG and PNG → JPEG chains

All tests parameterized across realistic image types and multiple author keys.
"""

import io

import numpy as np
import pytest
from conftest import (
    jpeg_roundtrip_rgb,
    make_natural_scene,
    make_photo_like_rgb,
    png_roundtrip_rgb,
)
from conftest import (
    psnr as compute_psnr,
)
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def jpeg_roundtrip_gray_quality(image: np.ndarray, quality: int) -> np.ndarray:
    """JPEG roundtrip for grayscale at a given quality level."""
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.float64)


def png_roundtrip_gray(image: np.ndarray) -> np.ndarray:
    """Lossless PNG roundtrip for grayscale: float64 -> uint8 -> PNG -> float64."""
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.float64)


# ---------------------------------------------------------------------------
# JPEG quality sweep — grayscale
# ---------------------------------------------------------------------------


class TestJPEGQualitySweepGrayscale:
    """Systematic JPEG quality sweep on grayscale images."""

    @pytest.mark.parametrize("quality", [99, 95, 90, 85, 80, 75])
    def test_jpeg_high_quality_full_detection(
        self, embedder, detector, realistic_image, multi_author_keys, quality
    ):
        """Q75+ should give full detection on realistic images."""
        name, img = realistic_image
        wm = embedder.embed(img, multi_author_keys)
        compressed = jpeg_roundtrip_gray_quality(wm, quality)
        result = detector.detect(compressed, multi_author_keys.public_key)

        assert result.detected, f"Detection failed on {name} after JPEG Q{quality}"
        assert result.author_id_match, f"Author ID mismatch on {name} after JPEG Q{quality}"

    @pytest.mark.parametrize("quality", [70, 60, 50])
    def test_jpeg_medium_quality_detection(self, embedder, detector, multi_author_keys, quality):
        """Q50-Q70: detection should survive, author match may degrade."""
        img = make_natural_scene()
        wm = embedder.embed(img, multi_author_keys)
        compressed = jpeg_roundtrip_gray_quality(wm, quality)
        result = detector.detect(compressed, multi_author_keys.public_key)

        assert result.detected or result.payload_confidence > 0.3, (
            f"No signal after JPEG Q{quality}"
        )

    @pytest.mark.parametrize("quality", [40, 30, 20])
    def test_jpeg_low_quality_signal_survival(self, embedder, detector, multi_author_keys, quality):
        """Q20-Q40: at least some signal should remain."""
        img = make_natural_scene()
        wm = embedder.embed(img, multi_author_keys)
        compressed = jpeg_roundtrip_gray_quality(wm, quality)
        result = detector.detect(compressed, multi_author_keys.public_key)

        assert (
            result.beacon_found or result.ring_confidence > 0.2 or result.payload_confidence > 0.3
        ), f"No trace after JPEG Q{quality}"


# ---------------------------------------------------------------------------
# JPEG quality sweep — RGB
# ---------------------------------------------------------------------------


class TestJPEGQualitySweepRGB:
    """JPEG quality sweep on RGB images (production pipeline)."""

    @pytest.mark.parametrize("quality", [99, 95, 90, 85, 80, 75])
    def test_jpeg_rgb_high_quality(self, embedder, detector, multi_author_keys, quality):
        """Q75+ on RGB should give full detection."""
        img = make_photo_like_rgb()
        wm = embedder.embed(img, multi_author_keys)
        compressed = jpeg_roundtrip_rgb(wm, quality)
        result = detector.detect(compressed, multi_author_keys.public_key)

        assert result.detected, f"RGB detection failed after JPEG Q{quality}"
        assert result.author_id_match, f"RGB author mismatch after JPEG Q{quality}"

    @pytest.mark.parametrize("quality", [70, 60, 50])
    def test_jpeg_rgb_medium_quality(self, embedder, detector, multi_author_keys, quality):
        """Q50-Q70 on RGB: detection should survive."""
        img = make_photo_like_rgb()
        wm = embedder.embed(img, multi_author_keys)
        compressed = jpeg_roundtrip_rgb(wm, quality)
        result = detector.detect(compressed, multi_author_keys.public_key)

        assert result.detected or result.payload_confidence > 0.3, (
            f"No signal in RGB after JPEG Q{quality}"
        )


# ---------------------------------------------------------------------------
# PNG roundtrip
# ---------------------------------------------------------------------------


class TestPNGRoundtrip:
    """PNG lossless roundtrip — only uint8 quantization damage."""

    def test_png_grayscale_full_detection(
        self, embedder, detector, realistic_image, multi_author_keys
    ):
        """PNG roundtrip on grayscale realistic images."""
        name, img = realistic_image
        wm = embedder.embed(img, multi_author_keys)
        decoded = png_roundtrip_gray(wm)
        result = detector.detect(decoded, multi_author_keys.public_key)

        assert result.detected, f"Detection failed on {name} after PNG"
        assert result.author_id_match, f"Author ID mismatch on {name} after PNG"
        assert result.payload_confidence > 0.7, (
            f"Payload too low on {name} after PNG: {result.payload_confidence:.3f}"
        )

    def test_png_rgb_full_detection(self, embedder, detector, multi_author_keys):
        """PNG roundtrip on RGB image."""
        img = make_photo_like_rgb()
        wm = embedder.embed(img, multi_author_keys)
        decoded = png_roundtrip_rgb(wm)
        result = detector.detect(decoded, multi_author_keys.public_key)

        assert result.detected, "Detection failed after RGB PNG"
        assert result.author_id_match, "Author ID mismatch after RGB PNG"
        assert result.ghost_confidence > 0.2, (
            f"Ghost too low after PNG: {result.ghost_confidence:.3f}"
        )


# ---------------------------------------------------------------------------
# Sequential re-encoding
# ---------------------------------------------------------------------------


class TestSequentialReEncoding:
    """Multiple re-encoding passes (realistic for images shared across platforms)."""

    @pytest.mark.parametrize("rounds", [2, 3, 5])
    def test_multiple_jpeg_q90(self, embedder, detector, multi_author_keys, rounds):
        """Re-encode as JPEG Q90 multiple times (e.g., download + re-upload)."""
        img = make_natural_scene()
        wm = embedder.embed(img, multi_author_keys)

        current = wm
        for _ in range(rounds):
            current = jpeg_roundtrip_gray_quality(current, 90)

        result = detector.detect(current, multi_author_keys.public_key)
        assert result.detected, f"Detection failed after {rounds}x JPEG Q90"

    @pytest.mark.parametrize("rounds", [2, 3, 5])
    def test_multiple_jpeg_q75(self, embedder, detector, multi_author_keys, rounds):
        """Re-encode as JPEG Q75 multiple times."""
        img = make_natural_scene()
        wm = embedder.embed(img, multi_author_keys)

        current = wm
        for _ in range(rounds):
            current = jpeg_roundtrip_gray_quality(current, 75)

        result = detector.detect(current, multi_author_keys.public_key)
        if rounds <= 3:
            assert result.detected, f"Detection failed after {rounds}x JPEG Q75"
        else:
            assert result.beacon_found or result.payload_confidence > 0.3, (
                f"No signal after {rounds}x JPEG Q75"
            )

    def test_jpeg_then_png_then_jpeg(self, embedder, detector, multi_author_keys):
        """JPEG Q90 -> PNG -> JPEG Q90 chain."""
        img = make_natural_scene()
        wm = embedder.embed(img, multi_author_keys)

        step1 = jpeg_roundtrip_gray_quality(wm, 90)
        step2 = png_roundtrip_gray(step1)
        step3 = jpeg_roundtrip_gray_quality(step2, 90)

        result = detector.detect(step3, multi_author_keys.public_key)
        assert result.detected, "Detection failed after JPEG->PNG->JPEG"

    def test_descending_quality_jpeg(self, embedder, detector, multi_author_keys):
        """Realistic scenario: Q99 initial -> Q85 platform -> Q75 reshare."""
        img = make_natural_scene()
        wm = embedder.embed(img, multi_author_keys)

        step1 = jpeg_roundtrip_gray_quality(wm, 99)
        step2 = jpeg_roundtrip_gray_quality(step1, 85)
        step3 = jpeg_roundtrip_gray_quality(step2, 75)

        result = detector.detect(step3, multi_author_keys.public_key)
        assert result.detected, "Detection failed after Q99->Q85->Q75 chain"


# ---------------------------------------------------------------------------
# Quality measurement across JPEG levels
# ---------------------------------------------------------------------------


class TestJPEGQualityImpact:
    """Measure how JPEG compression affects watermark confidence at each level."""

    def test_jpeg_quality_report(self, embedder, detector, author_keys, capsys):
        """Print comprehensive JPEG quality vs. detection confidence report."""
        img = make_natural_scene()
        wm = embedder.embed(img, author_keys)
        base_psnr = compute_psnr(img, wm)

        qualities = [99, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20]

        print(f"\n{'=' * 85}")
        print(f"JPEG QUALITY SWEEP (embedding PSNR: {base_psnr:.1f} dB)")
        print(f"{'=' * 85}")
        print(
            f"{'Quality':<10} {'PSNR(wm→jpg)':<15} {'Detected':<10} "
            f"{'Ring':>8} {'Payload':>10} {'Ghost':>8} {'Author':>8}"
        )
        print(f"{'-' * 85}")

        for q in qualities:
            compressed = jpeg_roundtrip_gray_quality(wm, q)
            jpg_psnr = compute_psnr(wm, compressed)
            result = detector.detect(compressed, author_keys.public_key)
            print(
                f"Q{q:<8} {jpg_psnr:>12.1f} dB  "
                f"{'YES' if result.detected else 'no':<10} "
                f"{result.ring_confidence:>8.3f} "
                f"{result.payload_confidence:>10.3f} "
                f"{result.ghost_confidence:>8.3f} "
                f"{'YES' if result.author_id_match else 'no':>8}"
            )

        print(f"{'=' * 85}")

    def test_rgb_jpeg_quality_report(self, embedder, detector, author_keys, capsys):
        """Print RGB JPEG quality report."""
        img = make_photo_like_rgb()
        wm = embedder.embed(img, author_keys)
        base_psnr = compute_psnr(img, wm)

        qualities = [99, 95, 90, 85, 80, 75, 70, 60, 50]

        print(f"\n{'=' * 85}")
        print(f"RGB JPEG QUALITY SWEEP (embedding PSNR: {base_psnr:.1f} dB)")
        print(f"{'=' * 85}")
        print(
            f"{'Quality':<10} {'Detected':<10} "
            f"{'Ring':>8} {'Payload':>10} {'Ghost':>8} {'Author':>8}"
        )
        print(f"{'-' * 85}")

        for q in qualities:
            compressed = jpeg_roundtrip_rgb(wm, q)
            result = detector.detect(compressed, author_keys.public_key)
            print(
                f"Q{q:<8} {'YES' if result.detected else 'no':<10} "
                f"{result.ring_confidence:>8.3f} "
                f"{result.payload_confidence:>10.3f} "
                f"{result.ghost_confidence:>8.3f} "
                f"{'YES' if result.author_id_match else 'no':>8}"
            )

        print(f"{'=' * 85}")
