"""Expanded image suite: 20+ diverse procedural images covering
art styles and photographic scenarios the platform will encounter.

No external downloads needed — all images are procedurally generated
to have the spectral and statistical properties of real-world content.
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
    return generate_author_keys(seed=b"diverse-img-test-author-32bytes!")


@pytest.fixture
def embedder(config):
    return SigilEmbedder(config=config)


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


# --- Diverse Image Generators ---


def _smooth_gradient(size=(512, 512)):
    """Smooth gradient — like sky, fog, or soft airbrushed art."""
    h, w = size
    y = np.linspace(0, 255, h).reshape(-1, 1)
    x = np.linspace(0, 255, w).reshape(1, -1)
    return (y * 0.6 + x * 0.4).astype(np.float64)


def _natural_landscape(rng, size=(512, 512)):
    """Natural photo-like scene: gradients + multi-scale texture."""
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            img[i, j] = (
                128
                + 40 * np.sin(i / 20.0)
                + 30 * np.cos(j / 15.0)
                + 20 * np.sin((i + j) / 25.0)
                + 10 * np.cos((i - j) / 10.0)
            )
    img += rng.normal(0, 8, (h, w))
    return np.clip(img, 0, 255)


def _portrait_soft(rng, size=(512, 512)):
    """Portrait: smooth center (face), textured edges (hair/background)."""
    h, w = size
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (min(h, w) / 2)
    # Smooth center, noisy edge
    base = 180 - 80 * r**2
    texture = rng.normal(0, 5, (h, w)) * np.clip(r, 0.3, 1.0)
    return np.clip(base + texture, 0, 255).astype(np.float64)


def _high_texture_fabric(rng, size=(512, 512)):
    """Extreme texture — like fabric, grass, or animal fur."""
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            img[i, j] = (
                128
                + 50 * np.sin(i / 3.0) * np.cos(j / 4.0)
                + 30 * np.sin((i * 7 + j * 11) / 50.0)
                + 20 * np.cos(i / 2.0 + j / 3.0)
            )
    img += rng.normal(0, 10, (h, w))
    return np.clip(img, 0, 255)


def _line_art(rng, size=(512, 512)):
    """Ink drawing / line art: sparse, high contrast, mostly white."""
    h, w = size
    img = np.full((h, w), 240.0)
    # Add random lines
    for _ in range(50):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        cv2.line(img.astype(np.float32), (x0, y0), (x1, y1), 20.0, 2)
        img = img.astype(np.float64)
    # Add some curves
    for _ in range(20):
        cx, cy = rng.integers(50, w - 50), rng.integers(50, h - 50)
        r = rng.integers(20, 80)
        cv2.circle(img.astype(np.float32), (cx, cy), r, 30.0, 1)
        img = img.astype(np.float64)
    img += rng.normal(0, 2, (h, w))
    return np.clip(img, 0, 255)


def _watercolor_wash(rng, size=(512, 512)):
    """Watercolor: soft edges, blended washes, subtle texture."""
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    # Multiple overlapping soft blobs
    for _ in range(15):
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        radius = rng.integers(60, 200)
        intensity = rng.uniform(50, 200)
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        blob = intensity * np.exp(-(dist**2) / (2 * radius**2))
        img += blob
    img = img / img.max() * 200 + 30
    # Add soft noise for paper texture
    img += rng.normal(0, 3, (h, w))
    return np.clip(img, 0, 255)


def _digital_illustration(rng, size=(512, 512)):
    """Digital art: flat colors, sharp edges, geometric shapes."""
    h, w = size
    img = np.full((h, w), 200.0)
    # Random rectangles with flat fills
    for _ in range(30):
        x0 = rng.integers(0, w - 50)
        y0 = rng.integers(0, h - 50)
        x1 = x0 + rng.integers(30, 150)
        y1 = y0 + rng.integers(30, 150)
        val = rng.uniform(30, 220)
        img[y0 : min(y1, h), x0 : min(x1, w)] = val
    # Sharp circles
    for _ in range(10):
        cx = rng.integers(50, w - 50)
        cy = rng.integers(50, h - 50)
        r = rng.integers(20, 60)
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 < r**2
        img[mask] = rng.uniform(40, 200)
    return np.clip(img, 0, 255)


def _impressionist(rng, size=(512, 512)):
    """Impressionist painting: heavy brushstroke texture, warm tones."""
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    # Base landscape
    for i in range(h):
        for j in range(w):
            img[i, j] = 140 + 40 * np.sin(i / 40) + 30 * np.cos(j / 30)
    # Simulate brushstrokes with random rectangular patches
    for _ in range(500):
        cx = rng.integers(0, w)
        cy = rng.integers(0, h)
        bw = rng.integers(5, 20)
        bh = rng.integers(3, 10)
        val_shift = rng.normal(0, 20)
        y0, x0 = max(0, cy - bh), max(0, cx - bw)
        y1, x1 = min(h, cy + bh), min(w, cx + bw)
        img[y0:y1, x0:x1] += val_shift
    img += rng.normal(0, 5, (h, w))
    return np.clip(img, 0, 255)


def _macro_photo(rng, size=(512, 512)):
    """Macro photography: shallow DOF, sharp center, smooth bokeh."""
    h, w = size
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (min(h, w) / 4)
    # Sharp center detail
    detail = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            detail[i, j] = 15 * np.sin(i / 5) * np.cos(j / 4)
    # Depth-dependent blur simulation
    sharpness = np.exp(-(r**2) / 2)
    img = 128 + detail * sharpness + rng.normal(0, 3, (h, w)) * (1 - sharpness * 0.5)
    return np.clip(img, 0, 255)


def _dark_moody(rng, size=(512, 512)):
    """Dark/moody image: low key, shadows dominate."""
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            img[i, j] = 40 + 15 * np.sin(i / 30) + 10 * np.cos(j / 25)
    # A few bright highlights
    for _ in range(5):
        cx, cy = rng.integers(50, w - 50), rng.integers(50, h - 50)
        r = rng.integers(30, 80)
        y, x = np.ogrid[:h, :w]
        spot = 100 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r**2))
        img += spot
    img += rng.normal(0, 5, (h, w))
    return np.clip(img, 0, 255)


def _overexposed(rng, size=(512, 512)):
    """Overexposed / high-key image."""
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            img[i, j] = 210 + 20 * np.sin(i / 20) + 15 * np.cos(j / 25)
    img += rng.normal(0, 5, (h, w))
    return np.clip(img, 0, 255)


def _checkerboard(size=(512, 512)):
    """High-contrast checkerboard pattern."""
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            img[i, j] = 240 if (i // 64 + j // 64) % 2 == 0 else 15
    return img


def _noise_only(rng, size=(512, 512)):
    """Pure noise image (worst case for embedding)."""
    return np.clip(rng.normal(128, 40, size), 0, 255).astype(np.float64)


def _stripe_pattern(size=(512, 512)):
    """Horizontal stripes (periodic, tests DWT subband behavior)."""
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        img[i, :] = 128 + 80 * np.sin(2 * np.pi * i / 32)
    return np.clip(img, 0, 255)


def _concentric_circles(size=(512, 512)):
    """Concentric circles (tests DFT ring interference)."""
    h, w = size
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return np.clip(128 + 60 * np.sin(r / 10), 0, 255).astype(np.float64)


def _rgb_landscape(rng, size=(512, 512)):
    """RGB landscape: sky blue top, green bottom, gradual transition."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(h):
        sky_frac = max(0, 1 - i / (h * 0.6))
        ground_frac = 1 - sky_frac
        img[i, :, 0] = 80 * sky_frac + 80 * ground_frac  # R
        img[i, :, 1] = 130 * sky_frac + 140 * ground_frac  # G
        img[i, :, 2] = 200 * sky_frac + 60 * ground_frac  # B
    # Add texture
    for c in range(3):
        img[:, :, c] += 10 * np.sin(np.linspace(0, 20 * np.pi, w)).reshape(1, -1)
    img += rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255)


def _rgb_portrait(rng, size=(512, 512)):
    """RGB portrait: warm skin tones center, darker background."""
    h, w = size
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (min(h, w) / 2)
    img = np.zeros((h, w, 3), dtype=np.float64)
    # Background dark blue
    img[:, :, 0] = 40
    img[:, :, 1] = 50
    img[:, :, 2] = 70
    # Face region: warm skin tone
    face_mask = np.exp(-(r**2) / 0.5)
    img[:, :, 0] += 160 * face_mask
    img[:, :, 1] += 120 * face_mask
    img[:, :, 2] += 90 * face_mask
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255)


def _rgb_vibrant(rng, size=(512, 512)):
    """RGB vibrant digital art: saturated colors, geometric shapes."""
    h, w = size
    img = np.full((h, w, 3), 40.0)
    for _ in range(40):
        x0 = rng.integers(0, w - 40)
        y0 = rng.integers(0, h - 40)
        x1 = x0 + rng.integers(20, 120)
        y1 = y0 + rng.integers(20, 120)
        color = rng.uniform(50, 255, 3)
        img[y0 : min(y1, h), x0 : min(x1, w)] = color
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255)


# --- All Images Collection ---


def _build_all_images(rng):
    """Build dict of all diverse images."""
    return {
        "smooth_gradient": _smooth_gradient(),
        "natural_landscape": _natural_landscape(rng),
        "portrait_soft": _portrait_soft(rng),
        "high_texture": _high_texture_fabric(rng),
        "line_art": _line_art(rng),
        "watercolor": _watercolor_wash(rng),
        "digital_illustration": _digital_illustration(rng),
        "impressionist": _impressionist(rng),
        "macro_photo": _macro_photo(rng),
        "dark_moody": _dark_moody(rng),
        "overexposed": _overexposed(rng),
        "checkerboard": _checkerboard(),
        "noise_only": _noise_only(rng),
        "stripe_pattern": _stripe_pattern(),
        "concentric_circles": _concentric_circles(),
        "rgb_landscape": _rgb_landscape(rng),
        "rgb_portrait": _rgb_portrait(rng),
        "rgb_vibrant": _rgb_vibrant(rng),
    }


ALL_IMAGE_NAMES = list(_build_all_images(np.random.default_rng(42)).keys())
GRAY_IMAGE_NAMES = [n for n in ALL_IMAGE_NAMES if not n.startswith("rgb_")]
RGB_IMAGE_NAMES = [n for n in ALL_IMAGE_NAMES if n.startswith("rgb_")]


@pytest.fixture(params=ALL_IMAGE_NAMES)
def diverse_image(request):
    rng = np.random.default_rng(42)
    images = _build_all_images(rng)
    return request.param, images[request.param]


@pytest.fixture(params=GRAY_IMAGE_NAMES)
def gray_diverse_image(request):
    rng = np.random.default_rng(42)
    images = _build_all_images(rng)
    return request.param, images[request.param]


@pytest.fixture(params=RGB_IMAGE_NAMES)
def rgb_diverse_image(request):
    rng = np.random.default_rng(42)
    images = _build_all_images(rng)
    return request.param, images[request.param]


# --- Helper ---


def _jpeg_compress(image, quality):
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3:
        pil = Image.fromarray(img_u8, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"), dtype=np.float64)
    else:
        pil = Image.fromarray(img_u8, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return np.array(Image.open(buf).convert("L"), dtype=np.float64)


# --- Tests ---


class TestDiverseRoundTrip:
    """Basic embed/detect across all image types."""

    def test_detected(self, embedder, detector, author_keys, diverse_image):
        name, img = diverse_image
        watermarked = embedder.embed(img, author_keys)
        result = detector.detect(watermarked, author_keys.public_key)
        # Noise-only and extreme textures may have reduced confidence
        if name in ("noise_only", "high_texture"):
            assert result.detected or result.payload_confidence > 0.4, (
                f"{name}: conf={result.payload_confidence:.2f}"
            )
        else:
            assert result.detected, (
                f"{name}: detection failed, conf={result.payload_confidence:.2f}"
            )

    def test_no_false_positive(self, detector, author_keys, diverse_image):
        name, img = diverse_image
        result = detector.detect(img, author_keys.public_key)
        assert not result.detected, f"{name}: false positive!"

    def test_psnr(self, embedder, author_keys, diverse_image):
        name, img = diverse_image
        watermarked = embedder.embed(img, author_keys)
        mse = np.mean((watermarked - img) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255.0**2 / mse)
            # Extreme textures and noise get lower PSNR from perceptual mask
            min_psnr = 25.0 if name in ("noise_only", "high_texture") else 30.0
            assert psnr > min_psnr, f"{name}: PSNR {psnr:.1f} dB too low"


class TestDiverseJPEG:
    """JPEG survival across diverse images."""

    @pytest.mark.parametrize("quality", [75, 50])
    def test_jpeg(self, embedder, detector, author_keys, gray_diverse_image, quality):
        name, img = gray_diverse_image
        watermarked = embedder.embed(img, author_keys)
        compressed = _jpeg_compress(watermarked, quality)
        result = detector.detect(compressed, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.35, (
            f"{name} JPEG Q{quality}: conf={result.payload_confidence:.2f}"
        )

    def test_rgb_jpeg(self, embedder, detector, author_keys, rgb_diverse_image):
        name, img = rgb_diverse_image
        watermarked = embedder.embed(img, author_keys)
        compressed = _jpeg_compress(watermarked, 75)
        result = detector.detect(compressed, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.35, (
            f"{name} RGB JPEG Q75: conf={result.payload_confidence:.2f}"
        )


class TestDiverseNoise:
    """Noise survival across diverse images."""

    def test_noise_sigma10(self, embedder, detector, author_keys, gray_diverse_image):
        name, img = gray_diverse_image
        rng = np.random.default_rng(99)
        watermarked = embedder.embed(img, author_keys)
        noisy = np.clip(watermarked + rng.normal(0, 10, watermarked.shape), 0, 255)
        result = detector.detect(noisy, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.35, (
            f"{name} noise sigma=10: conf={result.payload_confidence:.2f}"
        )


class TestDiverseRotation:
    """Rotation survival across diverse images."""

    def test_rotation_90(self, embedder, detector, author_keys, gray_diverse_image):
        name, img = gray_diverse_image
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1.0)
        rotated = cv2.warpAffine(
            watermarked.astype(np.float32),
            M,
            (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float64)
        result = detector.detect(rotated, author_keys.public_key)
        if name in ("noise_only",):
            assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3
        else:
            assert result.detected or result.payload_confidence > 0.4, (
                f"{name} rot90: conf={result.payload_confidence:.2f}"
            )


class TestDiverseCrop:
    """Crop survival across diverse images."""

    def test_crop_10_percent(self, embedder, detector, author_keys, gray_diverse_image):
        name, img = gray_diverse_image
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        nh, nw = int(h * 0.9), int(w * 0.9)
        nh = max(nh - nh % 2, 64)
        nw = max(nw - nw % 2, 64)
        cropped = watermarked[
            (h - nh) // 2 : (h - nh) // 2 + nh,
            (w - nw) // 2 : (w - nw) // 2 + nw,
        ].copy()
        result = detector.detect(cropped, author_keys.public_key)
        assert result.ring_confidence > 0.3 or result.payload_confidence > 0.3, (
            f"{name} crop10%: ring={result.ring_confidence:.2f}, "
            f"payload={result.payload_confidence:.2f}"
        )


class TestDiverseCombined:
    """Combined attacks on diverse images."""

    def test_jpeg_plus_noise(self, embedder, detector, author_keys, gray_diverse_image):
        name, img = gray_diverse_image
        rng = np.random.default_rng(99)
        watermarked = embedder.embed(img, author_keys)
        attacked = _jpeg_compress(watermarked, 75)
        attacked = np.clip(attacked + rng.normal(0, 5, attacked.shape), 0, 255)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, (
            f"{name} JPEG+noise: conf={result.payload_confidence:.2f}"
        )


class TestDiverseReport:
    """Comprehensive robustness report across all image types."""

    def test_print_report(self, embedder, detector, author_keys, capsys):
        rng = np.random.default_rng(42)
        images = _build_all_images(rng)

        print(f"\n{'=' * 95}")
        print("DIVERSE IMAGE ROBUSTNESS REPORT")
        print(f"{'=' * 95}")
        print(
            f"{'Image':<22} {'PSNR':>6} {'Clean':>7} {'JPGQ75':>7} "
            f"{'JPGQ50':>7} {'Noise':>7} {'Rot90':>7} {'Crop10':>7} {'Combo':>7}"
        )
        print(f"{'-' * 95}")

        for name, img in images.items():
            watermarked = embedder.embed(img, author_keys)
            mse = np.mean((watermarked - img) ** 2)
            psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")

            # Clean
            r_clean = detector.detect(watermarked, author_keys.public_key)

            # JPEG Q75
            r_j75 = detector.detect(_jpeg_compress(watermarked, 75), author_keys.public_key)

            # JPEG Q50
            r_j50 = detector.detect(_jpeg_compress(watermarked, 50), author_keys.public_key)

            # Noise sigma=10
            noisy = np.clip(watermarked + rng.normal(0, 10, watermarked.shape), 0, 255)
            # For RGB, extract Y for detection
            r_noise = detector.detect(noisy, author_keys.public_key)

            # Rotation 90
            if img.ndim == 2:
                h, w = watermarked.shape
                M = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1.0)
                rotated = cv2.warpAffine(
                    watermarked.astype(np.float32),
                    M,
                    (w, h),
                    borderMode=cv2.BORDER_REFLECT_101,
                ).astype(np.float64)
            else:
                h, w = watermarked.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1.0)
                rotated = cv2.warpAffine(
                    watermarked.astype(np.float32),
                    M,
                    (w, h),
                    borderMode=cv2.BORDER_REFLECT_101,
                ).astype(np.float64)
            r_rot = detector.detect(rotated, author_keys.public_key)

            # Crop 10%
            if img.ndim == 2:
                h, w = watermarked.shape
            else:
                h, w = watermarked.shape[:2]
            nh, nw = int(h * 0.9), int(w * 0.9)
            nh = max(nh - nh % 2, 64)
            nw = max(nw - nw % 2, 64)
            cropped = watermarked[
                (h - nh) // 2 : (h - nh) // 2 + nh,
                (w - nw) // 2 : (w - nw) // 2 + nw,
            ].copy()
            r_crop = detector.detect(cropped, author_keys.public_key)

            # Combined: JPEG Q75 + noise sigma=5
            combo = _jpeg_compress(watermarked, 75)
            combo = np.clip(combo + rng.normal(0, 5, combo.shape), 0, 255)
            r_combo = detector.detect(combo, author_keys.public_key)

            def _fmt(r):
                if r.detected:
                    return f"{r.payload_confidence:.2f}*"
                return f"{r.payload_confidence:.2f}"

            print(
                f"{name:<22} {psnr:>5.1f}  "
                f"{_fmt(r_clean):>7} {_fmt(r_j75):>7} {_fmt(r_j50):>7} "
                f"{_fmt(r_noise):>7} {_fmt(r_rot):>7} {_fmt(r_crop):>7} "
                f"{_fmt(r_combo):>7}"
            )

        print(f"{'=' * 95}")
        print("(* = detected)")
