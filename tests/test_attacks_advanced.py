"""Advanced attack tests for extreme robustness validation.

Covers:
- Geometric attacks (perspective, affine, barrel distortion)
- Print-scan simulation
- Removal attacks (median filter, Wiener, histogram equalization, etc.)
- Lighting/color attacks on RGB images
- Combined/chained attacks
"""

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.detect import SigilDetector
from sigil_watermark.keygen import generate_author_keys
from sigil_watermark.config import SigilConfig
from sigil_watermark.color import extract_y_channel


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def author_keys():
    return generate_author_keys(seed=b"advanced-attack-test-32-bytes!!")


@pytest.fixture
def embedder(config):
    return SigilEmbedder(config=config)


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


# --- Image Generators ---


def _make_natural(rng, size=(512, 512)):
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            img[i, j] = (
                128
                + 40 * np.sin(i / 20.0)
                + 30 * np.cos(j / 15.0)
                + 20 * np.sin((i + j) / 25.0)
            )
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255)


def _make_rgb(rng, size=(512, 512)):
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        for i in range(h):
            for j in range(w):
                img[i, j, c] = (
                    128
                    + 40 * np.sin((i + c * 30) / 25)
                    + 30 * np.cos((j + c * 20) / 20)
                )
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255)


# --- Attack Helper Functions ---


def jpeg_compress(image, quality):
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3:
        pil = Image.fromarray(img_uint8, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"), dtype=np.float64)
    else:
        pil = Image.fromarray(img_uint8, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return np.array(Image.open(buf).convert("L"), dtype=np.float64)


def crop_center(image, keep_frac):
    if image.ndim == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    nh, nw = int(h * keep_frac), int(w * keep_frac)
    nh = max(nh - nh % 2, 64)
    nw = max(nw - nw % 2, 64)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    return image[y0 : y0 + nh, x0 : x0 + nw].copy()


def resize_roundtrip(image, scale):
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    if image.ndim == 3:
        small = cv2.resize(image.astype(np.float32), (new_w, new_h))
        return cv2.resize(small, (w, h)).astype(np.float64)
    else:
        small = cv2.resize(image.astype(np.float32), (new_w, new_h))
        return cv2.resize(small, (w, h)).astype(np.float64)


# --- Geometric Attacks ---


class TestPerspectiveAttack:
    """Perspective/projective transforms (simulating viewing angle)."""

    def _perspective_warp(self, image, magnitude=0.05):
        h, w = image.shape[:2]
        rng = np.random.default_rng(123)
        # Source corners
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        # Perturb destination corners
        dx = w * magnitude
        dy = h * magnitude
        dst = src + rng.uniform(-1, 1, src.shape).astype(np.float32) * np.float32(
            [[dx, dy], [dx, dy], [dx, dy], [dx, dy]]
        )
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            image.astype(np.float32), M, (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float64)

    def test_mild_perspective(self, embedder, detector, author_keys):
        """Mild perspective warp (3%) — phone at slight angle."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        warped = self._perspective_warp(watermarked, magnitude=0.03)
        result = detector.detect(warped, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, \
            f"Mild perspective failed: conf={result.payload_confidence:.2f}"

    def test_moderate_perspective(self, embedder, detector, author_keys):
        """Moderate perspective warp (5%)."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        warped = self._perspective_warp(watermarked, magnitude=0.05)
        result = detector.detect(warped, author_keys.public_key)
        assert result.payload_confidence > 0.3 or result.ring_confidence > 0.3, \
            f"Moderate perspective: payload={result.payload_confidence:.2f}, ring={result.ring_confidence:.2f}"


class TestAffineAttack:
    """Affine transforms (shear + rotation + scale)."""

    def _affine_shear(self, image, shear_x=0.05, shear_y=0.0):
        h, w = image.shape[:2]
        M = np.float32([[1, shear_x, 0], [shear_y, 1, 0]])
        return cv2.warpAffine(
            image.astype(np.float32), M, (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float64)

    def test_mild_shear(self, embedder, detector, author_keys):
        """5% horizontal shear."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        sheared = self._affine_shear(watermarked, shear_x=0.05)
        result = detector.detect(sheared, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"Mild shear failed: conf={result.payload_confidence:.2f}"

    def test_combined_shear_rotation(self, embedder, detector, author_keys):
        """Shear + small rotation."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        sheared = self._affine_shear(watermarked, shear_x=0.03)
        h, w = sheared.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0)
        attacked = cv2.warpAffine(
            sheared.astype(np.float32), M, (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float64)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.payload_confidence > 0.25 or result.ring_confidence > 0.3, \
            f"Shear+rotation: payload={result.payload_confidence:.2f}"


class TestBarrelDistortion:
    """Barrel/pincushion distortion (lens distortion)."""

    def _barrel_distort(self, image, k1=0.1):
        h, w = image.shape[:2]
        # Camera matrix (identity-like)
        cam = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
        undist_map1, undist_map2 = cv2.initUndistortRectifyMap(
            cam, dist, None, cam, (w, h), cv2.CV_32FC1
        )
        return cv2.remap(
            image.astype(np.float32), undist_map1, undist_map2,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float64)

    def test_mild_barrel(self, embedder, detector, author_keys):
        """Mild barrel distortion (k1=0.1)."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        distorted = self._barrel_distort(watermarked, k1=0.1)
        result = detector.detect(distorted, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"Mild barrel: conf={result.payload_confidence:.2f}"

    def test_pincushion(self, embedder, detector, author_keys):
        """Pincushion distortion (k1=-0.1)."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        distorted = self._barrel_distort(watermarked, k1=-0.1)
        result = detector.detect(distorted, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"Pincushion: conf={result.payload_confidence:.2f}"


class TestNonUniformScale:
    """Angular stretching / non-uniform scaling."""

    def test_stretch_horizontal(self, embedder, detector, author_keys):
        """Stretch 10% horizontally, then resize back."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        stretched = cv2.resize(
            watermarked.astype(np.float32), (int(w * 1.1), h)
        )
        # Crop back to original width
        x0 = (stretched.shape[1] - w) // 2
        result_img = stretched[:, x0 : x0 + w].astype(np.float64)
        result = detector.detect(result_img, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"H-stretch: conf={result.payload_confidence:.2f}"

    def test_stretch_vertical(self, embedder, detector, author_keys):
        """Stretch 10% vertically, then crop back."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        stretched = cv2.resize(
            watermarked.astype(np.float32), (w, int(h * 1.1))
        )
        y0 = (stretched.shape[0] - h) // 2
        result_img = stretched[y0 : y0 + h, :].astype(np.float64)
        result = detector.detect(result_img, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"V-stretch: conf={result.payload_confidence:.2f}"


# --- Print-Scan Simulation ---


class TestPrintScanSimulation:
    """Simulates print-scan pipeline (based on StegaStamp CVPR 2020)."""

    def _simulate_print_scan(self, image, rng, severity="mild"):
        """Full print-scan pipeline.

        Steps: gamma -> blur -> perspective -> noise -> color jitter -> JPEG -> crop
        """
        result = image.copy()

        if severity == "mild":
            gamma, blur_sigma, persp_mag = 1.1, 0.5, 0.02
            noise_sigma, jitter, jpeg_q, crop_keep = 2, 0.05, 85, 0.95
        elif severity == "moderate":
            gamma, blur_sigma, persp_mag = 1.2, 0.8, 0.04
            noise_sigma, jitter, jpeg_q, crop_keep = 4, 0.10, 75, 0.90
        else:  # severe
            gamma, blur_sigma, persp_mag = 1.3, 1.2, 0.06
            noise_sigma, jitter, jpeg_q, crop_keep = 6, 0.15, 65, 0.85

        is_color = result.ndim == 3

        # 1. Gamma correction (printer response curve)
        result = np.clip(result, 0, 255)
        result = np.power(result / 255.0, gamma) * 255.0

        # 2. Gaussian blur (print dot spread)
        ksize = max(3, int(blur_sigma * 4) | 1)
        result = cv2.GaussianBlur(result.astype(np.float32), (ksize, ksize), blur_sigma)
        result = result.astype(np.float64)

        # 3. Perspective warp (camera angle)
        h, w = result.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dx, dy = w * persp_mag, h * persp_mag
        offsets = rng.uniform(-1, 1, (4, 2)).astype(np.float32)
        dst = src + offsets * np.float32([[dx, dy]])
        M = cv2.getPerspectiveTransform(src, dst)
        result = cv2.warpPerspective(
            result.astype(np.float32), M, (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float64)

        # 4. Additive noise (sensor noise)
        result += rng.normal(0, noise_sigma, result.shape)

        # 5. Color/brightness jitter (lighting)
        result = result * rng.uniform(1 - jitter, 1 + jitter)
        result = np.clip(result, 0, 255)

        # 6. JPEG compression (sharing)
        result = jpeg_compress(result, jpeg_q)

        # 7. Random crop (framing)
        result = crop_center(result, crop_keep)

        return result

    def test_mild_print_scan(self, embedder, detector, author_keys):
        """Mild print-scan: high-quality print, good lighting."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        scanned = self._simulate_print_scan(watermarked, rng, "mild")
        result = detector.detect(scanned, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"Mild print-scan: conf={result.payload_confidence:.2f}"

    def test_moderate_print_scan(self, embedder, detector, author_keys):
        """Moderate print-scan: typical office scan."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        scanned = self._simulate_print_scan(watermarked, rng, "moderate")
        result = detector.detect(scanned, author_keys.public_key)
        assert result.payload_confidence > 0.25 or result.ring_confidence > 0.3, \
            f"Moderate print-scan: payload={result.payload_confidence:.2f}, ring={result.ring_confidence:.2f}"

    def test_severe_print_scan(self, embedder, detector, author_keys):
        """Severe print-scan: low quality, bad angle, poor lighting."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        scanned = self._simulate_print_scan(watermarked, rng, "severe")
        result = detector.detect(scanned, author_keys.public_key)
        # Severe is very hard — ring detection is our fallback
        assert result.payload_confidence > 0.2 or result.ring_confidence > 0.2, \
            f"Severe print-scan: payload={result.payload_confidence:.2f}, ring={result.ring_confidence:.2f}"

    def test_rgb_print_scan(self, embedder, detector, author_keys):
        """Print-scan on RGB image."""
        rng = np.random.default_rng(42)
        img = _make_rgb(rng)
        watermarked = embedder.embed(img, author_keys)
        scanned = self._simulate_print_scan(watermarked, rng, "mild")
        result = detector.detect(scanned, author_keys.public_key)
        assert result.payload_confidence > 0.25 or result.ring_confidence > 0.3, \
            f"RGB print-scan: payload={result.payload_confidence:.2f}"


# --- Removal Attacks ---


class TestMedianFilter:
    """Median filter attacks (non-linear, hard to survive)."""

    def test_median_3x3(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        filtered = cv2.medianBlur(watermarked.astype(np.float32), 3).astype(np.float64)
        result = detector.detect(filtered, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"Median 3x3: conf={result.payload_confidence:.2f}"

    def test_median_5x5(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        # Need uint8 for medianBlur with ksize > 3 on float
        img_u8 = np.clip(watermarked, 0, 255).astype(np.uint8)
        filtered = cv2.medianBlur(img_u8, 5).astype(np.float64)
        result = detector.detect(filtered, author_keys.public_key)
        assert result.payload_confidence > 0.25 or result.ring_confidence > 0.3, \
            f"Median 5x5: payload={result.payload_confidence:.2f}"


class TestGaussianBlurAttack:
    """Gaussian blur removal attacks."""

    @pytest.mark.parametrize("sigma", [1.0, 2.0])
    def test_gaussian_blur(self, embedder, detector, author_keys, sigma):
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        ksize = max(3, int(sigma * 6) | 1)
        blurred = cv2.GaussianBlur(
            watermarked.astype(np.float32), (ksize, ksize), sigma
        ).astype(np.float64)
        result = detector.detect(blurred, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"Gaussian blur sigma={sigma}: conf={result.payload_confidence:.2f}"

    def test_heavy_blur_sigma3(self, embedder, detector, author_keys):
        """Heavy blur (sigma=3) — very destructive."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        blurred = cv2.GaussianBlur(
            watermarked.astype(np.float32), (19, 19), 3.0
        ).astype(np.float64)
        result = detector.detect(blurred, author_keys.public_key)
        assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3, \
            f"Heavy blur: payload={result.payload_confidence:.2f}"


class TestHistogramEqualization:
    """Histogram equalization attack."""

    def test_histeq(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        img_u8 = np.clip(watermarked, 0, 255).astype(np.uint8)
        equalized = cv2.equalizeHist(img_u8).astype(np.float64)
        result = detector.detect(equalized, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"Histogram equalization: conf={result.payload_confidence:.2f}"

    def test_clahe(self, embedder, detector, author_keys):
        """CLAHE (adaptive histogram equalization)."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        img_u8 = np.clip(watermarked, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(img_u8).astype(np.float64)
        result = detector.detect(equalized, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"CLAHE: conf={result.payload_confidence:.2f}"


class TestGammaCorrection:
    """Gamma correction attacks."""

    @pytest.mark.parametrize("gamma", [0.5, 1.5, 2.0])
    def test_gamma(self, embedder, detector, author_keys, gamma):
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        corrected = np.clip(watermarked, 0, 255)
        corrected = np.power(corrected / 255.0, gamma) * 255.0
        result = detector.detect(corrected, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"Gamma {gamma}: conf={result.payload_confidence:.2f}"


class TestBitDepthReduction:
    """Bit depth reduction (quantization) attacks."""

    @pytest.mark.parametrize("bits", [6, 4])
    def test_reduced_depth(self, embedder, detector, author_keys, bits):
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        # Quantize to fewer bits
        levels = 2**bits
        quantized = np.round(watermarked / 255.0 * (levels - 1)) / (levels - 1) * 255.0
        result = detector.detect(quantized, author_keys.public_key)
        if bits == 6:
            assert result.detected or result.payload_confidence > 0.3, \
                f"{bits}-bit: conf={result.payload_confidence:.2f}"
        else:  # 4-bit is harsh
            assert result.payload_confidence > 0.2 or result.ring_confidence > 0.2, \
                f"{bits}-bit: payload={result.payload_confidence:.2f}"


class TestFrequencyNotchFilter:
    """Attempt to remove ring frequencies with a notch filter."""

    def test_notch_at_ring_frequencies(self, embedder, detector, author_keys, config):
        """Try to notch out the DFT ring frequencies."""
        from sigil_watermark.keygen import derive_ring_radii

        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)

        # Get the ring radii that were embedded
        ring_radii = derive_ring_radii(author_keys.public_key, config=config)

        # Notch filter in frequency domain
        h, w = watermarked.shape
        f = np.fft.fftshift(np.fft.fft2(watermarked))
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        freq_dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / min(h, w)

        for radius in ring_radii:
            # Notch out a band around each ring radius
            notch = 1.0 - np.exp(-((freq_dist - radius) ** 2) / (2 * 0.005**2))
            f *= notch

        filtered = np.real(np.fft.ifft2(np.fft.ifftshift(f)))
        result = detector.detect(filtered, author_keys.public_key)
        # Notch removes rings but payload in DWT should partially survive
        assert result.payload_confidence > 0.25, \
            f"Notch filter: payload={result.payload_confidence:.2f}"


# --- Color/Lighting Attacks ---


class TestColorAttacks:
    """Color-specific attacks on RGB images."""

    def test_hue_rotation_90(self, embedder, detector, author_keys):
        """Rotate hue by 90 degrees — Y channel should be partially preserved."""
        rng = np.random.default_rng(42)
        img = _make_rgb(rng)
        watermarked = embedder.embed(img, author_keys)
        # Convert to HSV, rotate hue, convert back
        img_u8 = np.clip(watermarked, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + 45) % 180  # 90 degrees in OpenCV
        rotated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float64)
        result = detector.detect(rotated, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"Hue rotation 90: conf={result.payload_confidence:.2f}"

    def test_desaturation(self, embedder, detector, author_keys):
        """Full desaturation (convert to grayscale) — Y channel preserved."""
        rng = np.random.default_rng(42)
        img = _make_rgb(rng)
        watermarked = embedder.embed(img, author_keys)
        gray = extract_y_channel(watermarked)
        result = detector.detect(gray, author_keys.public_key)
        assert result.detected, f"Desaturation failed: {result}"

    def test_oversaturation(self, embedder, detector, author_keys):
        """2x saturation boost."""
        rng = np.random.default_rng(42)
        img = _make_rgb(rng)
        watermarked = embedder.embed(img, author_keys)
        img_u8 = np.clip(watermarked, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float64)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.0, 0, 255)
        saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float64)
        result = detector.detect(saturated, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"Oversaturation: conf={result.payload_confidence:.2f}"

    def test_white_balance_shift(self, embedder, detector, author_keys):
        """Simulate white balance shift (warm: boost R, reduce B)."""
        rng = np.random.default_rng(42)
        img = _make_rgb(rng)
        watermarked = embedder.embed(img, author_keys)
        shifted = watermarked.copy()
        shifted[:, :, 0] = np.clip(shifted[:, :, 0] * 1.15, 0, 255)  # R up
        shifted[:, :, 2] = np.clip(shifted[:, :, 2] * 0.85, 0, 255)  # B down
        result = detector.detect(shifted, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"WB shift: conf={result.payload_confidence:.2f}"

    def test_vignetting(self, embedder, detector, author_keys):
        """Vignetting (center-weighted brightness falloff)."""
        rng = np.random.default_rng(42)
        img = _make_rgb(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape[:2]
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (min(h, w) / 2)
        vignette = np.clip(1.0 - 0.4 * r**2, 0.3, 1.0)
        if watermarked.ndim == 3:
            vignette = vignette[:, :, np.newaxis]
        vignetted = np.clip(watermarked * vignette, 0, 255)
        result = detector.detect(vignetted, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"Vignetting: conf={result.payload_confidence:.2f}"


# --- Combined/Chained Attacks ---


class TestCombinedAttacksAdvanced:
    """Multi-step attack chains simulating real-world scenarios."""

    def test_jpeg_crop_resize(self, embedder, detector, author_keys):
        """JPEG Q50 + 25% crop + 0.5x resize."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        attacked = jpeg_compress(watermarked, 50)
        attacked = crop_center(attacked, 0.75)
        attacked = resize_roundtrip(attacked, 0.5)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3, \
            f"JPEG+crop+resize: payload={result.payload_confidence:.2f}"

    def test_rotation_noise_jpeg(self, embedder, detector, author_keys):
        """5-degree rotation + noise sigma=10 + JPEG Q75."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 5.0, 1.0)
        attacked = cv2.warpAffine(
            watermarked.astype(np.float32), M, (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float64)
        attacked += rng.normal(0, 10, attacked.shape)
        attacked = np.clip(attacked, 0, 255)
        attacked = jpeg_compress(attacked, 75)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3, \
            f"Rot+noise+JPEG: payload={result.payload_confidence:.2f}"

    def test_perspective_brightness_jpeg(self, embedder, detector, author_keys):
        """Perspective warp + brightness shift + JPEG Q50."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dx, dy = w * 0.04, h * 0.04
        offsets = rng.uniform(-1, 1, (4, 2)).astype(np.float32)
        dst = src + offsets * np.float32([[dx, dy]])
        M = cv2.getPerspectiveTransform(src, dst)
        attacked = cv2.warpPerspective(
            watermarked.astype(np.float32), M, (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float64)
        attacked = np.clip(attacked + 25, 0, 255)
        attacked = jpeg_compress(attacked, 50)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.payload_confidence > 0.2 or result.ring_confidence > 0.2, \
            f"Persp+bright+JPEG: payload={result.payload_confidence:.2f}"

    def test_blur_gamma_jpeg_crop(self, embedder, detector, author_keys):
        """Blur + gamma + JPEG + crop — realistic social media pipeline."""
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)
        # Mild blur
        attacked = cv2.GaussianBlur(
            watermarked.astype(np.float32), (3, 3), 0.8
        ).astype(np.float64)
        # Gamma
        attacked = np.power(np.clip(attacked, 0, 255) / 255.0, 1.1) * 255.0
        # JPEG
        attacked = jpeg_compress(attacked, 75)
        # Small crop
        attacked = crop_center(attacked, 0.95)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, \
            f"Social media pipeline: conf={result.payload_confidence:.2f}"

    def test_rgb_full_pipeline(self, embedder, detector, author_keys):
        """Full attack pipeline on RGB: blur + WB + JPEG + crop."""
        rng = np.random.default_rng(42)
        img = _make_rgb(rng)
        watermarked = embedder.embed(img, author_keys)
        # Blur
        attacked = cv2.GaussianBlur(
            watermarked.astype(np.float32), (3, 3), 0.5
        ).astype(np.float64)
        # White balance
        attacked[:, :, 0] = np.clip(attacked[:, :, 0] * 1.1, 0, 255)
        attacked[:, :, 2] = np.clip(attacked[:, :, 2] * 0.9, 0, 255)
        # JPEG
        attacked = jpeg_compress(attacked, 75)
        # Crop
        attacked = crop_center(attacked, 0.90)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.payload_confidence > 0.25 or result.ring_confidence > 0.3, \
            f"RGB full pipeline: payload={result.payload_confidence:.2f}"


# --- Robustness Report ---


class TestAdvancedRobustnessReport:
    """Print comprehensive robustness matrix (informational, always passes)."""

    def test_print_advanced_report(self, embedder, detector, author_keys, config, capsys):
        rng = np.random.default_rng(42)
        img = _make_natural(rng)
        watermarked = embedder.embed(img, author_keys)

        from sigil_watermark.keygen import derive_ring_radii

        mse = np.mean((watermarked - img) ** 2)
        psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")

        attacks = {}

        # Clean
        attacks["No attack"] = watermarked

        # Geometric
        h, w = watermarked.shape
        M90 = cv2.getRotationMatrix2D((w / 2, h / 2), 90, 1.0)
        attacks["Rotation 90"] = cv2.warpAffine(
            watermarked.astype(np.float32), M90, (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float64)
        M5 = cv2.getRotationMatrix2D((w / 2, h / 2), 5, 1.0)
        attacks["Rotation 5"] = cv2.warpAffine(
            watermarked.astype(np.float32), M5, (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float64)

        # Perspective
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[-10, -10], [10, -5], [5, 10], [-8, 8]])
        Mp = cv2.getPerspectiveTransform(src, dst)
        attacks["Perspective 3%"] = cv2.warpPerspective(
            watermarked.astype(np.float32), Mp, (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float64)

        # Filters
        attacks["Median 3x3"] = cv2.medianBlur(
            watermarked.astype(np.float32), 3
        ).astype(np.float64)
        attacks["Blur sigma=1"] = cv2.GaussianBlur(
            watermarked.astype(np.float32), (7, 7), 1.0
        ).astype(np.float64)
        attacks["Blur sigma=2"] = cv2.GaussianBlur(
            watermarked.astype(np.float32), (13, 13), 2.0
        ).astype(np.float64)

        # Tonal
        attacks["Gamma 0.5"] = np.power(np.clip(watermarked, 0, 255) / 255.0, 0.5) * 255
        attacks["Gamma 2.0"] = np.power(np.clip(watermarked, 0, 255) / 255.0, 2.0) * 255
        img_u8 = np.clip(watermarked, 0, 255).astype(np.uint8)
        attacks["HistEq"] = cv2.equalizeHist(img_u8).astype(np.float64)

        # Quantization
        attacks["6-bit"] = np.round(watermarked / 255 * 63) / 63 * 255
        attacks["4-bit"] = np.round(watermarked / 255 * 15) / 15 * 255

        # JPEG
        attacks["JPEG Q50"] = jpeg_compress(watermarked, 50)
        attacks["JPEG Q20"] = jpeg_compress(watermarked, 20)

        # Crop
        attacks["Crop 25%"] = crop_center(watermarked, 0.75)
        attacks["Crop 50%"] = crop_center(watermarked, 0.50)

        # Combined
        attacks["JPEG50+Crop25%"] = crop_center(jpeg_compress(watermarked, 50), 0.75)

        print(f"\n{'=' * 80}")
        print("ADVANCED ROBUSTNESS REPORT")
        print(f"{'=' * 80}")
        print(f"Image: 512x512 synthetic | PSNR: {psnr:.1f} dB")
        print(f"{'=' * 80}")
        print(
            f"{'Attack':<20} {'Detected':>8} {'Beacon':>8} "
            f"{'Payload':>8} {'Ring':>8} {'AuthID':>8}"
        )
        print(f"{'-' * 80}")

        for name, attacked in attacks.items():
            r = detector.detect(attacked, author_keys.public_key)
            print(
                f"{name:<20} "
                f"{'YES' if r.detected else 'no':>8} "
                f"{'YES' if r.beacon_found else 'no':>8} "
                f"{r.payload_confidence:>8.2f} "
                f"{r.ring_confidence:>8.2f} "
                f"{'YES' if r.author_id_match else 'no':>8}"
            )

        print(f"{'=' * 80}")
