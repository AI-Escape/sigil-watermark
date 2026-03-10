"""Social media platform simulation tests.

Simulates the image processing pipelines of major platforms:
- Instagram-like: resize + sharpen + JPEG Q80
- Twitter/X-like: resize + JPEG Q85
- TikTok-like: heavy compression + resize
- Facebook-like: resize + strip metadata + JPEG Q71
- Pinterest-like: resize to fixed width + JPEG
- WhatsApp-like: aggressive compression + resize
- Multiple re-shares (image uploaded, downloaded, re-uploaded)
"""

import io

import cv2
import numpy as np
import pytest
from PIL import Image, ImageFilter

from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.detect import SigilDetector
from sigil_watermark.keygen import generate_author_keys
from sigil_watermark.config import SigilConfig


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def author_keys():
    return generate_author_keys(seed=b"social-media-test-author32bytes!")


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
                128
                + 40 * np.sin(i / 20.0)
                + 30 * np.cos(j / 15.0)
                + 20 * np.sin((i + j) / 25.0)
            )
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255)


def _make_rgb_image(rng, size=(512, 512)):
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
    img += rng.normal(0, 5, img.shape)
    return np.clip(img, 0, 255)


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


def _resize_to(image, new_h, new_w):
    """Resize image to specific dimensions."""
    if image.ndim == 3:
        return cv2.resize(
            image.astype(np.float32), (new_w, new_h)
        ).astype(np.float64)
    else:
        return cv2.resize(
            image.astype(np.float32), (new_w, new_h)
        ).astype(np.float64)


def _sharpen(image, amount=1.0):
    """Unsharp masking sharpening."""
    blurred = cv2.GaussianBlur(
        image.astype(np.float32), (0, 0), 3
    ).astype(np.float64)
    sharpened = image + amount * (image - blurred)
    return np.clip(sharpened, 0, 255)


# --- Platform Simulations ---


class TestInstagramPipeline:
    """Instagram: resize to max 1080px, sharpen, JPEG ~Q80."""

    def test_instagram_gray(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng, size=(1024, 1024))
        watermarked = embedder.embed(img, author_keys)
        # Resize to 1080px max (keeping as 512 since we start smaller)
        attacked = _resize_to(watermarked, 512, 512)
        attacked = _sharpen(attacked, amount=0.5)
        attacked = _jpeg_compress(attacked, 80)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Instagram gray: conf={result.payload_confidence:.2f}"
        )

    def test_instagram_rgb(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng)
        watermarked = embedder.embed(img, author_keys)
        attacked = _sharpen(watermarked, amount=0.5)
        attacked = _jpeg_compress(attacked, 80)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Instagram RGB: conf={result.payload_confidence:.2f}"
        )


class TestTwitterPipeline:
    """Twitter/X: resize, JPEG ~Q85, minor color shifts."""

    def test_twitter_standard(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        attacked = _jpeg_compress(watermarked, 85)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected, f"Twitter: conf={result.payload_confidence:.2f}"

    def test_twitter_rgb(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng)
        watermarked = embedder.embed(img, author_keys)
        # Color shift (Twitter applies slight auto-adjust)
        attacked = watermarked.copy()
        attacked[:, :, 0] = np.clip(attacked[:, :, 0] * 1.02, 0, 255)
        attacked = _jpeg_compress(attacked, 85)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4


class TestFacebookPipeline:
    """Facebook: resize, JPEG Q71 (notoriously aggressive)."""

    def test_facebook_standard(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        # Facebook uses Q71 JPEG
        attacked = _jpeg_compress(watermarked, 71)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Facebook: conf={result.payload_confidence:.2f}"
        )

    def test_facebook_with_resize(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng, size=(1024, 1024))
        watermarked = embedder.embed(img, author_keys)
        # Resize down to 960px then JPEG Q71
        attacked = _resize_to(watermarked, 512, 512)
        attacked = _jpeg_compress(attacked, 71)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3


class TestWhatsAppPipeline:
    """WhatsApp: very aggressive compression + resize."""

    def test_whatsapp(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        # WhatsApp reduces to ~600px and uses Q~60
        attacked = _resize_to(watermarked, 400, 400)
        attacked = _jpeg_compress(attacked, 60)
        # Resize back for detection
        attacked = _resize_to(attacked, 512, 512)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.payload_confidence > 0.2 or result.ring_confidence > 0.2, (
            f"WhatsApp: payload={result.payload_confidence:.2f}, "
            f"ring={result.ring_confidence:.2f}"
        )


class TestPinterestPipeline:
    """Pinterest: resize to fixed width, JPEG Q80."""

    def test_pinterest(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        attacked = _jpeg_compress(watermarked, 80)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected, f"Pinterest: conf={result.payload_confidence:.2f}"


# --- Re-sharing / Multiple Saves ---


class TestMultipleResaves:
    """Simulate image being saved/uploaded multiple times."""

    def test_double_jpeg(self, embedder, detector, author_keys):
        """Saved as JPEG twice (download + re-upload)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        attacked = _jpeg_compress(watermarked, 85)
        attacked = _jpeg_compress(attacked, 85)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected, f"Double JPEG: conf={result.payload_confidence:.2f}"

    def test_triple_jpeg(self, embedder, detector, author_keys):
        """Saved as JPEG three times."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        attacked = _jpeg_compress(watermarked, 80)
        attacked = _jpeg_compress(attacked, 80)
        attacked = _jpeg_compress(attacked, 80)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Triple JPEG: conf={result.payload_confidence:.2f}"
        )

    def test_decreasing_quality_chain(self, embedder, detector, author_keys):
        """JPEG Q95 -> Q85 -> Q75 -> Q65 (progressively worse shares)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        attacked = watermarked
        for q in [95, 85, 75, 65]:
            attacked = _jpeg_compress(attacked, q)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, (
            f"Decreasing Q chain: conf={result.payload_confidence:.2f}"
        )

    def test_cross_platform_sharing(self, embedder, detector, author_keys):
        """Instagram -> download -> Twitter -> download."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        # Instagram pipeline
        attacked = _sharpen(watermarked, amount=0.5)
        attacked = _jpeg_compress(attacked, 80)
        # Twitter pipeline
        attacked = _jpeg_compress(attacked, 85)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3

    def test_reshare_with_crop(self, embedder, detector, author_keys):
        """Upload -> crop (user reframing) -> re-upload."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        # First upload
        attacked = _jpeg_compress(watermarked, 80)
        # User crops 10% from sides
        h, w = attacked.shape
        nh, nw = int(h * 0.9), int(w * 0.9)
        nh = max(nh - nh % 2, 64)
        nw = max(nw - nw % 2, 64)
        attacked = attacked[
            (h - nh) // 2 : (h - nh) // 2 + nh,
            (w - nw) // 2 : (w - nw) // 2 + nw,
        ].copy()
        # Re-upload
        attacked = _jpeg_compress(attacked, 80)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.ring_confidence > 0.3 or result.payload_confidence > 0.3


# --- Full Pipeline with Color ---


class TestRGBPlatformPipelines:
    """Platform simulations on RGB images specifically."""

    def test_instagram_full_rgb(self, embedder, detector, author_keys):
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng)
        watermarked = embedder.embed(img, author_keys)
        # Slight warm filter (Instagram aesthetic)
        attacked = watermarked.copy()
        attacked[:, :, 0] = np.clip(attacked[:, :, 0] * 1.05, 0, 255)  # Warm R
        attacked[:, :, 2] = np.clip(attacked[:, :, 2] * 0.95, 0, 255)  # Cool B
        # Sharpen + vignette
        attacked = _sharpen(attacked, 0.3)
        # Vignette
        h, w = attacked.shape[:2]
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2, w / 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (min(h, w) / 2)
        vignette = np.clip(1.0 - 0.3 * r**2, 0.3, 1.0)[:, :, np.newaxis]
        attacked = np.clip(attacked * vignette, 0, 255)
        # JPEG
        attacked = _jpeg_compress(attacked, 80)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3

    def test_tiktok_rgb(self, embedder, detector, author_keys):
        """TikTok: heavy compression + potential brightness boost."""
        rng = np.random.default_rng(42)
        img = _make_rgb_image(rng)
        watermarked = embedder.embed(img, author_keys)
        attacked = np.clip(watermarked * 1.05 + 5, 0, 255)  # Brightness boost
        attacked = _jpeg_compress(attacked, 70)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3


# --- Resize Variants ---


class TestResizeVariants:
    """Different resize algorithms and ratios."""

    @pytest.mark.parametrize("scale", [0.75, 0.5, 0.33])
    def test_downscale_then_detect(self, embedder, detector, author_keys, scale):
        """Detect directly on downscaled image (no upscale back)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        h, w = watermarked.shape
        new_h = max(int(h * scale) - int(h * scale) % 2, 64)
        new_w = max(int(w * scale) - int(w * scale) % 2, 64)
        attacked = _resize_to(watermarked, new_h, new_w)
        result = detector.detect(attacked, author_keys.public_key)
        if scale >= 0.5:
            assert result.detected or result.payload_confidence > 0.3, (
                f"Scale {scale}: conf={result.payload_confidence:.2f}"
            )
        else:
            assert result.payload_confidence > 0.2 or result.ring_confidence > 0.2

    def test_upscale_2x(self, embedder, detector, author_keys):
        """Upscale 2x (AI upscaling simulation)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng, size=(256, 256))
        watermarked = embedder.embed(img, author_keys)
        attacked = _resize_to(watermarked, 512, 512)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3

    def test_non_square_resize(self, embedder, detector, author_keys):
        """Resize to non-square aspect ratio."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        # 512x512 -> 512x384 (4:3 aspect)
        attacked = _resize_to(watermarked, 384, 512)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3
