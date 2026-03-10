"""Shared test fixtures for sigil watermark tests.

Provides:
- Natural image generators (vectorized, fast) for diverse test coverage
- Multi-key fixtures for cross-key false positive testing
- Shared embedder/detector fixtures
- Utility functions (PSNR, JPEG roundtrip)
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


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def rng():
    """Seeded RNG for deterministic tests."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def embedder(config):
    return SigilEmbedder(config=config)


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


# ---------------------------------------------------------------------------
# Author key fixtures — multiple keys for cross-key testing
# ---------------------------------------------------------------------------

AUTHOR_SEEDS = [
    b"test-author-key-alpha-32-bytes!",
    b"test-author-key-bravo-32-bytes!",
    b"test-author-key-charlie-32byte!",
]


@pytest.fixture
def author_keys():
    """Default author keys (seed alpha)."""
    return generate_author_keys(seed=AUTHOR_SEEDS[0])


@pytest.fixture
def author_keys_b():
    """Second author keys (seed bravo)."""
    return generate_author_keys(seed=AUTHOR_SEEDS[1])


@pytest.fixture
def author_keys_c():
    """Third author keys (seed charlie)."""
    return generate_author_keys(seed=AUTHOR_SEEDS[2])


@pytest.fixture(params=AUTHOR_SEEDS, ids=["key-alpha", "key-bravo", "key-charlie"])
def multi_author_keys(request):
    """Parametrized fixture: runs the test with each of 3 author keys."""
    return generate_author_keys(seed=request.param)


# ---------------------------------------------------------------------------
# Fast vectorized image generators
# ---------------------------------------------------------------------------


def make_gradient_image(h=512, w=512):
    """Smooth diagonal gradient — easy to embed, hard to hide in."""
    y = np.linspace(0, 255, h).reshape(-1, 1)
    x = np.linspace(0, 255, w).reshape(1, -1)
    return (y * 0.5 + x * 0.5).astype(np.float64)


def make_texture_image(h=512, w=512, seed=42):
    """High-frequency texture (multi-frequency sinusoidal + noise)."""
    rng = np.random.default_rng(seed)
    yy = np.arange(h).reshape(-1, 1)
    xx = np.arange(w).reshape(1, -1)
    img = (
        128.0
        + 50 * np.sin(yy / 3.0) * np.cos(xx / 4.0)
        + 30 * np.sin((yy * 7 + xx * 11) / 50.0)
    )
    img += rng.normal(0, 5, (h, w))
    return np.clip(img, 0, 255)


def make_edges_image(h=512, w=512):
    """Strong edges — checkerboard pattern."""
    yy = np.arange(h).reshape(-1, 1)
    xx = np.arange(w).reshape(1, -1)
    return np.where((yy // 64 + xx // 64) % 2 == 0, 240.0, 15.0)


def make_highfreq_image(h=512, w=512, seed=42):
    """Very high frequency content — worst case for perceptual masking."""
    rng = np.random.default_rng(seed)
    yy = np.arange(h).reshape(-1, 1)
    xx = np.arange(w).reshape(1, -1)
    img = (
        128.0
        + 40 * np.sin(yy / 2.0) * np.cos(xx / 2.5)
        + 25 * np.sin(yy * 0.8 + xx * 1.1)
        + 20 * np.cos(yy * 1.3 - xx * 0.7)
    )
    img += rng.normal(0, 12, (h, w))
    return np.clip(img, 0, 255)


def make_photo_like_image(h=512, w=512, seed=42):
    """Photo-like: mixed frequencies, gradient sky, textured ground."""
    rng = np.random.default_rng(seed)
    yy = np.arange(h).reshape(-1, 1).astype(np.float64)
    xx = np.arange(w).reshape(1, -1).astype(np.float64)
    # Sky region (smooth gradient, top half)
    sky = 180 + 30 * (yy / h) + 10 * np.sin(xx / 40.0)
    # Ground region (textured, bottom half)
    ground = (
        100
        + 30 * np.sin(yy / 15.0) * np.cos(xx / 12.0)
        + 20 * np.sin((yy + xx) / 25.0)
        + 15 * np.cos(yy / 7.0)
    )
    # Blend: top half sky, bottom half ground
    blend = np.where(yy < h / 2, sky, ground)
    blend += rng.normal(0, 8, (h, w))
    return np.clip(blend, 0, 255)


def make_dark_image(h=512, w=512, seed=42):
    """Dark image — low dynamic range, tests low-energy embedding."""
    rng = np.random.default_rng(seed)
    yy = np.arange(h).reshape(-1, 1).astype(np.float64)
    xx = np.arange(w).reshape(1, -1).astype(np.float64)
    img = 30 + 15 * np.sin(yy / 30.0) + 10 * np.cos(xx / 25.0)
    img += rng.normal(0, 5, (h, w))
    return np.clip(img, 0, 255)


def make_natural_scene(h=512, w=512, seed=42):
    """Multi-frequency natural-like scene (legacy default)."""
    rng = np.random.default_rng(seed)
    yy = np.arange(h).reshape(-1, 1).astype(np.float64)
    xx = np.arange(w).reshape(1, -1).astype(np.float64)
    img = (
        128.0
        + 40 * np.sin(yy / 20.0)
        + 30 * np.cos(xx / 15.0)
        + 20 * np.sin((yy + xx) / 25.0)
    )
    img += rng.normal(0, 8, (h, w))
    return np.clip(img, 0, 255)


def make_photo_like_rgb(h=512, w=512, seed=42):
    """Photo-like RGB image (float64, 0-255)."""
    rng = np.random.default_rng(seed)
    yy = np.arange(h).reshape(-1, 1).astype(np.float64)
    xx = np.arange(w).reshape(1, -1).astype(np.float64)
    img = np.zeros((h, w, 3), dtype=np.float64)
    for c in range(3):
        freq = 15.0 + c * 5
        img[:, :, c] = (
            128
            + 40 * np.sin(yy / freq)
            + 30 * np.cos(xx / (freq + 3))
            + 15 * np.sin((yy + xx) / 25.0)
        )
        img[:, :, c] += rng.normal(0, 8, (h, w))
    return np.clip(img, 0, 255)


# Registry for parameterized fixtures
NATURAL_IMAGE_GENERATORS = {
    "gradient": make_gradient_image,
    "texture": make_texture_image,
    "edges": make_edges_image,
    "highfreq": make_highfreq_image,
    "photo_like": make_photo_like_image,
    "dark": make_dark_image,
    "natural_scene": make_natural_scene,
}

# Images that lack spectral diversity or dynamic range for reliable full-pipeline
# watermarking. Included in tests to document known limitations, but need relaxed
# thresholds. "dark" has very low dynamic range (30±15) making watermark fragile.
PATHOLOGICAL_IMAGES = {"gradient", "edges", "highfreq", "dark"}

REALISTIC_IMAGE_GENERATORS = {
    k: v for k, v in NATURAL_IMAGE_GENERATORS.items()
    if k not in PATHOLOGICAL_IMAGES
}


@pytest.fixture(params=list(NATURAL_IMAGE_GENERATORS.keys()))
def natural_image(request):
    """Parametrized: runs the test with each natural image type (512x512 grayscale)."""
    name = request.param
    return name, NATURAL_IMAGE_GENERATORS[name]()


@pytest.fixture(params=list(REALISTIC_IMAGE_GENERATORS.keys()))
def realistic_image(request):
    """Parametrized: only realistic image types that support full watermark fidelity."""
    name = request.param
    return name, REALISTIC_IMAGE_GENERATORS[name]()


@pytest.fixture(params=list(NATURAL_IMAGE_GENERATORS.keys()))
def natural_image_name(request):
    """Just the image name, for combining with other params."""
    return request.param


# ---------------------------------------------------------------------------
# Legacy fixtures (kept for backward compat with existing tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def small_gray_image(rng):
    """256x256 grayscale image with texture."""
    return make_natural_scene(256, 256, seed=42)


@pytest.fixture
def medium_gray_image(rng):
    """512x512 grayscale image with mixed texture and flat regions."""
    img = np.zeros((512, 512), dtype=np.float64)
    yy = np.arange(512).reshape(-1, 1).astype(np.float64)
    xx = np.arange(512).reshape(1, -1).astype(np.float64)
    # Textured quadrant
    img[:256, :256] = 100 + 50 * np.sin(yy[:256] / 10.0) * np.cos(xx[:, :256] / 8.0)
    # Flat quadrant
    img[:256, 256:] = 180
    # Gradient quadrant
    img[256:, :256] = (yy[256:] - 256) / 256.0 * 200 + 28
    # Noisy quadrant
    img[256:, 256:] = 128 + rng.normal(0, 30, (256, 256))
    return np.clip(img, 0, 255)


@pytest.fixture
def color_image(rng):
    """512x512 RGB color image (float64) resembling a photo."""
    return make_photo_like_rgb(512, 512, seed=42)


@pytest.fixture
def flat_image():
    """512x512 flat gray image — worst case for perceptual masking."""
    return np.full((512, 512), 128.0, dtype=np.float64)


@pytest.fixture
def synthetic_images(rng):
    """Collection of diverse synthetic test images (grayscale, float64, 0-255)."""
    images = {}
    images["noise"] = rng.uniform(0, 255, (512, 512))

    yy = np.arange(512).reshape(-1, 1).astype(np.float64)
    xx = np.arange(512).reshape(1, -1).astype(np.float64)

    images["stripes"] = 128 + 64 * np.sin(yy * 2 * np.pi / 32) + 0 * xx
    images["checkerboard"] = np.where((yy // 32 + xx // 32) % 2 == 0, 200.0, 55.0)

    y, x = np.ogrid[-256:256, -256:256]
    r = np.sqrt(x * x + y * y)
    images["radial"] = np.clip(r / np.sqrt(2) / 256 * 255, 0, 255)

    scene = (
        120
        + 30 * np.sin(yy / 40.0) * np.cos(xx / 30.0)
        + 20 * np.sin(yy / 7.0)
        + 15 * np.cos(xx / 11.0)
    )
    scene = scene + rng.normal(0, 8, (512, 512))
    images["scene"] = np.clip(scene, 0, 255)

    return images


# ---------------------------------------------------------------------------
# Utility functions (shared across test files)
# ---------------------------------------------------------------------------


def psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((original - modified) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255.0**2 / mse))


def jpeg_roundtrip_gray(image: np.ndarray, quality: int = 99) -> np.ndarray:
    """JPEG roundtrip for grayscale: float64 -> uint8 -> JPEG -> float64."""
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.float64)


def jpeg_roundtrip_rgb(image: np.ndarray, quality: int = 99) -> np.ndarray:
    """JPEG roundtrip for RGB: float64 -> uint8 BGR -> JPEG -> RGB float64."""
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded_bgr = cv2.imdecode(np.frombuffer(encoded.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    return decoded_rgb


def png_roundtrip_rgb(image: np.ndarray) -> np.ndarray:
    """Lossless PNG roundtrip: RGB float64 -> uint8 BGR -> PNG -> RGB float64."""
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".png", img_bgr)
    decoded_bgr = cv2.imdecode(np.frombuffer(encoded.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    return decoded_rgb
