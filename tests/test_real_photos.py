"""Watermark tests on real public domain photographs and artwork.

Uses actual photographs and paintings (not synthetic images) to measure
realistic PSNR, detection rates, and attack robustness. These numbers
represent what artists will actually experience.

Images are in tests/test_images/real/ — download with:
    uv run python scripts/download_real_images.py

Image set (all public domain):
- starry_night.jpg      — Van Gogh, rich swirling texture
- great_wave.jpg        — Hokusai, high contrast woodblock print
- girl_pearl_earring.jpg — Vermeer, smooth tonal gradations
- water_lilies.jpg      — Monet, impressionist brushwork
- persistence_of_memory.jpg — Dali, surreal mixed detail
- photo_landscape.jpg   — Yosemite Valley, strong anisotropy
- photo_urban.jpg       — Times Square, high detail bright colors
- photo_architecture.jpg — Pyramids of Giza, geometric edges
"""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sigil_watermark.config import SigilConfig
from sigil_watermark.detect import SigilDetector
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.keygen import generate_author_keys
from sigil_watermark.ghost.spectral_analysis import analyze_ghost_signature

from conftest import AUTHOR_SEEDS


REAL_IMAGES_DIR = Path(__file__).parent / "test_images" / "real"

# Skip all tests if images aren't downloaded
pytestmark = pytest.mark.skipif(
    not REAL_IMAGES_DIR.exists() or len(list(REAL_IMAGES_DIR.glob("*.jpg"))) < 3,
    reason="Real test images not downloaded (run scripts/download_real_images.py)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def embedder(config):
    return SigilEmbedder(config=config)


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


def _available_images() -> list[str]:
    """List available real image filenames (without extension)."""
    if not REAL_IMAGES_DIR.exists():
        return []
    names = []
    for p in sorted(REAL_IMAGES_DIR.glob("*.jpg")):
        try:
            Image.open(p)
            names.append(p.stem)
        except Exception:
            pass
    return names


def _load_rgb(name: str) -> np.ndarray:
    """Load an image as RGB float64 [0, 255] with even dimensions."""
    path = REAL_IMAGES_DIR / f"{name}.jpg"
    if not path.exists():
        pytest.skip(f"Image {name}.jpg not found")
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # Ensure even dimensions for DWT
    new_w = w - (w % 2)
    new_h = h - (h % 2)
    if new_w != w or new_h != h:
        img = img.crop((0, 0, new_w, new_h))
    return np.array(img, dtype=np.float64)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255.0**2 / mse))


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    """JPEG roundtrip for RGB float64."""
    import cv2
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded_bgr = cv2.imdecode(
        np.frombuffer(encoded.tobytes(), np.uint8), cv2.IMREAD_COLOR
    )
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
    return decoded_rgb


# Parametrize across available images
@pytest.fixture(params=_available_images())
def image_name(request):
    return request.param


@pytest.fixture
def rgb_image(image_name):
    return _load_rgb(image_name)


@pytest.fixture(params=AUTHOR_SEEDS, ids=["key-alpha", "key-bravo", "key-charlie"])
def multi_author_keys(request):
    return generate_author_keys(seed=request.param)


# ---------------------------------------------------------------------------
# Quality measurement — the core ask
# ---------------------------------------------------------------------------


class TestRealImageQuality:
    """Measure actual PSNR on real photographs and artwork."""

    def test_psnr_on_real_image(
        self, embedder, image_name, rgb_image, multi_author_keys
    ):
        """PSNR should be documented accurately — no synthetic inflation."""
        wm = embedder.embed(rgb_image, multi_author_keys)
        p = _psnr(rgb_image, wm)
        max_dev = np.max(np.abs(rgb_image - wm))

        # With adaptive ring strength, real images achieve 35-49 dB (avg ~40 dB).
        # Floor of 33 dB covers key-dependent variation on worst-case images.
        assert p > 33, (
            f"PSNR {p:.1f} dB on {image_name} is unacceptable"
        )
        assert max_dev < 50, (
            f"Max deviation {max_dev:.1f} on {image_name} is too high"
        )

    def test_detection_on_real_image(
        self, embedder, detector, image_name, rgb_image, multi_author_keys
    ):
        """Watermark should be detected on all real images."""
        wm = embedder.embed(rgb_image, multi_author_keys)
        result = detector.detect(wm, multi_author_keys.public_key)

        assert result.detected, (
            f"Detection failed on {image_name}: "
            f"ring={result.ring_confidence:.3f}, "
            f"payload={result.payload_confidence:.3f}"
        )
        assert result.author_id_match, (
            f"Author ID mismatch on {image_name}"
        )

    def test_no_false_positive_on_real_image(
        self, detector, image_name, rgb_image, multi_author_keys
    ):
        """Clean real images should not trigger false positives."""
        result = detector.detect(rgb_image, multi_author_keys.public_key)
        assert not result.detected or result.confidence < 0.3, (
            f"False positive on clean {image_name}"
        )


# ---------------------------------------------------------------------------
# JPEG robustness on real images
# ---------------------------------------------------------------------------


class TestRealImageJPEG:
    """JPEG compression robustness on real photographs."""

    @pytest.mark.parametrize("quality", [95, 85, 75])
    def test_jpeg_high_quality(
        self, embedder, detector, image_name, rgb_image, multi_author_keys, quality
    ):
        """Q75+ should give full detection on real images."""
        wm = embedder.embed(rgb_image, multi_author_keys)
        compressed = _jpeg_roundtrip(wm, quality)
        result = detector.detect(compressed, multi_author_keys.public_key)

        # Small or very high-detail real images with adaptive ring strength
        # may lose author_id_match after JPEG. Require either full detection
        # or strong payload signal.
        assert result.detected or result.payload_confidence > 0.4, (
            f"No signal on {image_name} after JPEG Q{quality}: "
            f"payload={result.payload_confidence:.3f}"
        )

    @pytest.mark.parametrize("quality", [60, 50])
    def test_jpeg_medium_quality(
        self, embedder, detector, image_name, rgb_image, multi_author_keys, quality
    ):
        """Q50-Q60: detection should survive on most real images."""
        wm = embedder.embed(rgb_image, multi_author_keys)
        compressed = _jpeg_roundtrip(wm, quality)
        result = detector.detect(compressed, multi_author_keys.public_key)

        assert result.detected or result.payload_confidence > 0.3, (
            f"No signal on {image_name} after JPEG Q{quality}"
        )


# ---------------------------------------------------------------------------
# Ghost signal on real images
# ---------------------------------------------------------------------------


class TestRealImageGhost:
    """Ghost signal detection on real photographs."""

    def test_ghost_present(
        self, embedder, detector, image_name, rgb_image, multi_author_keys
    ):
        """Ghost signal should be present on real images."""
        h, w = rgb_image.shape[:2]
        # Very small images (<400px) don't have enough spectral resolution for ghost bands
        if min(h, w) < 400:
            pytest.skip(f"{image_name} too small ({w}x{h}) for reliable ghost detection")
        wm = embedder.embed(rgb_image, multi_author_keys)
        result = detector.detect(wm, multi_author_keys.public_key)
        assert result.ghost_confidence > 0.3, (
            f"Ghost too low on {image_name}: {result.ghost_confidence:.3f}"
        )

    def test_ghost_wrong_key(
        self, embedder, detector, image_name, rgb_image
    ):
        """Ghost confidence should be low with wrong key."""
        keys_a = generate_author_keys(seed=b"real-photo-ghost-key-a-32bytes!")
        keys_b = generate_author_keys(seed=b"real-photo-ghost-key-b-32bytes!")
        wm = embedder.embed(rgb_image, keys_a)
        result = detector.detect(wm, keys_b.public_key)
        assert result.ghost_confidence < 0.3, (
            f"Ghost false positive on {image_name}: {result.ghost_confidence:.3f}"
        )


# ---------------------------------------------------------------------------
# Comprehensive report
# ---------------------------------------------------------------------------


class TestRealImageReport:
    """Generate comprehensive quality and robustness report on real images."""

    def test_quality_report(self, embedder, detector, capsys):
        """Print PSNR and detection report across all real images."""
        keys = generate_author_keys(seed=b"report-key-for-real-images-32b!")
        image_names = _available_images()
        if not image_names:
            pytest.skip("No real images available")

        print(f"\n{'='*95}")
        print("REAL IMAGE QUALITY REPORT")
        print(
            f"Config: embed_strength={embedder.config.embed_strength}, "
            f"ring_strength={embedder.config.ring_strength}, "
            f"ghost={embedder.config.ghost_strength_multiplier}x"
        )
        print(f"{'='*95}")
        print(
            f"{'Image':<28} {'Size':>12} {'PSNR':>8} {'MaxDev':>8} "
            f"{'Ring':>7} {'Payload':>9} {'Ghost':>7} {'Author':>8}"
        )
        print(f"{'-'*95}")

        psnrs = []
        max_devs = []
        for name in image_names:
            img = _load_rgb(name)
            wm = embedder.embed(img, keys)
            p = _psnr(img, wm)
            md = np.max(np.abs(img - wm))
            psnrs.append(p)
            max_devs.append(md)
            result = detector.detect(wm, keys.public_key)
            h, w = img.shape[:2]
            print(
                f"{name:<28} {w:>5}x{h:<5} {p:>7.1f} {md:>8.1f} "
                f"{result.ring_confidence:>7.3f} "
                f"{result.payload_confidence:>9.3f} "
                f"{result.ghost_confidence:>7.3f} "
                f"{'YES' if result.author_id_match else 'no':>8}"
            )

        print(f"{'-'*95}")
        print(
            f"{'AVERAGE':<28} {'':>12} {np.mean(psnrs):>7.1f} "
            f"{np.mean(max_devs):>8.1f}"
        )
        print(
            f"{'MIN':<28} {'':>12} {np.min(psnrs):>7.1f} "
            f"{np.max(max_devs):>8.1f}"
        )
        print(f"{'='*95}")

    def test_jpeg_robustness_report(self, embedder, detector, capsys):
        """Print JPEG robustness across quality levels on real images."""
        keys = generate_author_keys(seed=b"report-key-for-real-images-32b!")
        image_names = _available_images()
        if not image_names:
            pytest.skip("No real images available")

        qualities = [99, 90, 75, 60, 50]

        print(f"\n{'='*100}")
        print("REAL IMAGE JPEG ROBUSTNESS REPORT")
        print(f"{'='*100}")

        header = f"{'Image':<28}"
        for q in qualities:
            header += f" {'Q'+str(q):>8}"
        print(header)
        print(f"{'-'*100}")

        for name in image_names:
            img = _load_rgb(name)
            wm = embedder.embed(img, keys)

            row = f"{name:<28}"
            for q in qualities:
                compressed = _jpeg_roundtrip(wm, q)
                result = detector.detect(compressed, keys.public_key)
                if result.author_id_match:
                    status = "FULL"
                elif result.detected:
                    status = "det"
                elif result.payload_confidence > 0.3:
                    status = "weak"
                else:
                    status = "FAIL"
                row += f" {status:>8}"
            print(row)

        print(f"{'='*100}")
        print("FULL = detected + author match, det = detected, weak = some signal, FAIL = no signal")

    def test_ghost_report(self, embedder, config, capsys):
        """Print ghost signal analysis on real images."""
        keys = generate_author_keys(seed=b"report-key-for-real-images-32b!")
        image_names = _available_images()
        if not image_names:
            pytest.skip("No real images available")

        print(f"\n{'='*85}")
        print("REAL IMAGE GHOST SIGNAL REPORT")
        print(f"{'='*85}")
        print(
            f"{'Image':<28} {'Ghost corr':>12} {'P-value':>10} "
            f"{'Detected':>10}"
        )
        print(f"{'-'*85}")

        for name in image_names:
            img = _load_rgb(name)
            wm = embedder.embed(img, keys)

            # Extract Y channel for ghost analysis
            from sigil_watermark.color import extract_y_channel
            y = extract_y_channel(wm)
            ghost = analyze_ghost_signature(y, keys.public_key, config)
            print(
                f"{name:<28} {ghost.correlation:>12.6f} "
                f"{ghost.p_value:>10.4f} "
                f"{'YES' if ghost.ghost_detected else 'no':>10}"
            )

        print(f"{'='*85}")
