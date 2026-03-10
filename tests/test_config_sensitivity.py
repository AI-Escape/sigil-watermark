"""Configuration sensitivity tests.

Tests how the watermark system behaves with different configuration
parameters, measuring the quality vs. robustness trade-off.
"""

import io

import numpy as np
import pytest
from PIL import Image

from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.detect import SigilDetector
from sigil_watermark.keygen import generate_author_keys
from sigil_watermark.config import SigilConfig


@pytest.fixture
def author_keys():
    return generate_author_keys(seed=b"config-sensitivity-test32-bytes!")


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


def _jpeg(image, quality):
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.float64)


# --- Embed Strength Sensitivity ---


class TestEmbedStrength:
    """Different embed_strength values trade quality for robustness."""

    @pytest.mark.parametrize("strength", [2.0, 5.0, 10.0, 15.0])
    def test_strength_psnr(self, author_keys, strength):
        """Higher strength -> lower PSNR."""
        config = SigilConfig(embed_strength=strength)
        embedder = SigilEmbedder(config=config)
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        mse = np.mean((watermarked - img) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255.0**2 / mse)
            # All should be above minimum quality
            assert psnr > 25, f"Strength {strength}: PSNR {psnr:.1f} dB too low"

    @pytest.mark.parametrize("strength", [2.0, 5.0, 10.0])
    def test_strength_detection(self, author_keys, strength):
        """Detection should work at all reasonable strengths."""
        config = SigilConfig(embed_strength=strength)
        embedder = SigilEmbedder(config=config)
        detector = SigilDetector(config=config)
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Strength {strength}: conf={result.payload_confidence:.2f}"
        )

    def test_higher_strength_better_jpeg_robustness(self, author_keys):
        """Higher embed strength should survive JPEG better."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)

        config_low = SigilConfig(embed_strength=3.0)
        config_high = SigilConfig(embed_strength=8.0)

        embedder_low = SigilEmbedder(config=config_low)
        embedder_high = SigilEmbedder(config=config_high)
        detector_low = SigilDetector(config=config_low)
        detector_high = SigilDetector(config=config_high)

        wm_low = embedder_low.embed(img, author_keys)
        wm_high = embedder_high.embed(img, author_keys)

        comp_low = _jpeg(wm_low, 50)
        comp_high = _jpeg(wm_high, 50)

        result_low = detector_low.detect(comp_low, author_keys.public_key)
        result_high = detector_high.detect(comp_high, author_keys.public_key)

        assert result_high.payload_confidence >= result_low.payload_confidence, (
            f"High strength ({result_high.payload_confidence:.2f}) not better "
            f"than low ({result_low.payload_confidence:.2f}) after JPEG Q50"
        )

    def test_psnr_decreases_with_strength(self, author_keys):
        """PSNR should monotonically decrease as strength increases."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        prev_psnr = float("inf")
        for strength in [2.0, 5.0, 10.0, 15.0]:
            config = SigilConfig(embed_strength=strength)
            embedder = SigilEmbedder(config=config)
            watermarked = embedder.embed(img, author_keys)
            mse = np.mean((watermarked - img) ** 2)
            psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")
            assert psnr < prev_psnr, (
                f"PSNR not decreasing: strength {strength} -> {psnr:.1f} dB"
            )
            prev_psnr = psnr


# --- Ring Strength Sensitivity ---


class TestRingStrength:
    """Ring strength affects DFT layer robustness."""

    @pytest.mark.parametrize("ring_strength", [20.0, 50.0, 100.0])
    def test_ring_detection_at_different_strengths(
        self, author_keys, ring_strength
    ):
        config = SigilConfig(ring_strength=ring_strength)
        embedder = SigilEmbedder(config=config)
        detector = SigilDetector(config=config)
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        result = detector.detect(watermarked, author_keys.public_key)
        # With adaptive ring strength, ring_confidence scales down on images
        # with strong spectral energy. At ring_strength=20, expect > 0.05.
        min_ring_conf = 0.05 if ring_strength <= 20 else 0.3
        assert result.ring_confidence > min_ring_conf, (
            f"Ring strength {ring_strength}: ring_conf={result.ring_confidence:.2f}"
        )


# --- Spreading Factor ---


class TestSpreadingFactor:
    """Spreading factor trades capacity for robustness."""

    @pytest.mark.parametrize("sf", [64, 128, 256, 512])
    def test_spreading_factor_detection(self, author_keys, sf):
        config = SigilConfig(spreading_factor=sf)
        embedder = SigilEmbedder(config=config)
        detector = SigilDetector(config=config)
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"SF {sf}: conf={result.payload_confidence:.2f}"
        )

    def test_higher_sf_better_noise_robustness(self, author_keys):
        """Higher spreading factor should be more noise robust."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)

        config_low = SigilConfig(spreading_factor=64)
        config_high = SigilConfig(spreading_factor=512)

        embedder_low = SigilEmbedder(config=config_low)
        embedder_high = SigilEmbedder(config=config_high)
        detector_low = SigilDetector(config=config_low)
        detector_high = SigilDetector(config=config_high)

        wm_low = embedder_low.embed(img, author_keys)
        wm_high = embedder_high.embed(img, author_keys)

        noisy_low = np.clip(wm_low + rng.normal(0, 15, wm_low.shape), 0, 255)
        noisy_high = np.clip(wm_high + rng.normal(0, 15, wm_high.shape), 0, 255)

        result_low = detector_low.detect(noisy_low, author_keys.public_key)
        result_high = detector_high.detect(noisy_high, author_keys.public_key)

        assert result_high.payload_confidence >= result_low.payload_confidence, (
            f"High SF ({result_high.payload_confidence:.2f}) not better "
            f"than low ({result_low.payload_confidence:.2f}) after noise"
        )


# --- DWT Levels ---


class TestDWTLevels:
    """Different numbers of DWT decomposition levels."""

    @pytest.mark.parametrize("levels", [2, 3, 4])
    def test_dwt_levels_detection(self, author_keys, levels):
        config = SigilConfig(dwt_levels=levels)
        embedder = SigilEmbedder(config=config)
        detector = SigilDetector(config=config)
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"DWT levels {levels}: conf={result.payload_confidence:.2f}"
        )


# --- Wavelet Family ---


class TestWaveletFamily:
    """Different wavelet families."""

    @pytest.mark.parametrize("wavelet", ["db2", "db4", "db6", "haar"])
    def test_wavelet_roundtrip(self, author_keys, wavelet):
        config = SigilConfig(wavelet=wavelet)
        embedder = SigilEmbedder(config=config)
        detector = SigilDetector(config=config)
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Wavelet {wavelet}: conf={result.payload_confidence:.2f}"
        )

    @pytest.mark.parametrize("wavelet", ["db2", "db4", "db6"])
    def test_wavelet_jpeg_robustness(self, author_keys, wavelet):
        config = SigilConfig(wavelet=wavelet)
        embedder = SigilEmbedder(config=config)
        detector = SigilDetector(config=config)
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        compressed = _jpeg(watermarked, 75)
        result = detector.detect(compressed, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, (
            f"Wavelet {wavelet} JPEG Q75: conf={result.payload_confidence:.2f}"
        )


# --- Embed Subbands ---


class TestEmbedSubbands:
    """Different subband combinations."""

    @pytest.mark.parametrize(
        "subbands",
        [
            ("LH",),
            ("HL",),
            ("LH", "HL"),
            ("LH", "HL", "HH"),
        ],
    )
    def test_subband_combinations(self, author_keys, subbands):
        config = SigilConfig(embed_subbands=subbands)
        embedder = SigilEmbedder(config=config)
        detector = SigilDetector(config=config)
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, (
            f"Subbands {subbands}: conf={result.payload_confidence:.2f}"
        )


# --- Ghost Strength Multiplier ---


class TestGhostStrengthMultiplier:
    """Ghost signal strength parameter."""

    @pytest.mark.parametrize("mult", [0.5, 1.0, 1.5, 3.0])
    def test_ghost_multiplier_doesnt_break_detection(self, author_keys, mult):
        config = SigilConfig(ghost_strength_multiplier=mult)
        embedder = SigilEmbedder(config=config)
        detector = SigilDetector(config=config)
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        watermarked = embedder.embed(img, author_keys)
        result = detector.detect(watermarked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Ghost mult {mult}: conf={result.payload_confidence:.2f}"
        )


# --- Mask Floor ---


class TestMaskFloor:
    """Perceptual mask floor affects embedding in flat regions."""

    @pytest.mark.parametrize("floor", [0.1, 0.3, 0.5, 1.0])
    def test_mask_floor_on_flat_image(self, author_keys, floor):
        config = SigilConfig(mask_floor=floor)
        embedder = SigilEmbedder(config=config)
        detector = SigilDetector(config=config)
        img = np.full((512, 512), 128.0, dtype=np.float64)
        watermarked = embedder.embed(img, author_keys)
        result = detector.detect(watermarked, author_keys.public_key)
        # Flat image with low mask floor may not detect, that's expected
        if floor >= 0.3:
            assert result.detected or result.payload_confidence > 0.3, (
                f"Floor {floor}: conf={result.payload_confidence:.2f}"
            )


# --- Config Consistency Report ---


class TestConfigReport:
    """Generate a report comparing different configs (informational)."""

    def test_print_config_comparison(self, author_keys, capsys):
        rng = np.random.default_rng(42)
        img = _make_image(rng)

        configs = {
            "default": SigilConfig(),
            "strong": SigilConfig(embed_strength=10.0, ring_strength=80.0),
            "weak": SigilConfig(embed_strength=2.0, ring_strength=25.0),
            "high_sf": SigilConfig(spreading_factor=512),
            "3_subbands": SigilConfig(embed_subbands=("LH", "HL", "HH")),
        }

        print(f"\n{'=' * 80}")
        print("CONFIG SENSITIVITY COMPARISON")
        print(f"{'=' * 80}")
        print(
            f"{'Config':<15} {'PSNR':>6} {'Clean':>7} {'JPGQ75':>7} "
            f"{'JPGQ50':>7} {'Ring':>6}"
        )
        print(f"{'-' * 80}")

        for name, cfg in configs.items():
            emb = SigilEmbedder(config=cfg)
            det = SigilDetector(config=cfg)
            watermarked = emb.embed(img, author_keys)

            mse = np.mean((watermarked - img) ** 2)
            psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")

            r_clean = det.detect(watermarked, author_keys.public_key)
            r_j75 = det.detect(_jpeg(watermarked, 75), author_keys.public_key)
            r_j50 = det.detect(_jpeg(watermarked, 50), author_keys.public_key)

            print(
                f"{name:<15} {psnr:>5.1f}  "
                f"{r_clean.payload_confidence:>6.2f}  "
                f"{r_j75.payload_confidence:>6.2f}  "
                f"{r_j50.payload_confidence:>6.2f}  "
                f"{r_clean.ring_confidence:>5.2f}"
            )

        print(f"{'=' * 80}")
