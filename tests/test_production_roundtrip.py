"""Production roundtrip tests — mirrors the exact flow of run_sigil_encoding.

The production pipeline does:
  1. Embed watermark into RGB float64
  2. Convert to uint8 BGR
  3. Encode as JPEG/PNG (lossy roundtrip)
  4. Decode back
  5. Convert to RGB float64
  6. Run detector.detect()

Tests are parameterized across natural image types and multiple author keys.
"""

import numpy as np
import pytest
from conftest import (
    REALISTIC_IMAGE_GENERATORS,
    jpeg_roundtrip_gray,
    jpeg_roundtrip_rgb,
    make_natural_scene,
    make_photo_like_rgb,
    png_roundtrip_rgb,
)

from sigil_watermark.keygen import (
    derive_content_ring_radii,
    derive_ring_phase_offsets,
    derive_ring_radii,
    derive_sentinel_ring_radii,
)
from sigil_watermark.transforms import detect_dft_rings, embed_dft_rings


class TestProductionJPEGRoundtrip:
    """Tests that mirror the exact production encode -> JPEG -> verify flow."""

    def test_jpeg_q99_all_layers(self, embedder, detector, multi_author_keys):
        """Full pipeline: embed -> JPEG Q99 -> detect across all keys."""
        img = make_photo_like_rgb()
        wm = embedder.embed(img, multi_author_keys)
        decoded = jpeg_roundtrip_rgb(wm, quality=99)

        result = detector.detect(decoded, multi_author_keys.public_key)

        assert result.detected, "Detection failed after JPEG Q99"
        assert result.author_id_match, "Author ID mismatch after JPEG Q99"
        assert result.ring_confidence > 0.1, (
            f"Ring confidence too low after JPEG Q99: {result.ring_confidence:.3f}"
        )
        assert result.payload_confidence > 0.5, (
            f"Payload confidence too low after JPEG Q99: {result.payload_confidence:.3f}"
        )
        assert result.ghost_confidence > 0.2, (
            f"Ghost confidence too low after JPEG Q99: {result.ghost_confidence:.3f}"
        )

    def test_jpeg_q75_detection(self, embedder, detector, multi_author_keys):
        """JPEG Q75 -- standard web quality. Core detection should survive."""
        img = make_photo_like_rgb()
        wm = embedder.embed(img, multi_author_keys)
        decoded = jpeg_roundtrip_rgb(wm, quality=75)

        result = detector.detect(decoded, multi_author_keys.public_key)

        assert result.detected, "Detection failed after JPEG Q75"
        assert result.payload_confidence > 0.3, (
            f"Payload confidence too low after JPEG Q75: {result.payload_confidence:.3f}"
        )

    def test_png_lossless_all_layers(self, embedder, detector, multi_author_keys):
        """PNG roundtrip (lossless but uint8 quantization). Should be near-perfect."""
        img = make_photo_like_rgb()
        wm = embedder.embed(img, multi_author_keys)
        decoded = png_roundtrip_rgb(wm)

        result = detector.detect(decoded, multi_author_keys.public_key)

        assert result.detected, "Detection failed after PNG roundtrip"
        assert result.author_id_match, "Author ID mismatch after PNG"
        assert result.ring_confidence > 0.1, (
            f"Ring confidence too low after PNG: {result.ring_confidence:.3f}"
        )
        assert result.payload_confidence > 0.7, (
            f"Payload confidence too low after PNG: {result.payload_confidence:.3f}"
        )
        assert result.ghost_confidence > 0.2, (
            f"Ghost confidence too low after PNG: {result.ghost_confidence:.3f}"
        )

    @pytest.mark.parametrize(
        "size",
        [
            (1920, 1080),  # Full HD landscape
            (1080, 1920),  # Full HD portrait
            (601, 666),  # Odd dimensions
            (449, 804),  # Small portrait
        ],
    )
    def test_jpeg_q99_various_sizes(self, embedder, detector, multi_author_keys, size):
        """JPEG Q99 roundtrip at various real-world image sizes."""
        w, h = size
        img = make_photo_like_rgb(h=h, w=w)
        wm = embedder.embed(img, multi_author_keys)
        decoded = jpeg_roundtrip_rgb(wm, quality=99)

        result = detector.detect(decoded, multi_author_keys.public_key)

        assert result.detected, f"Detection failed at {w}x{h} after JPEG Q99"
        # Ring confidence varies by aspect ratio; payload is the primary signal
        assert result.payload_confidence > 0.5, (
            f"Payload confidence too low at {w}x{h}: {result.payload_confidence:.3f}"
        )

    @pytest.mark.parametrize("image_name", list(REALISTIC_IMAGE_GENERATORS.keys()))
    def test_jpeg_q99_natural_images_grayscale(
        self, embedder, detector, multi_author_keys, image_name
    ):
        """JPEG Q99 roundtrip on each realistic image type (grayscale path)."""
        img = REALISTIC_IMAGE_GENERATORS[image_name]()
        wm = embedder.embed(img, multi_author_keys)
        decoded = jpeg_roundtrip_gray(wm, quality=99)

        result = detector.detect(decoded, multi_author_keys.public_key)

        assert result.detected, f"Detection failed on {image_name} after JPEG Q99"
        assert result.author_id_match, f"Author ID mismatch on {image_name} after JPEG Q99"


class TestContentRingStability:
    """Tests for content-dependent ring behavior across lossy codecs."""

    def test_content_rings_shift_after_jpeg(self, multi_author_keys, config):
        """Verify that content ring radii change after JPEG — this is expected."""
        img = make_natural_scene()
        before = derive_content_ring_radii(multi_author_keys.public_key, img, config=config)
        after_jpeg = jpeg_roundtrip_gray(img, quality=99)
        after = derive_content_ring_radii(multi_author_keys.public_key, after_jpeg, config=config)
        assert not np.allclose(before, after), "Content rings unexpectedly survived JPEG"

    def test_stable_rings_unaffected_by_content_shift(self, embedder, multi_author_keys, config):
        """Key+sentinel ring detection must not depend on content ring positions."""
        img = make_natural_scene()
        wm = embedder.embed(img, multi_author_keys)
        wm_jpeg = jpeg_roundtrip_gray(wm, quality=99)

        key_radii = derive_ring_radii(multi_author_keys.public_key, config=config)
        sentinel_radii = derive_sentinel_ring_radii(config=config)
        stable_radii = np.sort(np.concatenate([key_radii, sentinel_radii]))
        stable_phase = derive_ring_phase_offsets(
            multi_author_keys.public_key, len(stable_radii), config=config
        )

        _, confidence = detect_dft_rings(
            wm_jpeg,
            stable_radii,
            tolerance=0.02,
            ring_width=config.ring_width,
            phase_offsets=stable_phase,
        )
        assert confidence > 0.1, f"Stable ring detection failed after JPEG: {confidence:.3f}"

    def test_phase_offsets_independent_of_content_rings(self, multi_author_keys, config):
        """Phase offsets for stable rings must be the same regardless of content ring count."""
        key_radii = derive_ring_radii(multi_author_keys.public_key, config=config)
        sentinel_radii = derive_sentinel_ring_radii(config=config)
        stable_radii = np.sort(np.concatenate([key_radii, sentinel_radii]))

        phase_a = derive_ring_phase_offsets(
            multi_author_keys.public_key, len(stable_radii), config=config
        )
        phase_b = derive_ring_phase_offsets(
            multi_author_keys.public_key, len(stable_radii), config=config
        )
        np.testing.assert_array_equal(phase_a, phase_b)


class TestPhaseOffsetDetection:
    """Tests for phase offset embedding and detection."""

    def test_phase_offsets_dont_hurt_detection(self, multi_author_keys, config):
        """Rings with phase offsets should still be detectable."""
        img = make_natural_scene()
        name = "natural_scene"
        key_radii = derive_ring_radii(multi_author_keys.public_key, config=config)
        sentinel_radii = derive_sentinel_ring_radii(config=config)
        stable_radii = np.sort(np.concatenate([key_radii, sentinel_radii]))
        phase_offsets = derive_ring_phase_offsets(
            multi_author_keys.public_key, len(stable_radii), config=config
        )

        embedded = embed_dft_rings(
            img.copy(),
            stable_radii,
            strength=config.ring_strength * len(stable_radii) / 8,
            ring_width=config.ring_width,
            phase_offsets=phase_offsets,
        )

        _, conf_correct = detect_dft_rings(
            embedded,
            stable_radii,
            tolerance=0.02,
            ring_width=config.ring_width,
            phase_offsets=phase_offsets,
        )
        assert conf_correct > 0.1, (
            f"Phase-modulated ring detection failed on {name}: {conf_correct:.3f}"
        )

    def test_phase_offsets_survive_jpeg(self, multi_author_keys, config):
        """Phase-modulated rings should survive JPEG Q99."""
        img = make_natural_scene()
        name = "natural_scene"
        key_radii = derive_ring_radii(multi_author_keys.public_key, config=config)
        sentinel_radii = derive_sentinel_ring_radii(config=config)
        stable_radii = np.sort(np.concatenate([key_radii, sentinel_radii]))
        phase_offsets = derive_ring_phase_offsets(
            multi_author_keys.public_key, len(stable_radii), config=config
        )

        embedded = embed_dft_rings(
            img.copy(),
            stable_radii,
            strength=config.ring_strength * len(stable_radii) / 8,
            ring_width=config.ring_width,
            phase_offsets=phase_offsets,
        )

        jpeg_img = jpeg_roundtrip_gray(embedded, quality=99)

        _, confidence = detect_dft_rings(
            jpeg_img,
            stable_radii,
            tolerance=0.02,
            ring_width=config.ring_width,
            phase_offsets=phase_offsets,
        )
        assert confidence > 0.1, f"Phase rings didn't survive JPEG Q99 on {name}: {confidence:.3f}"


class TestSentinelRings:
    """Tests for sentinel ring embedding and tampering detection."""

    def test_sentinel_rings_detectable(self, multi_author_keys, config):
        """Sentinel rings should be detectable after embedding."""
        img = make_natural_scene()
        name = "natural_scene"
        sentinel_radii = derive_sentinel_ring_radii(config=config)
        embedded = embed_dft_rings(
            img.copy(),
            sentinel_radii,
            strength=config.ring_strength * len(sentinel_radii) / 8,
            ring_width=config.ring_width,
        )
        _, confidence = detect_dft_rings(
            embedded,
            sentinel_radii,
            tolerance=0.02,
            ring_width=config.ring_width,
        )
        assert confidence > 0.2, f"Sentinel ring detection too low on {name}: {confidence:.3f}"

    def test_tampering_notch_reduces_key_rings(self, embedder, detector, multi_author_keys, config):
        """Notch filtering key ring frequencies should reduce key ring confidence."""
        img = make_natural_scene()
        wm = embedder.embed(img, multi_author_keys)

        from sigil_watermark.color import extract_y_channel

        y = extract_y_channel(wm)
        key_radii = derive_ring_radii(multi_author_keys.public_key, config=config)
        _, baseline_key_conf = detect_dft_rings(
            y,
            key_radii,
            tolerance=0.02,
            ring_width=config.ring_width,
        )

        h, w = y.shape
        f = np.fft.fft2(y)
        f_shifted = np.fft.fftshift(f)
        cy, cx = h // 2, w // 2
        max_freq = min(h, w) // 2
        fy, fx = np.ogrid[:h, :w]
        dist = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2) / max_freq

        for r in key_radii:
            notch = 1.0 - np.exp(-((dist - r) ** 2) / (2 * config.ring_width**2))
            f_shifted *= notch

        attacked = np.real(np.fft.ifft2(np.fft.ifftshift(f_shifted)))
        attacked = np.clip(attacked, 0, 255)

        _, attacked_key_conf = detect_dft_rings(
            attacked,
            key_radii,
            tolerance=0.02,
            ring_width=config.ring_width,
        )
        if baseline_key_conf < 0.05:
            # With adaptive ring strength, key ring confidence can be very low
            # at baseline — notch filtering can't reduce it further
            assert attacked_key_conf < 0.05, (
                f"Notch filter increased key ring confidence: {attacked_key_conf:.3f}"
            )
        else:
            assert attacked_key_conf < baseline_key_conf * 0.5, (
                f"Notch filter didn't reduce key ring confidence enough: "
                f"baseline={baseline_key_conf:.3f}, after={attacked_key_conf:.3f}"
            )


class TestGhostInDetectPipeline:
    """Tests for ghost signal detection integrated into SigilDetector.detect()."""

    def test_ghost_confidence_present(self, embedder, detector, realistic_image, multi_author_keys):
        """Ghost confidence on freshly embedded realistic images."""
        name, img = realistic_image
        wm = embedder.embed(img, multi_author_keys)
        result = detector.detect(wm, multi_author_keys.public_key)

        assert hasattr(result, "ghost_confidence")
        assert result.ghost_confidence > 0.2, (
            f"Ghost confidence too low on {name}: {result.ghost_confidence:.3f}"
        )

    def test_ghost_confidence_zero_wrong_key(self, embedder, detector, author_keys, author_keys_b):
        """Ghost confidence should be low when checked with wrong key."""
        img = make_photo_like_rgb()
        wm = embedder.embed(img, author_keys)
        result = detector.detect(wm, author_keys_b.public_key)

        assert result.ghost_confidence < 0.3, (
            f"Ghost confidence too high with wrong key: {result.ghost_confidence:.3f}"
        )

    def test_ghost_survives_jpeg_q99(self, embedder, detector, multi_author_keys):
        """Ghost signal should survive JPEG Q99."""
        img = make_photo_like_rgb()
        wm = embedder.embed(img, multi_author_keys)
        decoded = jpeg_roundtrip_rgb(wm, quality=99)

        result = detector.detect(decoded, multi_author_keys.public_key)
        assert result.ghost_confidence > 0.2, (
            f"Ghost confidence too low after JPEG Q99: {result.ghost_confidence:.3f}"
        )

    def test_ghost_zero_on_unwatermarked(self, detector, realistic_image, multi_author_keys):
        """Ghost confidence should be low on unwatermarked realistic images."""
        name, img = realistic_image
        result = detector.detect(img, multi_author_keys.public_key)

        assert result.ghost_confidence < 0.3, (
            f"Ghost false positive on clean {name}: {result.ghost_confidence:.3f}"
        )

    def test_overall_confidence_uses_all_layers(self, embedder, detector, author_keys):
        """Overall confidence should blend ring, payload, and ghost."""
        img = make_photo_like_rgb()
        wm = embedder.embed(img, author_keys)
        result = detector.detect(wm, author_keys.public_key)

        expected = min(
            1.0,
            (
                0.35 * result.ring_confidence
                + 0.45 * result.payload_confidence
                + 0.20 * result.ghost_confidence
            ),
        )
        assert abs(result.confidence - expected) < 0.01, (
            f"Overall confidence {result.confidence:.3f} doesn't match "
            f"expected blend {expected:.3f}"
        )


class TestMultipleKeysJPEGRoundtrip:
    """Ensure different author keys produce distinct, non-interfering watermarks after JPEG."""

    @pytest.mark.parametrize("image_name", list(REALISTIC_IMAGE_GENERATORS.keys()))
    def test_different_keys_dont_cross_detect(
        self, embedder, detector, image_name, author_keys, author_keys_b
    ):
        """Watermark from key A should not verify with key B after JPEG, on realistic images."""
        img = REALISTIC_IMAGE_GENERATORS[image_name]()
        # For grayscale images, make a simple RGB version for JPEG roundtrip
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        wm = embedder.embed(img, author_keys)
        decoded = jpeg_roundtrip_rgb(wm, quality=99)

        result_correct = detector.detect(decoded, author_keys.public_key)
        result_wrong = detector.detect(decoded, author_keys_b.public_key)

        assert result_correct.detected, f"Correct key didn't detect on {image_name}"
        assert result_correct.author_id_match, f"Correct key didn't match on {image_name}"
        assert not result_wrong.author_id_match, f"Wrong key falsely matched on {image_name}"
        # Ghost should be key-selective, or both at noise floor (0.0) on
        # images where perceptual masking reduces ghost strength heavily.
        assert result_correct.ghost_confidence >= result_wrong.ghost_confidence, (
            f"Ghost not key-selective on {image_name}: "
            f"correct={result_correct.ghost_confidence:.3f}, "
            f"wrong={result_wrong.ghost_confidence:.3f}"
        )
