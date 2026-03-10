"""False positive rate validation tests.

Verifies that unwatermarked images produce no false detections
across many image types, random keys, and attack scenarios.
"""

import numpy as np
import pytest

from sigil_watermark.config import SigilConfig
from sigil_watermark.detect import SigilDetector
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.keygen import derive_author_index, generate_author_keys


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


@pytest.fixture
def embedder(config):
    return SigilEmbedder(config=config)


# --- False Positive on Clean Images ---


class TestFalsePositiveCleanImages:
    """No false detections on unwatermarked images."""

    @pytest.mark.parametrize("seed", range(20))
    def test_random_key_on_natural_image(self, detector, seed):
        """Random key should not detect on a natural-ish image."""
        img_rng = np.random.default_rng(seed + 1000)
        key_seed = f"fp-test-key-{seed:04d}-32-bytes-pad!!".encode()
        keys = generate_author_keys(seed=key_seed)

        h, w = 512, 512
        img = np.zeros((h, w), dtype=np.float64)
        freq = 15 + seed * 2
        for i in range(h):
            for j in range(w):
                img[i, j] = (
                    128
                    + 40 * np.sin(i / freq)
                    + 30 * np.cos(j / (freq + 5))
                    + 15 * np.sin((i + j) / 30)
                )
        img += img_rng.normal(0, 10, (h, w))
        img = np.clip(img, 0, 255)

        result = detector.detect(img, keys.public_key)
        assert not result.detected, (
            f"False positive on clean image seed={seed}: conf={result.payload_confidence:.2f}"
        )

    @pytest.mark.parametrize(
        "image_type",
        ["flat", "gradient", "noise", "stripes", "checkerboard", "circles"],
    )
    def test_no_detection_on_patterns(self, detector, image_type):
        """Common patterns should not trigger false positives."""
        keys = generate_author_keys(seed=b"fp-pattern-test-key-32-bytes!!!")
        h, w = 512, 512

        if image_type == "flat":
            img = np.full((h, w), 128.0)
        elif image_type == "gradient":
            y = np.linspace(0, 255, h).reshape(-1, 1)
            x = np.linspace(0, 255, w).reshape(1, -1)
            img = y * 0.5 + x * 0.5
        elif image_type == "noise":
            img = np.random.default_rng(42).normal(128, 40, (h, w))
            img = np.clip(img, 0, 255)
        elif image_type == "stripes":
            img = np.zeros((h, w), dtype=np.float64)
            for i in range(h):
                img[i, :] = 128 + 80 * np.sin(2 * np.pi * i / 32)
        elif image_type == "checkerboard":
            img = np.zeros((h, w), dtype=np.float64)
            for i in range(h):
                for j in range(w):
                    img[i, j] = 240 if (i // 64 + j // 64) % 2 == 0 else 15
        else:  # circles
            y, x = np.ogrid[:h, :w]
            r = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
            img = np.clip(128 + 60 * np.sin(r / 10), 0, 255).astype(np.float64)

        result = detector.detect(img, keys.public_key)
        assert not result.detected, (
            f"False positive on {image_type}: conf={result.payload_confidence:.2f}"
        )


# --- False Positive with Wrong Key ---


class TestWrongKeyFalsePositive:
    """Watermarked image should not match the wrong key."""

    def test_50_random_wrong_keys(self, embedder, detector):
        """Test 50 wrong keys on a watermarked image."""
        rng = np.random.default_rng(42)
        h, w = 512, 512
        img = np.zeros((h, w), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                img[i, j] = 128 + 40 * np.sin(i / 20) + 30 * np.cos(j / 15)
        img += rng.normal(0, 8, (h, w))
        img = np.clip(img, 0, 255)

        author = generate_author_keys(seed=b"fp-wrong-key-author-32-bytes!!!")
        watermarked = embedder.embed(img, author)

        false_positives = 0
        for i in range(50):
            wrong_seed = f"fp-wrong-key-test-{i:04d}-32-bytes".encode()
            wrong = generate_author_keys(seed=wrong_seed)
            result = detector.detect(watermarked, wrong.public_key)
            if result.author_id_match:
                false_positives += 1

        assert false_positives == 0, f"{false_positives}/50 wrong keys matched (should be 0)"


# --- Beacon False Positive Rate ---


class TestBeaconFalsePositive:
    """Beacon detection should not trigger on clean images."""

    @pytest.mark.parametrize("seed", range(10))
    def test_no_beacon_on_clean(self, detector, seed):
        rng = np.random.default_rng(seed + 500)
        h, w = 512, 512
        img = np.zeros((h, w), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                img[i, j] = 128 + 40 * np.sin(i / (15 + seed)) + 30 * np.cos(j / 20)
        img += rng.normal(0, 10, (h, w))
        img = np.clip(img, 0, 255)
        assert detector.detect_beacon(img) is False

    def test_no_beacon_on_noise(self, detector):
        rng = np.random.default_rng(42)
        img = np.clip(rng.normal(128, 40, (512, 512)), 0, 255)
        assert detector.detect_beacon(img) is False

    def test_no_beacon_on_flat(self, detector):
        img = np.full((512, 512), 128.0)
        assert detector.detect_beacon(img) is False


# --- Author Index Collision Rate ---


class TestAuthorIndexCollisions:
    """Verify author indices are sufficiently unique."""

    def test_100_keys_no_collisions(self):
        """100 unique keys should produce 100 unique 20-bit indices."""
        indices = set()
        for i in range(100):
            seed = f"collision-test-key-{i:04d}-32bytes!".encode()
            keys = generate_author_keys(seed=seed)
            index = tuple(derive_author_index(keys.public_key))
            indices.add(index)

        assert len(indices) == 100, f"Only {len(indices)} unique indices from 100 keys"

    def test_1000_keys_collision_rate(self):
        """With 2^20 = 1M possible indices, 1000 keys should have ~0 collisions."""
        indices = []
        for i in range(1000):
            seed = f"collision-1k-test-{i:06d}-32bytes!".encode()
            keys = generate_author_keys(seed=seed)
            index = tuple(derive_author_index(keys.public_key))
            indices.append(index)

        unique = len(set(indices))
        collision_rate = 1.0 - unique / len(indices)
        # Birthday paradox: P(collision) ≈ n²/(2*M) = 1000²/(2*2^20) ≈ 0.048%
        assert collision_rate < 0.01, (
            f"Collision rate {collision_rate:.4f} too high ({len(indices) - unique} collisions)"
        )


# --- Confidence Distribution Tests ---


class TestConfidenceDistributions:
    """Verify confidence values are well-separated between positive and negative cases."""

    def test_positive_confidence_above_threshold(self, embedder, detector):
        """Watermarked images should consistently have high confidence."""
        author = generate_author_keys(seed=b"conf-dist-test-author-32-bytes!")
        np.random.default_rng(42)
        confidences = []
        for seed in range(10):
            img_rng = np.random.default_rng(seed + 100)
            h, w = 512, 512
            img = np.zeros((h, w), dtype=np.float64)
            freq = 15 + seed * 2
            for i in range(h):
                for j in range(w):
                    img[i, j] = 128 + 40 * np.sin(i / freq) + 30 * np.cos(j / (freq + 3))
            img += img_rng.normal(0, 8, (h, w))
            img = np.clip(img, 0, 255)
            watermarked = embedder.embed(img, author)
            result = detector.detect(watermarked, author.public_key)
            confidences.append(result.payload_confidence)

        min_conf = min(confidences)
        assert min_conf > 0.5, (
            f"Minimum positive confidence {min_conf:.2f} too low. "
            f"Distribution: {[f'{c:.2f}' for c in confidences]}"
        )

    def test_negative_confidence_below_threshold(self, detector):
        """Clean images should consistently have low confidence."""
        keys = generate_author_keys(seed=b"conf-neg-test-author-32-bytes!!")
        np.random.default_rng(42)
        confidences = []
        for seed in range(10):
            img_rng = np.random.default_rng(seed + 200)
            h, w = 512, 512
            img = np.zeros((h, w), dtype=np.float64)
            freq = 15 + seed * 2
            for i in range(h):
                for j in range(w):
                    img[i, j] = 128 + 40 * np.sin(i / freq) + 30 * np.cos(j / (freq + 3))
            img += img_rng.normal(0, 8, (h, w))
            img = np.clip(img, 0, 255)
            result = detector.detect(img, keys.public_key)
            confidences.append(result.payload_confidence)

        max_conf = max(confidences)
        assert max_conf <= 0.5, (
            f"Maximum negative confidence {max_conf:.2f} too high. "
            f"Distribution: {[f'{c:.2f}' for c in confidences]}"
        )

    def test_separation_between_positive_and_negative(self, embedder, detector):
        """Gap between worst positive and best negative should be significant."""
        author = generate_author_keys(seed=b"conf-sep-test-author-32-bytes!!")
        np.random.default_rng(42)

        pos_confidences = []
        neg_confidences = []

        for seed in range(8):
            img_rng = np.random.default_rng(seed + 300)
            h, w = 512, 512
            img = np.zeros((h, w), dtype=np.float64)
            freq = 15 + seed * 3
            for i in range(h):
                for j in range(w):
                    img[i, j] = 128 + 40 * np.sin(i / freq) + 30 * np.cos(j / (freq + 5))
            img += img_rng.normal(0, 8, (h, w))
            img = np.clip(img, 0, 255)

            # Positive
            watermarked = embedder.embed(img, author)
            pos_result = detector.detect(watermarked, author.public_key)
            pos_confidences.append(pos_result.payload_confidence)

            # Negative
            neg_result = detector.detect(img, author.public_key)
            neg_confidences.append(neg_result.payload_confidence)

        min_pos = min(pos_confidences)
        max_neg = max(neg_confidences)
        gap = min_pos - max_neg
        assert gap > 0.1, (
            f"Confidence gap {gap:.2f} too small (min_pos={min_pos:.2f}, max_neg={max_neg:.2f})"
        )
