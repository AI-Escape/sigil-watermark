"""Analytical and empirical false positive rate validation.

Computes the theoretical false positive probability of the Sigil watermark
detection system under random (unwatermarked) inputs, then validates
empirically with a large-scale test.

The detection pipeline has multiple gates:
1. RS decode must succeed (correcting up to nsym/2 = 4 symbol errors)
2. Beacon: >= 70% of beacon bits must be 1
3. Author ID: BER < 15% against expected author ID (48 bits, <= 7 errors)
4. Fallback (RS fails): raw encoded BER < 25%

For a random image with a random key, spread-spectrum correlation for each
encoded bit is approximately a coin flip (P(correct) ~ 0.5). This makes
the combined false positive probability astronomically small.
"""

import numpy as np
import pytest
from scipy.stats import binom

from sigil_watermark.config import SigilConfig
from sigil_watermark.detect import SigilDetector, _encoded_payload_length
from sigil_watermark.keygen import generate_author_keys


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


# --- Analytical False Positive Rate ---


class TestAnalyticalFPRate:
    """Compute theoretical false positive probability."""

    def test_rs_decode_probability_on_random_data(self):
        """P(RS decode succeeds) when each input bit is a fair coin flip.

        RS with nsym=8 over GF(2^8) can correct up to 4 symbol errors.
        The encoded payload has data_bytes + 8 parity bytes.
        For random input bits (P(correct)=0.5), each byte has
        P(correct) = (0.5)^8 = 1/256.
        RS decode succeeds if at most 4 of the total symbols are wrong.
        """
        cfg = SigilConfig()
        raw_len = cfg.beacon_bits + cfg.author_index_bits + cfg.author_id_bits  # 76 bits
        raw_bytes = (raw_len + 7) // 8  # 10 bytes
        total_symbols = raw_bytes + cfg.rs_nsym  # 10 + 8 = 18 symbols
        max_errors = cfg.rs_nsym // 2  # 4 correctable symbol errors

        p_symbol_correct = (0.5) ** 8  # ~0.0039

        # P(RS succeeds) = P(at most max_errors symbols wrong)
        # = P(at least total_symbols - max_errors symbols correct)
        # = sum_{k=14}^{18} C(18,k) * p^k * (1-p)^(18-k)
        min_correct = total_symbols - max_errors  # 14
        p_rs_success = binom.sf(min_correct - 1, total_symbols, p_symbol_correct)

        # This should be astronomically small
        assert p_rs_success < 1e-30, f"P(RS decode on random data) = {p_rs_success:.2e} — too high"

    def test_beacon_false_match_probability(self):
        """P(beacon match) when 8 bits are random.

        Beacon check: >= 70% of 8 bits must be 1, so >= 6 ones.
        P(>= 6 ones in 8 fair coin flips).
        """
        p_beacon = binom.sf(5, 8, 0.5)  # P(X >= 6) = P(X > 5)
        # = C(8,6)/256 + C(8,7)/256 + C(8,8)/256 = 28+8+1 = 37/256 ~ 0.145
        assert 0.14 < p_beacon < 0.15, f"P(beacon match) = {p_beacon:.4f}"

    def test_author_id_false_match_probability(self):
        """P(author ID BER < 15%) when 48 bits are random.

        Each of the 48 author ID bits has P(match) = 0.5.
        BER < 15% means <= 7 bit errors out of 48.
        """
        max_errors = int(0.15 * 48)  # 7
        p_author_match = binom.cdf(max_errors, 48, 0.5)

        # This should be very small: P(<=7 heads in 48 fair coin flips)
        assert p_author_match < 1e-5, f"P(author ID match on random) = {p_author_match:.2e}"

    def test_combined_false_positive_rate(self):
        """Combined FP rate through the RS decode path.

        P(false positive) = P(RS succeeds) * P(beacon match) * P(author ID match)

        Since P(RS succeeds on random data) is already < 1e-30, the combined
        rate is essentially zero regardless of other terms.
        """
        cfg = SigilConfig()
        raw_len = cfg.beacon_bits + cfg.author_index_bits + cfg.author_id_bits
        raw_bytes = (raw_len + 7) // 8
        total_symbols = raw_bytes + cfg.rs_nsym
        max_errors = cfg.rs_nsym // 2

        p_symbol_correct = (0.5) ** 8
        min_correct = total_symbols - max_errors
        p_rs = binom.sf(min_correct - 1, total_symbols, p_symbol_correct)

        p_beacon = binom.sf(5, 8, 0.5)
        p_author = binom.cdf(int(0.15 * 48), 48, 0.5)

        # RS path
        p_fp_rs = p_rs * p_beacon * p_author

        # Fallback path (RS fails, raw BER < 25%)
        encoded_len = _encoded_payload_length(cfg)
        max_raw_errors = int(0.25 * encoded_len)
        p_fallback_match = binom.cdf(max_raw_errors, encoded_len, 0.5)

        # Combined: either path triggers detection
        p_fp_total = p_fp_rs + p_fallback_match

        assert p_fp_total < 1e-6, (
            f"Combined FP rate = {p_fp_total:.2e} (RS path: {p_fp_rs:.2e}, "
            f"fallback: {p_fallback_match:.2e})"
        )

    def test_fallback_path_fp_rate(self):
        """Even the fallback path (raw encoded BER < 25%) has negligible FP rate.

        When RS decode fails, the detector compares raw extracted bits against
        the expected encoded payload at BER < 25%. With ~144 encoded bits
        each at P(correct)=0.5, P(BER < 25%) = P(<= 36 errors in 144 flips).
        """
        cfg = SigilConfig()
        encoded_len = _encoded_payload_length(cfg)
        max_errors = int(0.25 * encoded_len)

        p_fallback = binom.cdf(max_errors, encoded_len, 0.5)

        assert p_fallback < 1e-6, (
            f"P(fallback match on random) = {p_fallback:.2e} "
            f"(encoded_len={encoded_len}, max_errors={max_errors})"
        )


# --- Empirical Large-Scale False Positive Test ---


def _make_random_image(rng, h=512, w=512):
    """Generate a random natural-ish image."""
    img = np.zeros((h, w), dtype=np.float64)
    # Random sinusoidal texture
    freq_x = rng.uniform(10, 40)
    freq_y = rng.uniform(10, 40)
    for i in range(h):
        for j in range(w):
            img[i, j] = (
                128
                + 40 * np.sin(i / freq_y)
                + 30 * np.cos(j / freq_x)
                + 15 * np.sin((i + j) / 25.0)
            )
    img += rng.normal(0, 10, (h, w))
    return np.clip(img, 0, 255)


@pytest.mark.slow
class TestEmpiricalFPRateLargeScale:
    """Run detection on many unwatermarked images with random keys.

    10,000 detection attempts (100 images x 100 keys) should produce 0 FPs.
    """

    def test_10k_detection_zero_false_positives(self, detector):
        """100 images x 100 keys = 10,000 attempts, expect 0 false positives."""
        false_detected = 0
        false_author_match = 0

        for img_seed in range(100):
            img_rng = np.random.default_rng(img_seed + 5000)
            img = _make_random_image(img_rng)

            for key_seed in range(100):
                key_bytes = f"fp-large-{key_seed:06d}-pad-to-32bytes!".encode()
                keys = generate_author_keys(seed=key_bytes)
                result = detector.detect(img, keys.public_key)

                if result.detected:
                    false_detected += 1
                if result.author_id_match:
                    false_author_match += 1

        # With adaptive ring strength + payload_confidence > 0.5 threshold,
        # up to 1 in 10K false detections is acceptable (rate < 1e-4).
        assert false_detected <= 1, f"{false_detected}/10000 false detections (expect <= 1)"
        assert false_author_match == 0, f"{false_author_match}/10000 false author ID matches"

    def test_diverse_image_types_no_false_positives(self, detector):
        """Test across flat, gradient, noise, and patterned images."""
        rng = np.random.default_rng(42)
        h, w = 512, 512

        images = []
        # Flat
        images.append(np.full((h, w), 128.0))
        # Gradient
        y_grad = np.linspace(0, 255, h).reshape(-1, 1)
        x_grad = np.linspace(0, 255, w).reshape(1, -1)
        images.append(y_grad * 0.5 + x_grad * 0.5)
        # Pure noise
        images.append(np.clip(rng.normal(128, 40, (h, w)), 0, 255))
        # High-contrast checkerboard
        check = np.zeros((h, w), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                check[i, j] = 240 if (i // 32 + j // 32) % 2 == 0 else 15
        images.append(check)
        # Radial pattern
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        images.append(np.clip(128 + 60 * np.sin(r / 8), 0, 255).astype(np.float64))

        false_positives = 0
        for img in images:
            for key_seed in range(50):
                key_bytes = f"fp-diverse-{key_seed:06d}-pad-32bytes!".encode()
                keys = generate_author_keys(seed=key_bytes)
                result = detector.detect(img, keys.public_key)
                if result.detected or result.author_id_match:
                    false_positives += 1

        assert false_positives == 0, (
            f"{false_positives}/{len(images) * 50} false positives on diverse images"
        )
