"""Tests for the keygen module — crypto key generation, PN sequence derivation, author ID."""

import numpy as np
import pytest

from sigil_watermark.keygen import (
    AuthorKeys,
    generate_author_keys,
    derive_pn_sequence,
    derive_ring_radii,
    derive_author_id,
    derive_author_index,
    get_universal_beacon_pn,
    get_universal_index_pn,
)
from sigil_watermark.config import SigilConfig


class TestGenerateAuthorKeys:
    def test_returns_author_keys(self):
        keys = generate_author_keys()
        assert isinstance(keys, AuthorKeys)

    def test_private_key_is_32_bytes(self):
        keys = generate_author_keys()
        assert len(keys.private_key) == 32

    def test_public_key_is_32_bytes(self):
        keys = generate_author_keys()
        assert len(keys.public_key) == 32

    def test_different_calls_produce_different_keys(self):
        k1 = generate_author_keys()
        k2 = generate_author_keys()
        assert k1.private_key != k2.private_key
        assert k1.public_key != k2.public_key

    def test_from_private_key_recovers_public_key(self):
        keys = generate_author_keys()
        recovered = AuthorKeys.from_private_key(keys.private_key)
        assert recovered.public_key == keys.public_key

    def test_from_seed_is_deterministic(self):
        seed = b"test-seed-for-deterministic-keygen-00"
        k1 = generate_author_keys(seed=seed)
        k2 = generate_author_keys(seed=seed)
        assert k1.private_key == k2.private_key
        assert k1.public_key == k2.public_key

    def test_different_seeds_different_keys(self):
        k1 = generate_author_keys(seed=b"seed-a-that-is-long-enough-for-32")
        k2 = generate_author_keys(seed=b"seed-b-that-is-long-enough-for-32")
        assert k1.private_key != k2.private_key


class TestDerivePNSequence:
    def test_output_length(self, config):
        keys = generate_author_keys(seed=b"test-pn-len-seed-that-is-32bytes")
        pn = derive_pn_sequence(keys.public_key, length=1024, config=config)
        assert len(pn) == 1024

    def test_deterministic_from_same_key(self, config):
        keys = generate_author_keys(seed=b"test-pn-det-seed-that-is-32bytes")
        pn1 = derive_pn_sequence(keys.public_key, length=512, config=config)
        pn2 = derive_pn_sequence(keys.public_key, length=512, config=config)
        np.testing.assert_array_equal(pn1, pn2)

    def test_different_keys_produce_different_sequences(self, config):
        k1 = generate_author_keys(seed=b"test-pn-diff-a-seed-is-32-bytes!")
        k2 = generate_author_keys(seed=b"test-pn-diff-b-seed-is-32-bytes!")
        pn1 = derive_pn_sequence(k1.public_key, length=1024, config=config)
        pn2 = derive_pn_sequence(k2.public_key, length=1024, config=config)
        assert not np.array_equal(pn1, pn2)

    def test_values_are_bipolar(self, config):
        """PN sequence should be +1/-1 (bipolar) for spread spectrum."""
        keys = generate_author_keys(seed=b"test-pn-bipolar-seed-32-bytes!!")
        pn = derive_pn_sequence(keys.public_key, length=4096, config=config)
        unique_vals = set(pn.tolist())
        assert unique_vals == {-1.0, 1.0}

    def test_approximately_balanced(self, config):
        """Roughly equal number of +1 and -1 (within statistical tolerance)."""
        keys = generate_author_keys(seed=b"test-pn-balance-seed-32-bytes!!")
        pn = derive_pn_sequence(keys.public_key, length=10000, config=config)
        ones_count = np.sum(pn == 1)
        # Should be ~50% within 3 sigma for N=10000
        assert 4700 < ones_count < 5300

    def test_low_cross_correlation(self, config):
        """PN sequences from different keys should have low cross-correlation."""
        keys_list = [
            generate_author_keys(seed=f"test-pn-xcorr-{i}-seed-32-bytes!".encode())
            for i in range(5)
        ]
        pns = [
            derive_pn_sequence(k.public_key, length=4096, config=config)
            for k in keys_list
        ]

        for i in range(len(pns)):
            for j in range(i + 1, len(pns)):
                xcorr = np.abs(np.dot(pns[i], pns[j])) / len(pns[i])
                # Cross-correlation should be small (near zero for random sequences)
                # For N=4096, expected magnitude ~1/sqrt(N) = ~0.016
                assert xcorr < 0.05, f"Cross-correlation between key {i} and {j} too high: {xcorr}"

    def test_auto_correlation_peak(self, config):
        """Auto-correlation should have a clear peak at zero lag."""
        keys = generate_author_keys(seed=b"test-pn-acorr-seed-32-bytes!!!!")
        pn = derive_pn_sequence(keys.public_key, length=4096, config=config)
        auto_corr = np.dot(pn, pn) / len(pn)
        assert auto_corr == pytest.approx(1.0)


class TestDeriveRingRadii:
    def test_returns_correct_count(self, config):
        keys = generate_author_keys(seed=b"test-ring-count-seed-32-bytes!!")
        radii = derive_ring_radii(keys.public_key, config=config)
        assert len(radii) == config.num_rings

    def test_radii_within_bounds(self, config):
        keys = generate_author_keys(seed=b"test-ring-bounds-seed-32-bytes!")
        radii = derive_ring_radii(keys.public_key, config=config)
        for r in radii:
            assert config.ring_radius_min <= r <= config.ring_radius_max

    def test_deterministic(self, config):
        keys = generate_author_keys(seed=b"test-ring-det-seed-32-bytes!!!!")
        r1 = derive_ring_radii(keys.public_key, config=config)
        r2 = derive_ring_radii(keys.public_key, config=config)
        np.testing.assert_array_equal(r1, r2)

    def test_radii_are_distinct(self, config):
        keys = generate_author_keys(seed=b"test-ring-dist-seed-32-bytes!!!")
        radii = derive_ring_radii(keys.public_key, config=config)
        # All radii should be meaningfully different
        for i in range(len(radii)):
            for j in range(i + 1, len(radii)):
                assert abs(radii[i] - radii[j]) > 0.01

    def test_different_keys_different_radii(self, config):
        k1 = generate_author_keys(seed=b"test-ring-diffa-seed-32-bytes!!")
        k2 = generate_author_keys(seed=b"test-ring-diffb-seed-32-bytes!!")
        r1 = derive_ring_radii(k1.public_key, config=config)
        r2 = derive_ring_radii(k2.public_key, config=config)
        assert not np.array_equal(r1, r2)


class TestDeriveAuthorId:
    def test_correct_bit_length(self, config):
        keys = generate_author_keys(seed=b"test-aid-len-seed-32-bytes!!!!!!")
        aid = derive_author_id(keys.public_key, config=config)
        assert len(aid) == config.author_id_bits

    def test_is_binary(self, config):
        keys = generate_author_keys(seed=b"test-aid-bin-seed-32-bytes!!!!!!")
        aid = derive_author_id(keys.public_key, config=config)
        assert all(b in (0, 1) for b in aid)

    def test_deterministic(self, config):
        keys = generate_author_keys(seed=b"test-aid-det-seed-32-bytes!!!!!!")
        a1 = derive_author_id(keys.public_key, config=config)
        a2 = derive_author_id(keys.public_key, config=config)
        assert a1 == a2

    def test_different_keys_different_ids(self, config):
        k1 = generate_author_keys(seed=b"test-aid-diffa-seed-32-bytes!!!!")
        k2 = generate_author_keys(seed=b"test-aid-diffb-seed-32-bytes!!!!")
        a1 = derive_author_id(k1.public_key, config=config)
        a2 = derive_author_id(k2.public_key, config=config)
        assert a1 != a2


class TestDeriveAuthorIndex:
    def test_correct_bit_length(self, config):
        keys = generate_author_keys(seed=b"test-idx-len-seed-32-bytes!!!!!!")
        idx = derive_author_index(keys.public_key, config=config)
        assert len(idx) == config.author_index_bits

    def test_is_binary(self, config):
        keys = generate_author_keys(seed=b"test-idx-bin-seed-32-bytes!!!!!!")
        idx = derive_author_index(keys.public_key, config=config)
        assert all(b in (0, 1) for b in idx)

    def test_deterministic(self, config):
        keys = generate_author_keys(seed=b"test-idx-det-seed-32-bytes!!!!!!")
        i1 = derive_author_index(keys.public_key, config=config)
        i2 = derive_author_index(keys.public_key, config=config)
        assert i1 == i2

    def test_uniqueness_across_many_keys(self, config):
        """Generate 100 keys and check all author indices are unique."""
        indices = set()
        for i in range(100):
            keys = generate_author_keys(seed=f"test-idx-uniq-{i:04d}-32-bytes!!".encode())
            idx = derive_author_index(keys.public_key, config=config)
            idx_tuple = tuple(idx)
            indices.add(idx_tuple)
        # All 100 should be unique (vanishingly unlikely to collide with 20 bits = 1M space)
        assert len(indices) == 100


class TestUniversalSequences:
    def test_beacon_pn_is_deterministic(self, config):
        b1 = get_universal_beacon_pn(length=256, config=config)
        b2 = get_universal_beacon_pn(length=256, config=config)
        np.testing.assert_array_equal(b1, b2)

    def test_beacon_pn_is_bipolar(self, config):
        b = get_universal_beacon_pn(length=1024, config=config)
        assert set(b.tolist()) == {-1.0, 1.0}

    def test_index_pn_is_deterministic(self, config):
        p1 = get_universal_index_pn(length=512, config=config)
        p2 = get_universal_index_pn(length=512, config=config)
        np.testing.assert_array_equal(p1, p2)

    def test_index_pn_is_bipolar(self, config):
        p = get_universal_index_pn(length=1024, config=config)
        assert set(p.tolist()) == {-1.0, 1.0}

    def test_beacon_and_index_pn_are_different(self, config):
        b = get_universal_beacon_pn(length=1024, config=config)
        p = get_universal_index_pn(length=1024, config=config)
        assert not np.array_equal(b, p)

    def test_universal_pns_low_correlation_with_author_pn(self, config):
        """Universal PNs should not correlate with any author's PN."""
        beacon = get_universal_beacon_pn(length=4096, config=config)
        index_pn = get_universal_index_pn(length=4096, config=config)

        for i in range(5):
            keys = generate_author_keys(seed=f"test-univ-xcorr-{i}-32-bytes!!".encode())
            author_pn = derive_pn_sequence(keys.public_key, length=4096, config=config)

            xcorr_beacon = np.abs(np.dot(beacon, author_pn)) / 4096
            xcorr_index = np.abs(np.dot(index_pn, author_pn)) / 4096

            assert xcorr_beacon < 0.05
            assert xcorr_index < 0.05
