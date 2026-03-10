"""Tests for Reed-Solomon forward error correction."""

import pytest
from reedsolo import ReedSolomonError

from sigil_watermark.fec import (
    bits_to_bytes,
    bytes_to_bits,
    decode_payload,
    encode_payload,
)


class TestBitByteConversion:
    def test_roundtrip_byte_aligned(self):
        bits = [1, 0, 1, 0, 0, 1, 1, 0]  # 0xA6
        result = bytes_to_bits(bits_to_bytes(bits))
        assert result == bits

    def test_roundtrip_multi_byte(self):
        bits = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1]
        result = bytes_to_bits(bits_to_bytes(bits))
        assert result == bits

    def test_non_byte_aligned_pads_zeros(self):
        bits = [1, 0, 1]  # 3 bits -> padded to 8 -> 10100000
        packed = bits_to_bytes(bits)
        assert len(packed) == 1
        unpacked = bytes_to_bits(packed, num_bits=3)
        assert unpacked == [1, 0, 1]

    def test_empty_bits(self):
        assert bits_to_bytes([]) == b""
        assert bytes_to_bits(b"") == []


class TestEncodePayload:
    def test_basic_encode_increases_length(self):
        bits = [1, 0, 1, 0, 0, 1, 1, 0] * 6  # 48 bits = 6 bytes
        encoded = encode_payload(bits, nsym=8)
        # 6 data bytes + 8 parity bytes = 14 bytes = 112 bits
        assert len(encoded) == 14 * 8

    def test_encode_with_different_nsym(self):
        bits = [1, 0, 1, 0, 0, 1, 1, 0] * 6  # 48 bits
        encoded = encode_payload(bits, nsym=4)
        assert len(encoded) == 10 * 8  # 6 data + 4 parity


class TestDecodePayload:
    def test_clean_roundtrip(self):
        original = [1, 0, 1, 0, 0, 1, 1, 0] * 6  # 48 bits
        encoded = encode_payload(original, nsym=8)
        decoded, errors = decode_payload(encoded, nsym=8, original_bit_count=48)
        assert decoded == original
        assert errors == 0

    def test_corrects_bit_errors(self):
        original = [1, 0, 1, 0, 0, 1, 1, 0] * 6  # 48 bits
        encoded = encode_payload(original, nsym=8)

        # Flip some bits (within correction capacity)
        corrupted = list(encoded)
        # Flip 2 bytes worth of bits (within nsym//2 = 4 symbol corrections)
        for i in range(16):  # 2 bytes = 16 bits
            corrupted[i] ^= 1

        decoded, errors = decode_payload(corrupted, nsym=8, original_bit_count=48)
        assert decoded == original
        assert errors > 0

    def test_corrects_scattered_errors(self):
        original = [1, 0, 1, 0, 0, 1, 1, 0] * 6
        encoded = encode_payload(original, nsym=8)

        corrupted = list(encoded)
        # Flip 1 bit in each of 4 different bytes (4 symbol errors = max for nsym=8)
        corrupted[0] ^= 1
        corrupted[8] ^= 1
        corrupted[16] ^= 1
        corrupted[24] ^= 1

        decoded, errors = decode_payload(corrupted, nsym=8, original_bit_count=48)
        assert decoded == original

    def test_fails_on_too_many_errors(self):
        original = [1, 0, 1, 0, 0, 1, 1, 0] * 6
        encoded = encode_payload(original, nsym=8)

        corrupted = list(encoded)
        # Flip bits in 5+ different bytes (exceeds nsym//2 = 4)
        for byte_idx in range(6):
            corrupted[byte_idx * 8] ^= 1

        with pytest.raises(ReedSolomonError):
            decode_payload(corrupted, nsym=8, original_bit_count=48)

    def test_roundtrip_various_payload_sizes(self):
        for n_bits in [8, 16, 20, 48, 64]:
            original = [i % 2 for i in range(n_bits)]
            encoded = encode_payload(original, nsym=8)
            decoded, _ = decode_payload(encoded, nsym=8, original_bit_count=n_bits)
            assert decoded == original, f"Failed for {n_bits}-bit payload"

    def test_author_id_payload(self):
        """Realistic 48-bit author ID payload."""
        author_id = [1, 0, 0, 1, 1, 0, 1, 0] * 6  # 48 bits
        encoded = encode_payload(author_id, nsym=8)

        # Simulate 4 symbol errors (max correctable with nsym=8)
        corrupted = list(encoded)
        # Flip bits in 4 different bytes
        corrupted[5] ^= 1
        corrupted[13] ^= 1
        corrupted[21] ^= 1
        corrupted[35] ^= 1

        decoded, errors = decode_payload(corrupted, nsym=8, original_bit_count=48)
        assert decoded == author_id
        assert errors > 0
