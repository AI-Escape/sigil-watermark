"""Reed-Solomon forward error correction for watermark payloads.

Wraps reedsolo to encode/decode bit-level payloads with configurable
redundancy. Each RS symbol is 8 bits.
"""

from __future__ import annotations

import numpy as np
from reedsolo import RSCodec, ReedSolomonError


def bits_to_bytes(bits: list[int]) -> bytes:
    """Pack a list of bits (0/1) into bytes, MSB first. Pads to byte boundary."""
    # Pad to multiple of 8
    padded = list(bits)
    remainder = len(padded) % 8
    if remainder:
        padded.extend([0] * (8 - remainder))

    result = bytearray()
    for i in range(0, len(padded), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | padded[i + j]
        result.append(byte)
    return bytes(result)


def bytes_to_bits(data: bytes, num_bits: int | None = None) -> list[int]:
    """Unpack bytes into a list of bits, MSB first."""
    bits = []
    for byte in data:
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)
    if num_bits is not None:
        bits = bits[:num_bits]
    return bits


def encode_payload(bits: list[int], nsym: int = 8) -> list[int]:
    """RS-encode a bit payload, returning encoded bits.

    Args:
        bits: Original payload bits (0/1).
        nsym: Number of RS error-correction symbols (each 8 bits).
              Can correct up to nsym//2 symbol errors.

    Returns:
        Encoded bit list (original data + parity bits).
    """
    data_bytes = bits_to_bytes(bits)
    codec = RSCodec(nsym)
    encoded = codec.encode(data_bytes)
    # encoded is data_bytes + nsym parity bytes
    return bytes_to_bits(bytes(encoded))


def decode_payload(
    bits: list[int], nsym: int = 8, original_bit_count: int | None = None
) -> tuple[list[int], int]:
    """RS-decode an encoded bit payload.

    Args:
        bits: Encoded payload bits (data + parity).
        nsym: Number of RS error-correction symbols used during encoding.
        original_bit_count: If provided, trim decoded bits to this length.

    Returns:
        (decoded_bits, num_errors_corrected).
        Raises ReedSolomonError if too many errors to correct.
    """
    encoded_bytes = bits_to_bytes(bits)
    codec = RSCodec(nsym)
    decoded_msg, decoded_msgecc, errata_pos = codec.decode(encoded_bytes)
    num_corrected = len(errata_pos)

    decoded_bits = bytes_to_bits(bytes(decoded_msg))
    if original_bit_count is not None:
        decoded_bits = decoded_bits[:original_bit_count]

    return decoded_bits, num_corrected
