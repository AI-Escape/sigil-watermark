"""Fractal tiling for crop-robust watermark embedding.

Tiles the watermark payload independently into fixed-size blocks of the DWT
subband so that any surviving tile can recover the full payload. This is the
key mechanism for crop robustness — unlike linear PN embedding where a crop
misaligns the PN sequence, tiled embedding means each tile is self-contained.
"""

from __future__ import annotations

import numpy as np

from sigil_watermark.transforms import embed_spread_spectrum, extract_spread_spectrum


def tile_embed(
    subband: np.ndarray,
    pn_sequence: np.ndarray,
    payload_bits: list[int],
    tile_size: int,
    strength: float,
    spreading_factor: int,
) -> np.ndarray:
    """Embed payload independently in each tile of a 2D subband.

    Each tile gets the same payload with the same PN sequence, making any
    single surviving tile sufficient to recover the author ID.

    Args:
        subband: 2D DWT subband array.
        pn_sequence: Bipolar PN sequence (at least tile_size^2 long).
        payload_bits: Payload bits to embed in each tile.
        tile_size: Side length of each tile (e.g. 64).
        strength: Embedding strength.
        spreading_factor: Chips per payload bit.

    Returns:
        Modified subband with payload tiled throughout.
    """
    result = subband.copy()
    h, w = result.shape
    num_bits = len(payload_bits)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            th = min(tile_size, h - y)
            tw = min(tile_size, w - x)
            tile = result[y : y + th, x : x + tw]

            tile_n = th * tw
            tile_sf = min(spreading_factor, tile_n // num_bits)
            if tile_sf < 4:
                continue  # Tile too small for meaningful embedding

            tile_pn = pn_sequence[:tile_n]
            result[y : y + th, x : x + tw] = embed_spread_spectrum(
                tile,
                tile_pn,
                payload_bits,
                strength=strength,
                spreading_factor=tile_sf,
            )

    return result


def tile_extract(
    subband: np.ndarray,
    pn_sequence: np.ndarray,
    num_bits: int,
    tile_size: int,
    spreading_factor: int,
) -> tuple[list[int], float]:
    """Extract payload from all tiles and majority-vote across them.

    Args:
        subband: 2D DWT subband array.
        pn_sequence: Same PN sequence used for embedding.
        num_bits: Number of payload bits to extract.
        tile_size: Same tile size used for embedding.
        spreading_factor: Same spreading factor used for embedding.

    Returns:
        (voted_bits, confidence) where confidence is the average agreement ratio.
    """
    h, w = subband.shape
    all_bits: list[list[int]] = []

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            th = min(tile_size, h - y)
            tw = min(tile_size, w - x)
            tile = subband[y : y + th, x : x + tw]

            tile_n = th * tw
            tile_sf = min(spreading_factor, tile_n // num_bits)
            if tile_sf < 4:
                continue

            tile_pn = pn_sequence[:tile_n]
            bits = extract_spread_spectrum(
                tile,
                tile_pn,
                num_bits=num_bits,
                spreading_factor=tile_sf,
            )
            all_bits.append(bits)

    if not all_bits:
        return [0] * num_bits, 0.0

    return majority_vote(all_bits, num_bits)


def majority_vote(all_bits: list[list[int]], num_bits: int) -> tuple[list[int], float]:
    """Majority vote across multiple bit extractions.

    Returns:
        (voted_bits, confidence) where confidence is the average fraction
        of tiles that agree with the voted bit.
    """
    voted = []
    agreement_sum = 0.0

    for bit_idx in range(num_bits):
        votes = [bits[bit_idx] for bits in all_bits if bit_idx < len(bits)]
        if not votes:
            voted.append(0)
            continue
        ones = sum(votes)
        total = len(votes)
        if ones > total / 2:
            voted.append(1)
            agreement_sum += ones / total
        else:
            voted.append(0)
            agreement_sum += (total - ones) / total

    confidence = agreement_sum / num_bits if num_bits > 0 else 0.0
    return voted, confidence


def best_tile_size(
    subband_shape: tuple[int, int], tile_sizes: tuple[int, ...], num_bits: int
) -> int:
    """Choose the largest tile size that gives at least 2 tiles and enough capacity.

    Args:
        subband_shape: (h, w) of the subband.
        tile_sizes: Available tile sizes, sorted ascending.
        num_bits: Number of payload bits.

    Returns:
        Best tile size.
    """
    h, w = subband_shape
    # Try from largest to smallest
    for ts in sorted(tile_sizes, reverse=True):
        n_tiles_y = h // ts
        n_tiles_x = w // ts
        n_tiles = n_tiles_y * n_tiles_x
        capacity = ts * ts // num_bits  # spreading factor
        if n_tiles >= 2 and capacity >= 4:
            return ts

    # Fallback to smallest
    return min(tile_sizes)
