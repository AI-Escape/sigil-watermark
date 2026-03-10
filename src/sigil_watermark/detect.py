"""Sigil watermark detection pipeline.

Three-tier detection:
  Tier 1: Universal beacon check (is this a Signarture watermark?)
  Tier 2: Author index extraction (whose watermark is it?)
  Tier 3: Author verification (cryptographic proof of authorship)

All three tiers are extracted from a single combined payload that is
RS-encoded and tiled across DWT subbands for crop robustness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from reedsolo import ReedSolomonError

from sigil_watermark.config import SigilConfig, DEFAULT_CONFIG
from sigil_watermark.keygen import (
    derive_ring_radii,
    derive_ring_phase_offsets,
    derive_content_ring_radii,
    derive_sentinel_ring_radii,
    derive_author_id,
    derive_author_index,
    get_universal_beacon_pn,
)
from sigil_watermark.transforms import (
    detect_dft_rings,
    dwt_decompose,
)
from sigil_watermark.tiling import tile_extract, best_tile_size
from sigil_watermark.fec import encode_payload, decode_payload
from sigil_watermark.color import extract_y_channel
from sigil_watermark.geometric import auto_correct, try_rotations
from sigil_watermark.ghost.spectral_analysis import analyze_ghost_signature, extract_ghost_hash
from sigil_watermark.keygen import derive_ghost_hash


def _encoded_payload_length(config: SigilConfig) -> int:
    """Compute the length of the RS-encoded payload in bits."""
    raw_len = config.beacon_bits + config.author_index_bits + config.author_id_bits
    dummy = [0] * raw_len
    return len(encode_payload(dummy, nsym=config.rs_nsym))


@dataclass
class DetectionResult:
    """Result of watermark detection.

    Contains per-layer confidence scores and the outputs of all three
    detection tiers: beacon presence, author index extraction, and
    cryptographic author verification.

    Attributes:
        detected: ``True`` if any watermark was found with sufficient confidence.
        confidence: Overall confidence score (0–1), a weighted blend of
            ring (35%), payload (45%), and ghost (20%) confidences.
        author_id_match: ``True`` if the extracted author ID matches the
            provided public key (Tier 3 verification).
        beacon_found: ``True`` if the universal Signarture beacon was
            detected (Tier 1).
        author_index: Extracted 20-bit author index as a list of ints,
            or ``None`` if RS decoding failed (Tier 2).
        ring_confidence: DFT ring detection confidence (0–1).
        payload_confidence: Spread-spectrum payload correlation (0–1).
        ghost_confidence: Ghost signal correlation strength (0–1).
        ghost_hash: Extracted ghost hash bits (blind, no key needed),
            or ``None`` if extraction failed.
        ghost_hash_match: ``True`` if the ghost hash matches the candidate key.
        tampering_suspected: ``True`` if sentinel rings are present but
            key-derived rings have been selectively removed.
    """

    detected: bool
    confidence: float
    author_id_match: bool
    beacon_found: bool
    author_index: list[int] | None
    ring_confidence: float
    payload_confidence: float
    ghost_confidence: float = 0.0
    ghost_hash: list[int] | None = None
    ghost_hash_match: bool = False
    tampering_suspected: bool = False


class SigilDetector:
    """Detects and verifies the three-layer Sigil watermark.

    Implements three detection tiers:

    1. **Beacon** — universal marker shared by all Signarture watermarks.
       Answers "is this image watermarked at all?"
    2. **Author Index** — 20-bit index for O(1) database lookup.
       Answers "whose watermark is this?" without the author's key.
    3. **Author Verification** — cryptographic proof using the author's
       Ed25519 public key. Answers "does this specific key match?"

    Includes automatic geometric correction: if initial detection is weak
    but rings are found, common rotation corrections are tried.

    Args:
        config: Watermark configuration. Defaults to :data:`DEFAULT_CONFIG`.

    Example:
        >>> from sigil_watermark import SigilDetector
        >>> detector = SigilDetector()
        >>> result = detector.detect(image, public_key)
        >>> if result.detected:
        ...     print(f"Watermark found (confidence={result.confidence:.2f})")
    """

    def __init__(self, config: SigilConfig = DEFAULT_CONFIG):
        self.config = config

    def _extract_combined_payload(self, image: np.ndarray) -> tuple[list[int], float]:
        """Extract the combined tiled payload from DWT subbands.

        Args:
            image: Grayscale (H,W) or RGB (H,W,3) image.

        Returns:
            (voted_encoded_bits, avg_confidence)
        """
        cfg = self.config
        y = extract_y_channel(image)
        coeffs = dwt_decompose(y, wavelet=cfg.wavelet, level=cfg.dwt_levels)

        max_tile = max(cfg.tile_sizes)
        pn_length = max(image.shape[0] * image.shape[1], max_tile * max_tile)
        payload_pn = get_universal_beacon_pn(length=pn_length, config=cfg)
        encoded_len = _encoded_payload_length(cfg)

        all_bits = []
        all_conf = []

        for level_idx in range(1, len(coeffs)):
            detail_tuple = coeffs[level_idx]
            subband_names = ("LH", "HL", "HH")
            for sb_idx, sb_name in enumerate(subband_names):
                if sb_name not in cfg.embed_subbands:
                    continue
                subband = detail_tuple[sb_idx]

                ts = best_tile_size(subband.shape, cfg.tile_sizes, encoded_len)

                bits, conf = tile_extract(
                    subband, payload_pn,
                    num_bits=encoded_len,
                    tile_size=ts,
                    spreading_factor=cfg.spreading_factor,
                )
                all_bits.append(bits)
                all_conf.append(conf)

        if not all_bits:
            return [0] * encoded_len, 0.0

        # Majority vote across subbands/levels
        voted = []
        for bit_idx in range(encoded_len):
            votes = [bits[bit_idx] for bits in all_bits if bit_idx < len(bits)]
            if votes:
                voted.append(1 if sum(votes) > len(votes) / 2 else 0)
            else:
                voted.append(0)

        avg_conf = sum(all_conf) / len(all_conf) if all_conf else 0.0
        return voted, avg_conf

    def _decode_combined_payload(
        self, encoded_bits: list[int]
    ) -> tuple[list[int] | None, list[int] | None, list[int] | None, bool]:
        """RS-decode and split the combined payload.

        Returns:
            (beacon_bits, author_index, author_id, rs_success)
        """
        cfg = self.config
        raw_len = cfg.beacon_bits + cfg.author_index_bits + cfg.author_id_bits

        try:
            decoded, num_corrected = decode_payload(
                encoded_bits, nsym=cfg.rs_nsym, original_bit_count=raw_len
            )
            beacon = decoded[:cfg.beacon_bits]
            index = decoded[cfg.beacon_bits:cfg.beacon_bits + cfg.author_index_bits]
            author_id = decoded[cfg.beacon_bits + cfg.author_index_bits:]
            return beacon, index, author_id, True
        except ReedSolomonError:
            return None, None, None, False

    def detect_beacon(self, image: np.ndarray) -> bool:
        """Tier 1: Check if the image contains a Signarture beacon."""
        encoded_bits, conf = self._extract_combined_payload(image)
        beacon, _, _, rs_ok = self._decode_combined_payload(encoded_bits)

        if rs_ok and beacon is not None:
            match_count = sum(1 for b in beacon if b == 1)
            return match_count / self.config.beacon_bits > 0.7

        # RS failed — try raw check on first beacon_bits of the encoded data
        # (won't be accurate but gives a rough signal)
        return False

    def extract_author_index(self, image: np.ndarray) -> list[int] | None:
        """Tier 2: Extract the author index from the combined payload."""
        encoded_bits, conf = self._extract_combined_payload(image)
        _, index, _, rs_ok = self._decode_combined_payload(encoded_bits)
        return index if rs_ok else None

    def _detect_on_image(self, y: np.ndarray, public_key: bytes) -> DetectionResult:
        """Core detection logic on a single grayscale image."""
        cfg = self.config

        # --- Ring detection ---
        # Stable rings: key-derived + sentinel (positions don't depend on image)
        key_radii = derive_ring_radii(public_key, config=cfg)
        sentinel_radii = derive_sentinel_ring_radii(config=cfg)
        stable_radii = np.sort(np.concatenate([key_radii, sentinel_radii]))
        stable_phase = derive_ring_phase_offsets(
            public_key, len(stable_radii), config=cfg
        )

        _, ring_confidence = detect_dft_rings(
            y, stable_radii, tolerance=0.02, ring_width=cfg.ring_width,
            phase_offsets=stable_phase,
        )

        # Tampering detection: sentinel rings present but key rings absent
        _, sentinel_conf = detect_dft_rings(
            y, sentinel_radii, tolerance=0.02, ring_width=cfg.ring_width,
        )
        _, key_ring_conf = detect_dft_rings(
            y, key_radii, tolerance=0.02, ring_width=cfg.ring_width,
        )
        tampering_suspected = sentinel_conf > 0.5 and key_ring_conf < 0.2

        # --- Extract combined tiled payload ---
        encoded_bits, tile_conf = self._extract_combined_payload(y)
        beacon_bits, author_index, extracted_id, rs_ok = self._decode_combined_payload(
            encoded_bits
        )

        # --- Tier 1: Beacon ---
        if rs_ok and beacon_bits is not None:
            beacon_match = sum(1 for b in beacon_bits if b == 1) / cfg.beacon_bits
            beacon_found = beacon_match > 0.7
        else:
            beacon_found = False

        # --- Tier 3: Author verification ---
        expected_author_id = derive_author_id(public_key, config=cfg)

        if rs_ok and extracted_id is not None:
            errors = sum(a != b for a, b in zip(expected_author_id, extracted_id))
            ber = errors / cfg.author_id_bits
            author_id_match = ber < 0.15
            payload_confidence = 1.0 - ber
        else:
            raw_payload = [1] * cfg.beacon_bits + \
                derive_author_index(public_key, config=cfg) + \
                derive_author_id(public_key, config=cfg)
            expected_encoded = encode_payload(raw_payload, nsym=cfg.rs_nsym)
            errors = sum(a != b for a, b in zip(expected_encoded, encoded_bits))
            ber = errors / len(expected_encoded)
            author_id_match = ber < 0.25
            payload_confidence = max(0.0, 1.0 - ber)

        # --- Ghost signal analysis ---
        ghost_result = analyze_ghost_signature(y, public_key, cfg)

        # Ghost hash: compare extracted bits with expected
        extracted_ghost_hash = ghost_result.ghost_hash
        expected_ghost_hash = derive_ghost_hash(public_key, cfg)
        if extracted_ghost_hash is not None:
            ghost_hash_errors = sum(
                a != b for a, b in zip(extracted_ghost_hash, expected_ghost_hash)
            )
            ghost_hash_match = ghost_hash_errors <= 2
        else:
            ghost_hash_errors = cfg.ghost_hash_bits
            ghost_hash_match = False

        # Ghost confidence combines correlation strength with hash match quality.
        # The composite PN approach means a wrong key with a similar hash still
        # shows partial correlation, so we scale by hash bit error rate to
        # discriminate authors.
        raw_ghost_conf = min(1.0, max(0.0, ghost_result.correlation / 0.015))
        hash_ber = ghost_hash_errors / max(cfg.ghost_hash_bits, 1)
        ghost_confidence = raw_ghost_conf * max(0.0, 1.0 - hash_ber * 2.0)

        detected = bool(payload_confidence > 0.5 and (
            beacon_found or ring_confidence > 0.5 or author_id_match
        ))

        # Overall confidence: weighted blend of all three layers
        overall_confidence = min(1.0, (
            0.35 * ring_confidence +
            0.45 * payload_confidence +
            0.20 * ghost_confidence
        ))

        return DetectionResult(
            detected=detected,
            confidence=overall_confidence,
            author_id_match=author_id_match,
            beacon_found=beacon_found,
            author_index=author_index,
            ring_confidence=ring_confidence,
            payload_confidence=payload_confidence,
            ghost_confidence=ghost_confidence,
            ghost_hash=extracted_ghost_hash,
            ghost_hash_match=ghost_hash_match,
            tampering_suspected=tampering_suspected,
        )

    def detect(self, image: np.ndarray, public_key: bytes) -> DetectionResult:
        """Full three-tier detection with a known candidate public key.

        Includes automatic geometric correction: if initial detection is weak,
        tries common rotation corrections and uses the best result.

        Args:
            image: Grayscale (H,W) or RGB (H,W,3) image to check.
            public_key: Candidate author's public key

        Returns:
            DetectionResult with all detection details.
        """
        y = extract_y_channel(image)

        # First attempt: detect without correction
        result = self._detect_on_image(y, public_key)

        # If detection is confident, return immediately
        if result.detected and result.payload_confidence > 0.7:
            return result

        # If ring confidence is high but payload is low, try geometric correction
        if result.ring_confidence > 0.3 and result.payload_confidence < 0.5:
            def conf_fn(img):
                r = self._detect_on_image(img, public_key)
                return r.payload_confidence

            corrected, best_angle, best_conf = try_rotations(y, conf_fn)
            if best_conf > result.payload_confidence:
                result = self._detect_on_image(corrected, public_key)

        return result
