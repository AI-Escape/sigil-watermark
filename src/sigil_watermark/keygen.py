"""Cryptographic key generation and PN sequence derivation for the Sigil watermark system.

Uses Ed25519 for author identity, HKDF for key derivation, and HMAC-based DRBG
for deterministic pseudo-random noise sequences.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from sigil_watermark.config import DEFAULT_CONFIG, SigilConfig


@dataclass
class AuthorKeys:
    """Ed25519 keypair for a watermark author.

    Attributes:
        private_key: 32-byte raw Ed25519 private key seed.
        public_key: 32-byte raw Ed25519 public key, used to derive all
            watermark parameters (ring positions, PN sequences, author ID).
    """

    private_key: bytes
    public_key: bytes

    @classmethod
    def from_private_key(cls, private_key_bytes: bytes) -> AuthorKeys:
        priv = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        pub = priv.public_key()
        pub_bytes = pub.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        return cls(private_key=private_key_bytes, public_key=pub_bytes)


def generate_author_keys(seed: bytes | None = None) -> AuthorKeys:
    """Generate a new Ed25519 keypair for watermark embedding.

    Args:
        seed: Optional seed bytes for deterministic key generation.
            If ``None``, a cryptographically random key is generated.
            Useful for reproducible testing.

    Returns:
        An :class:`AuthorKeys` instance with both private and public keys.

    Example:
        >>> keys = generate_author_keys(seed=b"my-secret-seed")
        >>> len(keys.public_key)
        32
    """
    if seed is not None:
        # Use HKDF to derive a 32-byte key seed from arbitrary-length seed
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"signarture-keygen-v1",
            info=b"ed25519-seed",
        )
        derived = hkdf.derive(seed)
        priv = Ed25519PrivateKey.from_private_bytes(derived)
    else:
        priv = Ed25519PrivateKey.generate()

    priv_bytes = priv.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    pub_bytes = priv.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return AuthorKeys(private_key=priv_bytes, public_key=pub_bytes)


def _hkdf_derive(key_material: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """Derive key material using HKDF-SHA256."""
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        info=info,
    )
    return hkdf.derive(key_material)


def _bytes_to_bipolar_pn(data: bytes, length: int) -> np.ndarray:
    """Convert raw bytes into a bipolar (+1/-1) PN sequence of given length.

    Uses HMAC-DRBG-like expansion: iteratively HMAC to produce enough bytes,
    then map each bit to +1 or -1.
    """
    # We need `length` bits = ceil(length / 8) bytes
    needed_bytes = (length + 7) // 8
    expanded = bytearray()
    counter = 0
    while len(expanded) < needed_bytes:
        expanded.extend(hmac.new(data, counter.to_bytes(4, "big"), hashlib.sha256).digest())
        counter += 1

    # Convert bytes to bits, then to bipolar
    bits = np.unpackbits(np.frombuffer(bytes(expanded[:needed_bytes]), dtype=np.uint8))
    bits = bits[:length]  # Trim to exact length
    # Map 0 -> -1, 1 -> +1
    return bits.astype(np.float64) * 2 - 1


def derive_pn_sequence(
    public_key: bytes,
    length: int,
    config: SigilConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Derive a deterministic bipolar (+1/-1) PN sequence from an author's public key."""
    seed = _hkdf_derive(public_key, salt=config.pn_salt, info=b"pn-sequence", length=32)
    return _bytes_to_bipolar_pn(seed, length)


def derive_ring_radii(
    public_key: bytes,
    config: SigilConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Derive DFT ring radii from an author's public key.

    Returns array of `config.num_rings` radii, each in [ring_radius_min, ring_radius_max],
    guaranteed to be distinct.
    """
    seed = _hkdf_derive(public_key, salt=config.ring_salt, info=b"ring-radii", length=32)
    rng = np.random.default_rng(seed=int.from_bytes(seed[:8], "big"))
    span = config.ring_radius_max - config.ring_radius_min

    # Generate evenly-spaced base radii with random jitter for distinctness
    base = np.linspace(0, 1, config.num_rings + 2)[1:-1]  # Exclude endpoints
    jitter = rng.uniform(-0.3 / config.num_rings, 0.3 / config.num_rings, config.num_rings)
    radii = np.clip(base + jitter, 0, 1) * span + config.ring_radius_min

    return np.sort(radii)


def derive_ring_phase_offsets(
    public_key: bytes,
    num_rings: int,
    config: SigilConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Derive key-dependent phase offsets for ring frequencies.

    Returns array of phase offsets in [0, 2*pi) for each ring.
    """
    seed = _hkdf_derive(public_key, salt=config.ring_salt, info=b"ring-phase-offsets", length=32)
    rng = np.random.default_rng(seed=int.from_bytes(seed[:8], "big"))
    return rng.uniform(0, 2 * np.pi, num_rings)


def derive_content_ring_radii(
    public_key: bytes,
    image: np.ndarray,
    config: SigilConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Derive content-dependent ring radii from image hash + author key.

    The image content is hashed (low-frequency DCT to be robust to minor
    changes) and combined with the key to produce ring positions that an
    attacker cannot predict without the original image.

    Returns array of `config.num_content_rings` radii.
    """
    # Use a coarse image hash: downsample to 8x8, take mean of blocks
    h, w = image.shape[:2]
    if image.ndim == 3:
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image
    block_h, block_w = h // 8, w // 8
    coarse = np.zeros(64, dtype=np.float64)
    for i in range(8):
        for j in range(8):
            block = gray[i * block_h : (i + 1) * block_h, j * block_w : (j + 1) * block_w]
            coarse[i * 8 + j] = block.mean()
    # Quantize to bytes for hashing (robust to small pixel changes)
    coarse_bytes = np.clip(coarse, 0, 255).astype(np.uint8).tobytes()

    # Combine image hash with key
    content_hash = hashlib.sha256(public_key + coarse_bytes).digest()
    seed = _hkdf_derive(
        content_hash, salt=config.content_ring_salt, info=b"content-ring-radii", length=32
    )
    rng = np.random.default_rng(seed=int.from_bytes(seed[:8], "big"))

    span = config.ring_radius_max - config.ring_radius_min
    radii = rng.uniform(0, 1, config.num_content_rings) * span + config.ring_radius_min
    return np.sort(radii)


def derive_sentinel_ring_radii(
    config: SigilConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Derive sentinel ring radii from server secret.

    Sentinel rings are at fixed positions known only to the server.
    If key-derived rings are removed but sentinels remain, tampering
    is suspected.

    Returns array of `config.num_sentinel_rings` radii.
    """
    seed = _hkdf_derive(
        config.sentinel_secret, salt=config.sentinel_salt, info=b"sentinel-ring-radii", length=32
    )
    rng = np.random.default_rng(seed=int.from_bytes(seed[:8], "big"))

    span = config.ring_radius_max - config.ring_radius_min
    radii = rng.uniform(0, 1, config.num_sentinel_rings) * span + config.ring_radius_min
    return np.sort(radii)


def derive_author_id(
    public_key: bytes,
    config: SigilConfig = DEFAULT_CONFIG,
) -> list[int]:
    """Derive a truncated author ID (list of bits) from the public key."""
    h = hashlib.sha256(b"signarture-author-id-v1" + public_key).digest()
    bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
    return bits[: config.author_id_bits].tolist()


def derive_author_index(
    public_key: bytes,
    config: SigilConfig = DEFAULT_CONFIG,
) -> list[int]:
    """Derive the 20-bit author index for tier-2 blind scanning."""
    h = hashlib.sha256(b"signarture-author-index-v1" + public_key).digest()
    bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))
    return bits[: config.author_index_bits].tolist()


def derive_ghost_hash(
    public_key: bytes,
    config: SigilConfig = DEFAULT_CONFIG,
) -> list[int]:
    """Derive ghost hash bits from public key for ghost-layer author binning.

    Returns a list of ghost_hash_bits bits derived deterministically from
    the public key. Used to narrow O(N) author search to O(1) via indexed
    lookup in the database.
    """
    h = hashlib.sha256(b"signarture-ghost-hash-v1" + public_key).digest()
    bits = []
    for i in range(config.ghost_hash_bits):
        byte_idx = i // 8
        bit_idx = i % 8
        bits.append((h[byte_idx] >> bit_idx) & 1)
    return bits


def get_ghost_hash_pns(
    num_bits: int,
    length: int,
    config: SigilConfig = DEFAULT_CONFIG,
) -> list[np.ndarray]:
    """Generate universal PN sequences for ghost hash bit encoding.

    Each PN is derived from a unique universal seed, independent of any
    author key. During embedding, each PN is added with a sign (+/-1)
    determined by the author's ghost hash bit. During detection, correlating
    with each PN recovers the sign = the hash bit.
    """
    pns = []
    for i in range(num_bits):
        seed_bytes = hashlib.sha256(f"signarture-ghost-hash-pn-v1-{i}".encode()).digest()
        pn = _bytes_to_bipolar_pn(seed_bytes, length)
        pns.append(pn)
    return pns


def build_ghost_composite_pn(
    public_key: bytes,
    length: int,
    config: SigilConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Build the composite ghost PN for an author.

    Encodes the author's ghost hash bits into a composite PN sequence
    by summing sign-modulated universal hash PNs. The composite has
    unit variance regardless of the number of hash bits.
    """
    ghost_hash = derive_ghost_hash(public_key, config)
    hash_pns = get_ghost_hash_pns(config.ghost_hash_bits, length, config)

    composite = np.zeros(length)
    for i, bit in enumerate(ghost_hash):
        sign = 1.0 if bit == 1 else -1.0
        composite += sign * hash_pns[i]
    composite /= np.sqrt(len(ghost_hash))
    return composite


def get_universal_beacon_pn(
    length: int,
    config: SigilConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Get the universal Signarture beacon PN sequence (same for all watermarks)."""
    seed = _hkdf_derive(config.beacon_seed, salt=config.beacon_salt, info=b"beacon-pn", length=32)
    return _bytes_to_bipolar_pn(seed, length)


def get_universal_index_pn(
    length: int,
    config: SigilConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Get the universal PN sequence used for author index embedding (tier 2)."""
    seed = _hkdf_derive(
        config.universal_pn_seed, salt=config.beacon_salt, info=b"index-pn", length=32
    )
    return _bytes_to_bipolar_pn(seed, length)
