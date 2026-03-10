"""Configuration for the Sigil watermark system.

Provides :class:`SigilConfig`, an immutable dataclass that controls every
tunable parameter across the three embedding layers (DFT rings, DWT
spread-spectrum, and ghost spectral signal) as well as crypto salts,
perceptual masking, and quality targets.

The module-level :data:`DEFAULT_CONFIG` instance uses production-tuned
defaults and is shared by all other modules when no explicit config is passed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SigilConfig:
    """Immutable configuration for the Sigil watermark system.

    Parameters are grouped by layer:

    **Layer 1 — DFT Ring Anchor:** Controls the concentric frequency-domain
    rings that act as a geometric compass, surviving rotation/scale attacks.

    **Layer 2 — DWT Spread-Spectrum:** Controls the fractal tiling of
    CDMA-encoded payload (beacon + author index + author ID) into wavelet
    detail subbands.

    **Layer 3 — Ghost Signal:** Controls the spectral bias signal at
    VAE-passband frequencies that survives AI training pipelines.

    **Payload / Crypto / Masking / Quality:** Supporting parameters for
    Reed-Solomon FEC, HKDF key derivation, perceptual masking, and
    adaptive ring strength.

    Example:
        >>> from sigil_watermark import SigilConfig
        >>> cfg = SigilConfig(ring_strength=15.0, ghost_strength_multiplier=150.0)
        >>> cfg.ring_strength
        15.0
    """

    # --- Layer 1: DFT Ring Anchor ---
    # Number of concentric rings embedded in Fourier magnitude spectrum
    num_rings: int = 4
    # Min/max radius in Fourier space (as fraction of image size / 2)
    # Mid-frequency range: not too fine (lost to blur) nor too coarse (lost to crop)
    ring_radius_min: float = 0.1
    ring_radius_max: float = 0.35
    # Embedding strength for ring peaks (multiplicative magnitude boost).
    # Lower than additive era (was 20) because wider rings + multiplicative
    # scaling adds more total energy.
    ring_strength: float = 20.0
    # Gaussian width of ring profiles (fraction of Nyquist); wider rings
    # force attackers to notch-filter broader bands, increasing quality cost
    ring_width: float = 0.04

    # Number of content-dependent rings (positions derived from image hash + key)
    num_content_rings: int = 2
    # Number of sentinel rings (fixed positions, server-secret-derived)
    num_sentinel_rings: int = 2
    # Sentinel ring secret (in production, loaded from server config, not hardcoded)
    sentinel_secret: bytes = b"signarture-sentinel-v1-secret"
    # HKDF salt for sentinel ring derivation
    sentinel_salt: bytes = b"signarture-sigil-v1-sentinel"
    # HKDF salt for content-dependent ring derivation
    content_ring_salt: bytes = b"signarture-sigil-v1-content-rings"

    # --- Layer 2: Fractal Sigil Tiling ---
    # Tile sizes for multi-scale fractal embedding
    tile_sizes: tuple[int, ...] = (32, 64, 128, 256)
    # DWT wavelet family
    wavelet: str = "db4"
    # DWT decomposition levels for embedding
    dwt_levels: int = 3
    # Subbands to embed in (HL=horizontal detail, LH=vertical detail)
    embed_subbands: tuple[str, ...] = ("LH", "HL")
    # Base embedding strength (scaled by perceptual mask)
    embed_strength: float = 3.0
    # CDMA spreading factor (chips per payload bit)
    spreading_factor: int = 256

    # --- Layer 3: Training Ghost Signal ---
    # Frequency bands selected for high VAE survival (empirically measured
    # via scripts/vae_passband_analysis.py against stabilityai/sd-vae-ft-mse).
    # Survival ratios: 0.343→55×, 0.425→26×, 0.218→24×, 0.163→14×, 0.286→9×
    ghost_bands: tuple[float, ...] = (0.163, 0.218, 0.286, 0.343, 0.425)
    # Bandwidth of each ghost band (fraction of Nyquist)
    ghost_bandwidth: float = 0.05
    # Ghost signal strength multiplier (relative to embed_strength)
    # Sweep showed no quality impact up to 200× (PSNR stays >46.9dB, SSIM >0.994).
    # At 100× with VAE-optimized bands, single-image ghost detection hits 100%
    # after real SD VAE encode/decode.
    ghost_strength_multiplier: float = 200.0
    # Number of ghost hash bits for author binning (2^N bins for O(K) lookup).
    # Per-bit SNR drops by 1/sqrt(N) — 8 bits balances robustness (survives VAE)
    # with useful binning (256 bins → ~39 candidates for 10K artists).
    ghost_hash_bits: int = 8

    # --- Payload ---
    # Author ID size in bits
    author_id_bits: int = 48
    # Beacon size in bits (universal Signarture marker)
    beacon_bits: int = 8
    # Author index size in bits (for blind scanning)
    author_index_bits: int = 20
    # Reed-Solomon error correction symbols for author index
    rs_nsym: int = 8

    # --- Crypto ---
    # HKDF salt for PN sequence derivation
    pn_salt: bytes = b"signarture-sigil-v1-pn"
    # HKDF salt for ring parameter derivation
    ring_salt: bytes = b"signarture-sigil-v1-rings"
    # HKDF salt for universal beacon
    beacon_salt: bytes = b"signarture-sigil-v1-beacon"
    # Universal beacon seed (fixed, public)
    beacon_seed: bytes = b"signarture-universal-beacon-v1"
    # Universal PN seed for author index tier
    universal_pn_seed: bytes = b"signarture-universal-pn-v1"

    # --- Perceptual Masking ---
    # Minimum embedding strength (even in flat regions)
    mask_floor: float = 0.3
    # Noise sensitivity threshold for JND
    jnd_threshold: float = 3.0

    # --- Adaptive ring strength ---
    # When enabled, ring embedding alpha is scaled down on images with strong
    # spectral energy at ring frequencies so the ring layer alone stays above
    # ring_target_psnr.  Uses Parseval's theorem for exact MSE prediction.
    adaptive_ring_strength: bool = True
    # Target PSNR for the ring layer alone (dB). Since DWT + ghost add more
    # distortion, overall PSNR will be ~2-4 dB lower.
    ring_target_psnr: float = 36.0
    # Floor: alpha never drops below this fraction of the nominal value,
    # ensuring ring detection robustness even on extreme images.
    # 0.30 = ring_conf ≥ 0.16 on all tested real photos while achieving
    # avg 40 dB PSNR (vs 30 dB without adaptation).
    ring_min_alpha_fraction: float = 0.30

    # --- Quality targets ---
    target_psnr_db: float = 40.0
    target_ssim: float = 0.98


# Default configuration
DEFAULT_CONFIG = SigilConfig()
