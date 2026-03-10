"""Sigil watermark embedding pipeline.

Three-layer embedding:
  Layer 1: DFT ring anchor (geometric compass)
  Layer 2: Fractal sigil tiling (combined payload via spread-spectrum in DWT)
  Layer 3: Training ghost signal (spectral bias at diffusion upsampling frequencies)
"""

from __future__ import annotations

import numpy as np

from sigil_watermark.config import SigilConfig, DEFAULT_CONFIG
from sigil_watermark.keygen import (
    AuthorKeys,
    derive_pn_sequence,
    derive_ring_radii,
    derive_ring_phase_offsets,
    derive_content_ring_radii,
    derive_sentinel_ring_radii,
    derive_author_id,
    derive_author_index,
    get_universal_beacon_pn,
    build_ghost_composite_pn,
)
from sigil_watermark.transforms import (
    embed_dft_rings,
    dwt_decompose,
    dwt_reconstruct,
)
from sigil_watermark.perceptual import compute_perceptual_mask
from sigil_watermark.tiling import tile_embed, best_tile_size
from sigil_watermark.fec import encode_payload
from sigil_watermark.color import prepare_for_embedding, reconstruct_from_embedding


def build_payload(author_keys: AuthorKeys, config: SigilConfig) -> list[int]:
    """Build the combined payload: beacon + author_index + author_id, RS-encoded."""
    beacon_bits = [1] * config.beacon_bits
    author_index = derive_author_index(author_keys.public_key, config=config)
    author_id = derive_author_id(author_keys.public_key, config=config)
    raw_payload = beacon_bits + author_index + author_id
    return encode_payload(raw_payload, nsym=config.rs_nsym)


class SigilEmbedder:
    """Embeds the three-layer Sigil watermark into an image.

    The embedder applies three complementary layers, each targeting a
    different attack class:

    1. **DFT Ring Anchor** — concentric rings in the Fourier magnitude
       spectrum survive geometric transforms (rotation, scale, crop).
    2. **DWT Spread-Spectrum** — CDMA-encoded payload tiled across wavelet
       subbands carries the beacon, author index, and author ID.
    3. **Ghost Signal** — multiplicative spectral modulation at VAE-passband
       frequencies survives AI training pipelines (Stable Diffusion VAE).

    Args:
        config: Watermark configuration. Defaults to :data:`DEFAULT_CONFIG`.

    Example:
        >>> from sigil_watermark import SigilEmbedder, generate_author_keys
        >>> keys = generate_author_keys(seed=b"example")
        >>> embedder = SigilEmbedder()
        >>> watermarked = embedder.embed(image, keys)
    """

    def __init__(self, config: SigilConfig = DEFAULT_CONFIG):
        self.config = config

    def embed(self, image: np.ndarray, author_keys: AuthorKeys) -> np.ndarray:
        """Embed the full Sigil watermark into an image.

        Applies all three layers sequentially. The image can be grayscale
        or RGB — color images are embedded in the Y (luminance) channel
        only, preserving chrominance.

        Args:
            image: Input image as a float64 NumPy array, pixel values 0–255.
                Accepts grayscale ``(H, W)`` or RGB ``(H, W, 3)``.
            author_keys: Author's Ed25519 keypair (see :func:`generate_author_keys`).

        Returns:
            Watermarked image as float64, same shape and format as input.

        Raises:
            ValueError: If image dimensions are too small for the configured
                tile sizes.
        """
        cfg = self.config

        # Handle color: extract Y channel, embed in Y, reconstruct
        y_channel, color_meta = prepare_for_embedding(image)
        result = y_channel.copy()

        # Compute perceptual mask for adaptive embedding strength
        mask = compute_perceptual_mask(result, config=cfg)

        # --- Layer 1: DFT Ring Anchor ---
        # Key-derived rings + sentinel rings (stable — positions don't depend on image)
        key_radii = derive_ring_radii(author_keys.public_key, config=cfg)
        sentinel_radii = derive_sentinel_ring_radii(config=cfg)
        stable_radii = np.sort(np.concatenate([key_radii, sentinel_radii]))
        stable_phase = derive_ring_phase_offsets(
            author_keys.public_key, len(stable_radii), config=cfg
        )
        # Content-dependent rings (positions depend on image hash + key)
        content_radii = derive_content_ring_radii(
            author_keys.public_key, result, config=cfg
        )
        # embed_dft_rings uses per-ring alpha = (strength/50) * (4/num_rings).
        # To keep total energy constant across separate calls, scale strength
        # so each subset gets the same per-ring alpha as if all rings were in one call.
        total_rings = len(stable_radii) + len(content_radii)
        stable_strength = cfg.ring_strength * len(stable_radii) / total_rings
        content_strength = cfg.ring_strength * len(content_radii) / total_rings

        adaptive_psnr = cfg.ring_target_psnr if cfg.adaptive_ring_strength else None
        result = embed_dft_rings(
            result, stable_radii, strength=stable_strength,
            ring_width=cfg.ring_width, phase_offsets=stable_phase,
            target_psnr=adaptive_psnr,
            min_alpha_fraction=cfg.ring_min_alpha_fraction,
        )
        result = embed_dft_rings(
            result, content_radii, strength=content_strength,
            ring_width=cfg.ring_width,
            target_psnr=adaptive_psnr,
            min_alpha_fraction=cfg.ring_min_alpha_fraction,
        )

        # --- Layer 2: Fractal Sigil Tiling ---
        # Combined RS-encoded payload: beacon + index + author_id
        encoded_payload = build_payload(author_keys, cfg)

        # Use universal beacon PN for the tiled payload so blind detection works
        h, w = result.shape
        max_tile = max(cfg.tile_sizes)
        pn_length = max(h * w, max_tile * max_tile)
        payload_pn = get_universal_beacon_pn(length=pn_length, config=cfg)

        # Build composite ghost PN encoding author's ghost hash bits
        ghost_pn = build_ghost_composite_pn(
            author_keys.public_key, length=pn_length, config=cfg
        )

        # DWT decompose
        coeffs = dwt_decompose(result, wavelet=cfg.wavelet, level=cfg.dwt_levels)

        # Embed tiled payload in each DWT level's detail subbands
        for level_idx in range(1, len(coeffs)):
            detail_tuple = coeffs[level_idx]
            subband_names = ("LH", "HL", "HH")

            new_details = list(detail_tuple)
            for sb_idx, sb_name in enumerate(subband_names):
                if sb_name not in cfg.embed_subbands:
                    continue

                subband = new_details[sb_idx].copy()
                sh, sw = subband.shape
                mean_mask = _resize_mask(mask, sh, sw).mean()

                ts = best_tile_size(
                    (sh, sw), cfg.tile_sizes, len(encoded_payload)
                )

                subband = tile_embed(
                    subband, payload_pn, encoded_payload,
                    tile_size=ts,
                    strength=cfg.embed_strength * mean_mask,
                    spreading_factor=cfg.spreading_factor,
                )

                new_details[sb_idx] = subband

            coeffs[level_idx] = tuple(new_details)

        # Reconstruct from modified DWT coefficients
        result = dwt_reconstruct(coeffs, wavelet=cfg.wavelet)
        result = result[: image.shape[0], : image.shape[1]]

        # --- Layer 3: Training Ghost Signal ---
        # Ghost is embedded in Y channel only. Multi-channel (RGB) embedding
        # was tested but hurts VAE survival — the SD VAE mixes channels in its
        # latent space, destroying per-channel PN coherence.
        # Ghost PN encodes author's ghost hash bits for blind author binning.
        result = self._embed_ghost_signal(result, ghost_pn, mask)

        # Reconstruct color image if needed
        return reconstruct_from_embedding(result, color_meta)

    def _embed_ghost_signal(
        self, image: np.ndarray, author_pn: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Embed training ghost signal via multiplicative modulation.

        Uses multiplicative embedding at ghost frequency bands: the image's
        existing spectrum magnitude is modulated by ±strength based on the
        PN sequence sign. This makes the ghost signal proportional to image
        energy (robust to natural image spectral variation) and survives
        VAE encode/decode because the magnitude pattern is partially preserved.
        """
        cfg = self.config
        h, w = image.shape
        f = np.fft.fft2(image)
        f_shifted = np.fft.fftshift(f)

        cy, cx = h // 2, w // 2
        max_freq = min(h, w) // 2

        y, x = np.ogrid[:h, :w]
        freq_dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_freq

        pn_2d = author_pn[: h * w].reshape(h, w)
        # Use PN sign pattern (+1/-1) for multiplicative modulation
        pn_sign = np.sign(pn_2d)

        # Scale ghost modulation depth by perceptual mask: full strength in
        # textured regions, reduced in smooth regions (sky, gradients) where
        # banding from spectral modulation could be visible. Use sqrt for a
        # gentle reduction — the ghost is already very subtle (2% modulation
        # depth at 200×) so aggressive scaling would kill detectability
        # without meaningful quality benefit.
        mask_scale = np.sqrt(mask.mean())
        ghost_mod_depth = cfg.ghost_strength_multiplier / 10000.0 * mask_scale
        for band_freq in cfg.ghost_bands:
            band_mask = np.exp(
                -((freq_dist - band_freq) ** 2) / (2 * cfg.ghost_bandwidth**2)
            )
            # Multiplicative modulation: f *= (1 + depth * pn_sign * band_mask)
            modulation = 1.0 + ghost_mod_depth * pn_sign * band_mask
            f_shifted *= modulation

        result = np.real(np.fft.ifft2(np.fft.ifftshift(f_shifted)))
        return result


def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize perceptual mask to target dimensions using block averaging."""
    h, w = mask.shape
    if h == target_h and w == target_w:
        return mask

    block_h = max(1, h // target_h)
    block_w = max(1, w // target_w)

    result = np.zeros((target_h, target_w), dtype=np.float64)
    for i in range(target_h):
        for j in range(target_w):
            src_i = min(i * block_h, h - 1)
            src_j = min(j * block_w, w - 1)
            end_i = min(src_i + block_h, h)
            end_j = min(src_j + block_w, w)
            result[i, j] = mask[src_i:end_i, src_j:end_j].mean()

    return result
