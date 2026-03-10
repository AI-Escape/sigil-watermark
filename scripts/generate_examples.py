#!/usr/bin/env python3
"""Generate example images showing each stage of the Sigil watermark pipeline.

Outputs to scripts/output/ by default (or use --output-dir to override).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sigil_watermark.color import prepare_for_embedding, reconstruct_from_embedding
from sigil_watermark.config import DEFAULT_CONFIG
from sigil_watermark.embed import SigilEmbedder, build_payload
from sigil_watermark.keygen import (
    derive_content_ring_radii,
    derive_ring_phase_offsets,
    derive_ring_radii,
    derive_sentinel_ring_radii,
    generate_author_keys,
    get_universal_beacon_pn,
)
from sigil_watermark.perceptual import compute_perceptual_mask
from sigil_watermark.tiling import best_tile_size, tile_embed
from sigil_watermark.transforms import dwt_decompose, dwt_reconstruct, embed_dft_rings


def amplified_diff_color(a: np.ndarray, b: np.ndarray, gain: float = 20.0) -> np.ndarray:
    """Compute amplified absolute difference, output as heatmap RGB."""
    # Work on float
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)

    if a_f.ndim == 3:
        # Per-channel diff, take max across channels for visibility
        diff = np.max(np.abs(a_f - b_f), axis=2)
    else:
        diff = np.abs(a_f - b_f)

    diff = np.clip(diff * gain, 0, 255)

    # Create a heatmap: black -> red -> yellow -> white
    out = np.zeros((*diff.shape, 3), dtype=np.uint8)
    norm = diff / 255.0
    out[..., 0] = np.clip(norm * 3, 0, 1) * 255  # Red ramps up first
    out[..., 1] = np.clip((norm - 0.33) * 3, 0, 1) * 255  # Green second
    out[..., 2] = np.clip((norm - 0.66) * 3, 0, 1) * 255  # Blue last
    return out


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def save_spectrum(image: np.ndarray, path: Path):
    """Save log-magnitude spectrum without any overlays."""
    # If color, use Y channel
    if image.ndim == 3:
        y = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        y = image

    f = np.fft.fft2(y)
    f_shifted = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(f_shifted))
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10) * 255
    Image.fromarray(magnitude.astype(np.uint8)).save(path)


def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
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


def main():
    parser = argparse.ArgumentParser(description="Generate Sigil watermark example images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Directory to write output images (default: scripts/output/)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source image path (default: <output-dir>/source.jpg)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load user's color image
    source_img_path = args.source or (output_dir / "source.jpg")
    if not source_img_path.exists():
        print(f"Error: {source_img_path} not found.")
        sys.exit(1)

    source_pil = Image.open(source_img_path).convert("RGB")
    # Ensure even dimensions (required for DWT)
    w_orig, h_orig = source_pil.size
    new_w = w_orig - (w_orig % 2)
    new_h = h_orig - (h_orig % 2)
    if new_w != w_orig or new_h != h_orig:
        source_pil = source_pil.crop((0, 0, new_w, new_h))
    original = np.array(source_pil, dtype=np.float64)

    # Generate deterministic keys
    keys = generate_author_keys(seed=b"how-it-works-demo")
    cfg = DEFAULT_CONFIG

    # Save original
    Image.fromarray(to_uint8(original)).save(output_dir / "original.png")
    print("Saved original.png")

    # Use the full embedder to get the final result
    embedder = SigilEmbedder(config=cfg)
    after_all = embedder.embed(original.copy(), keys)
    Image.fromarray(to_uint8(after_all)).save(output_dir / "after_all.png")
    print("Saved after_all.png")

    # Now do per-layer intermediates on the Y channel to show what each layer adds

    y_channel, color_meta = prepare_for_embedding(original.copy())
    y_channel.copy()

    # --- Layer 1: DFT Ring Anchor ---
    # Must match SigilEmbedder.embed() exactly: derive content rings from
    # pre-ring image, and scale strength per subset so total energy matches
    # a single embed_dft_rings call with all rings.
    key_radii = derive_ring_radii(keys.public_key, config=cfg)
    sentinel_radii = derive_sentinel_ring_radii(config=cfg)
    stable_radii = np.sort(np.concatenate([key_radii, sentinel_radii]))
    stable_phase = derive_ring_phase_offsets(keys.public_key, len(stable_radii), config=cfg)
    content_radii = derive_content_ring_radii(keys.public_key, y_channel, config=cfg)
    total_rings = len(stable_radii) + len(content_radii)
    stable_strength = cfg.ring_strength * len(stable_radii) / total_rings
    content_strength = cfg.ring_strength * len(content_radii) / total_rings
    adaptive_psnr = cfg.ring_target_psnr if cfg.adaptive_ring_strength else None
    y_after_l1 = embed_dft_rings(
        y_channel.copy(),
        stable_radii,
        strength=stable_strength,
        ring_width=cfg.ring_width,
        phase_offsets=stable_phase,
        target_psnr=adaptive_psnr,
        min_alpha_fraction=cfg.ring_min_alpha_fraction,
    )
    y_after_l1 = embed_dft_rings(
        y_after_l1,
        content_radii,
        strength=content_strength,
        ring_width=cfg.ring_width,
        target_psnr=adaptive_psnr,
        min_alpha_fraction=cfg.ring_min_alpha_fraction,
    )
    after_layer1 = reconstruct_from_embedding(y_after_l1, color_meta)
    Image.fromarray(to_uint8(after_layer1)).save(output_dir / "after_layer1.png")
    print("Saved after_layer1.png")

    # Save frequency spectra (no ring highlights - show the actual data)
    save_spectrum(original, output_dir / "spectrum_before.png")
    save_spectrum(after_layer1, output_dir / "spectrum_after.png")
    print("Saved spectrum_before.png, spectrum_after.png")

    # --- Layer 2: DWT Tiled Payload ---
    # Real embedder computes mask from pre-ring Y channel (before Layer 1)
    mask = compute_perceptual_mask(y_channel, config=cfg)

    # Save perceptual mask as heatmap
    mask_norm = mask / (mask.max() + 1e-10)
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_rgb[..., 0] = np.clip(mask_norm * 2, 0, 1) * 255
    mask_rgb[..., 1] = np.clip(mask_norm, 0, 1) * 180
    mask_rgb[..., 2] = np.clip(mask_norm * 0.3, 0, 1) * 255
    Image.fromarray(mask_rgb).save(output_dir / "perceptual_mask.png")
    print("Saved perceptual_mask.png")

    encoded_payload = build_payload(keys, cfg)
    h, w = y_after_l1.shape
    max_tile = max(cfg.tile_sizes)
    pn_length = max(h * w, max_tile * max_tile)
    payload_pn = get_universal_beacon_pn(length=pn_length, config=cfg)

    y_for_l2 = y_after_l1.copy()
    coeffs = dwt_decompose(y_for_l2, wavelet=cfg.wavelet, level=cfg.dwt_levels)
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
            ts = best_tile_size((sh, sw), cfg.tile_sizes, len(encoded_payload))
            subband = tile_embed(
                subband,
                payload_pn,
                encoded_payload,
                tile_size=ts,
                strength=cfg.embed_strength * mean_mask,
                spreading_factor=cfg.spreading_factor,
            )
            new_details[sb_idx] = subband
        coeffs[level_idx] = tuple(new_details)

    y_after_l2 = dwt_reconstruct(coeffs, wavelet=cfg.wavelet)[:h, :w]
    after_layer2 = reconstruct_from_embedding(y_after_l2, color_meta)
    Image.fromarray(to_uint8(after_layer2)).save(output_dir / "after_layer2.png")
    print("Saved after_layer2.png")

    # --- Diff images ---
    # Layer 1 and 2 diffs at 20x, ghost diff at 100x (it's very subtle)
    diff_layer1 = amplified_diff_color(original, after_layer1, gain=20.0)
    diff_layer2 = amplified_diff_color(after_layer1, after_layer2, gain=20.0)
    diff_layer3 = amplified_diff_color(after_layer2, after_all, gain=100.0)  # Ghost is subtle
    diff_total = amplified_diff_color(original, after_all, gain=20.0)

    Image.fromarray(diff_layer1).save(output_dir / "diff_layer1.png")
    Image.fromarray(diff_layer2).save(output_dir / "diff_layer2.png")
    Image.fromarray(diff_layer3).save(output_dir / "diff_layer3.png")
    Image.fromarray(diff_total).save(output_dir / "diff_total.png")
    print("Saved diff images")

    # --- Ghost spectrum visualization ---
    # Show the spectrum difference highlighting where the ghost signal lives
    save_spectrum(after_layer2, output_dir / "spectrum_before_ghost.png")
    save_spectrum(after_all, output_dir / "spectrum_after_ghost.png")
    print("Saved spectrum_before_ghost.png, spectrum_after_ghost.png")

    # Compute PSNR
    mse = np.mean((original - after_all) ** 2)
    psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")
    print(f"\nFinal PSNR: {psnr:.1f} dB")
    print(f"Output directory: {output_dir}")
    print(f"Generated {len(list(output_dir.glob('*.png')))} images")


if __name__ == "__main__":
    main()
