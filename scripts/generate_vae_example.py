#!/usr/bin/env python3
"""Generate VAE roundtrip example images.

Shows: original -> watermarked -> VAE encode/decode -> ghost hash extracted -> author identified.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sigil_watermark.config import DEFAULT_CONFIG
from sigil_watermark.keygen import generate_author_keys, derive_ghost_hash
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.detect import SigilDetector
from sigil_watermark.ghost.spectral_analysis import extract_ghost_hash


def vae_roundtrip(vae, image_rgb: np.ndarray) -> np.ndarray:
    """Run image through SD VAE encode/decode cycle."""
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype

    # Grayscale/RGB -> 3-channel tensor [-1, 1]
    if image_rgb.ndim == 2:
        rgb = np.stack([image_rgb] * 3, axis=-1)
    else:
        rgb = image_rgb

    img = rgb.astype(np.float32) / 255.0
    img = (img * 2.0) - 1.0  # [-1, 1]
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

    # Pad to multiple of 8 for VAE
    _, _, h, w = tensor.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.sample()
        decoded = vae.decode(latent).sample

    # Crop back
    decoded = decoded[:, :, :h, :w]

    # Back to numpy RGB [0, 255]
    out = decoded.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    out = ((out + 1.0) / 2.0) * 255.0
    return np.clip(out, 0, 255)


def amplified_diff(a: np.ndarray, b: np.ndarray, gain: float = 30.0) -> np.ndarray:
    """Amplified absolute difference as heatmap."""
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    if a_f.ndim == 3:
        diff = np.max(np.abs(a_f - b_f), axis=2)
    else:
        diff = np.abs(a_f - b_f)
    diff = np.clip(diff * gain, 0, 255)
    norm = diff / 255.0
    out = np.zeros((*diff.shape, 3), dtype=np.uint8)
    out[..., 0] = np.clip(norm * 3, 0, 1) * 255
    out[..., 1] = np.clip((norm - 0.33) * 3, 0, 1) * 255
    out[..., 2] = np.clip((norm - 0.66) * 3, 0, 1) * 255
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate VAE roundtrip example images.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "output",
        help="Directory to write output images (default: scripts/output/)",
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Source image path (default: <output-dir>/source.jpg)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source image
    source_path = args.source or (output_dir / "source.jpg")
    source_pil = Image.open(source_path).convert("RGB")
    w, h = source_pil.size
    w -= w % 2
    h -= h % 2
    source_pil = source_pil.crop((0, 0, w, h))
    original = np.array(source_pil, dtype=np.float64)
    print(f"Source image: {w}x{h}")

    cfg = DEFAULT_CONFIG
    keys = generate_author_keys(seed=b"how-it-works-demo")
    expected_ghost_hash = derive_ghost_hash(keys.public_key, cfg)
    print(f"Author ghost hash: {''.join(str(b) for b in expected_ghost_hash)}")

    # Embed watermark
    embedder = SigilEmbedder(config=cfg)
    watermarked = embedder.embed(original.copy(), keys)
    watermarked_uint8 = np.clip(watermarked, 0, 255).astype(np.uint8)
    Image.fromarray(watermarked_uint8).save(output_dir / "vae_watermarked.png")
    print("Saved vae_watermarked.png")

    # Load SD VAE
    print("Loading SD 1.5 VAE (stabilityai/sd-vae-ft-mse)...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32)
    vae = vae.to("cuda").eval()
    print("VAE loaded")

    # VAE roundtrip
    print("Running VAE encode/decode...")
    after_vae = vae_roundtrip(vae, watermarked_uint8)
    after_vae_uint8 = np.clip(after_vae, 0, 255).astype(np.uint8)
    Image.fromarray(after_vae_uint8).save(output_dir / "vae_after.png")
    print("Saved vae_after.png")

    # Diff: watermarked vs after VAE (shows what VAE destroyed)
    diff_vae = amplified_diff(watermarked_uint8, after_vae_uint8, gain=10.0)
    Image.fromarray(diff_vae).save(output_dir / "vae_diff.png")
    print("Saved vae_diff.png")

    # Extract ghost hash (blind - no key needed)
    extracted_hash, confidences = extract_ghost_hash(after_vae, cfg)
    extracted_str = "".join(str(b) for b in extracted_hash)
    expected_str = "".join(str(b) for b in expected_ghost_hash)
    errors = sum(a != b for a, b in zip(extracted_hash, expected_ghost_hash))
    print(f"\nGhost hash extraction after VAE:")
    print(f"  Expected:  {expected_str}")
    print(f"  Extracted: {extracted_str}")
    print(f"  Bit errors: {errors}/8")
    print(f"  Match (Hamming <= 2): {'YES' if errors <= 2 else 'NO'}")
    print(f"  Per-bit confidences: {[f'{c:.3f}' for c in confidences]}")

    # Full detection with correct key
    detector = SigilDetector(config=cfg)
    result = detector.detect(after_vae, keys.public_key)
    print(f"\nFull detection after VAE:")
    print(f"  Detected: {result.detected}")
    print(f"  Ghost confidence: {result.ghost_confidence:.3f}")
    print(f"  Ghost hash match: {result.ghost_hash_match}")
    print(f"  Ring confidence: {result.ring_confidence:.3f}")
    print(f"  Payload confidence: {result.payload_confidence:.3f}")

    # PSNR between original and watermarked
    mse_wm = np.mean((original - watermarked) ** 2)
    psnr_wm = 10 * np.log10(255.0**2 / mse_wm) if mse_wm > 0 else float("inf")
    print(f"\nQuality: watermarked PSNR = {psnr_wm:.1f} dB")

    # Save a summary text file for reference
    summary = (
        f"VAE Roundtrip Example\n"
        f"====================\n"
        f"Image: {w}x{h}\n"
        f"VAE: stabilityai/sd-vae-ft-mse\n"
        f"Watermark PSNR: {psnr_wm:.1f} dB\n"
        f"Ghost hash (expected):  {expected_str}\n"
        f"Ghost hash (extracted): {extracted_str}\n"
        f"Bit errors: {errors}/8\n"
        f"Match: {'YES' if errors <= 2 else 'NO'}\n"
        f"Ghost confidence: {result.ghost_confidence:.3f}\n"
    )
    (output_dir / "vae_summary.txt").write_text(summary)
    print(f"\nSaved summary to vae_summary.txt")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
