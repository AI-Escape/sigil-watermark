#!/usr/bin/env python3
"""VAE Passband Analysis — measure per-frequency attenuation through the SD VAE.

Identifies which frequency bands survive the VAE encode/decode bottleneck,
so ghost signal energy can be placed where it actually propagates.

Requires: torch, diffusers

Usage:
    cd backend/services/sigil-watermark
    uv run python scripts/vae_passband_analysis.py

Outputs:
    scripts/output/vae_passband_radial.npy     — radial attenuation profile
    scripts/output/vae_passband_2d.npy         — full 2D attenuation map
    scripts/output/vae_passband_plot.png        — visualization
    scripts/output/vae_passband_report.txt      — text summary with band recommendations
"""

from pathlib import Path
import sys

import numpy as np

try:
    import torch
    from diffusers import AutoencoderKL
except ImportError:
    print("ERROR: torch and diffusers required. Install with:")
    print("  pip install sigil-watermark[gpu-tests]")
    sys.exit(1)

OUTPUT_DIR = Path(__file__).parent / "output"


def load_vae():
    """Load the SD VAE model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading VAE on {device}...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32
    )
    vae = vae.to(device)
    vae.eval()
    return vae, device


def vae_roundtrip(vae, image_gray, device):
    """Pass a grayscale image through real VAE encode/decode.

    Args:
        image_gray: (H, W) float64 [0, 255]

    Returns:
        (H, W) float64 [0, 255] reconstructed image
    """
    h, w = image_gray.shape
    img_u8 = np.clip(image_gray, 0, 255).astype(np.uint8)
    rgb = np.stack([img_u8, img_u8, img_u8], axis=2)

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    tensor = (tensor / 127.5) - 1.0
    tensor = tensor.to(device)

    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.sample()
        decoded = vae.decode(latent).sample

    decoded_np = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
    decoded_np = (decoded_np + 1.0) * 127.5
    decoded_np = np.clip(decoded_np, 0, 255)

    y = 0.299 * decoded_np[:, :, 0] + 0.587 * decoded_np[:, :, 1] + 0.114 * decoded_np[:, :, 2]
    return y[:h, :w]


def generate_diverse_images(n=50, size=(512, 512)):
    """Generate diverse synthetic images for passband analysis."""
    rng = np.random.default_rng(42)
    h, w = size
    images = []

    for idx in range(n):
        img = np.zeros((h, w), dtype=np.float64)

        if idx < 10:
            # Smooth textures (low frequency)
            freq = 5 + idx * 8
            img += 128 + 50 * np.sin(np.arange(h).reshape(-1, 1) / freq)
            img += 30 * np.cos(np.arange(w).reshape(1, -1) / (freq + 3))
        elif idx < 20:
            # Detailed textures (mid frequency)
            for k in range(3):
                f1 = rng.uniform(5, 40)
                f2 = rng.uniform(5, 40)
                a = rng.uniform(20, 50)
                phase = rng.uniform(0, 2 * np.pi)
                yy = np.arange(h).reshape(-1, 1)
                xx = np.arange(w).reshape(1, -1)
                img += a * np.sin(yy / f1 + xx / f2 + phase)
            img += 128
        elif idx < 30:
            # High-frequency noise textures
            img = rng.normal(128, 40, (h, w))
        elif idx < 40:
            # Gradients with fine detail
            yy = np.linspace(0, 255, h).reshape(-1, 1)
            xx = np.linspace(0, 255, w).reshape(1, -1)
            img = yy * 0.4 + xx * 0.4
            freq = 3 + idx
            img += 20 * np.sin(np.arange(h).reshape(-1, 1) * 2 * np.pi / freq)
        else:
            # Mixed: broad spectrum images
            for k in range(5):
                f = rng.uniform(3, 80)
                angle = rng.uniform(0, 2 * np.pi)
                a = rng.uniform(10, 40)
                yy = np.arange(h).reshape(-1, 1)
                xx = np.arange(w).reshape(1, -1)
                img += a * np.sin(
                    (yy * np.cos(angle) + xx * np.sin(angle)) / f
                )
            img += 128

        img += rng.normal(0, 5, (h, w))
        images.append(np.clip(img, 0, 255))

    return images


def radial_average(map_2d):
    """Compute radial average of a 2D map centered at the middle.

    Returns:
        (radii, values) where radii is in normalized frequency [0, 0.5]
    """
    h, w = map_2d.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    n_bins = max_r
    radii = np.linspace(0, 0.5, n_bins)
    values = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        r_low = i * max_r / n_bins
        r_high = (i + 1) * max_r / n_bins
        mask = (dist >= r_low) & (dist < r_high)
        if mask.any():
            values[i] = np.mean(map_2d[mask])
            counts[i] = mask.sum()

    return radii, values


def analyze_vae_passband(vae, device, images):
    """Measure per-frequency attenuation through the VAE.

    Returns:
        (radial_profile, attenuation_map_2d)
    """
    h, w = images[0].shape
    attenuation_sum = np.zeros((h, w), dtype=np.float64)
    count = 0

    for i, img in enumerate(images):
        if (i + 1) % 10 == 0:
            print(f"  Processing image {i + 1}/{len(images)}...")

        # Power spectrum before VAE
        spec_before = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2

        # VAE roundtrip
        img_after = vae_roundtrip(vae, img, device)

        # Power spectrum after VAE
        spec_after = np.abs(np.fft.fftshift(np.fft.fft2(img_after))) ** 2

        # Per-frequency attenuation ratio
        ratio = spec_after / (spec_before + 1e-10)
        attenuation_sum += ratio
        count += 1

    attenuation_map = attenuation_sum / count

    # Radial profile
    radii, radial_values = radial_average(attenuation_map)

    return radii, radial_values, attenuation_map


def find_best_bands(radii, radial_values, n_bands=6, min_freq=0.05, max_freq=0.45):
    """Find the frequency bands with highest VAE survival.

    Returns list of (center_freq, survival_value) sorted by survival.
    """
    mask = (radii >= min_freq) & (radii <= max_freq)
    masked_radii = radii[mask]
    masked_vals = radial_values[mask]

    # Smooth to avoid noise peaks
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(masked_vals, size=5)

    # Find peaks (local maxima)
    peaks = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            peaks.append((masked_radii[i], smoothed[i]))

    # Sort by survival (highest first)
    peaks.sort(key=lambda x: x[1], reverse=True)

    # Also add the overall best if no clear peaks
    if not peaks:
        best_idx = np.argmax(smoothed)
        peaks = [(masked_radii[best_idx], smoothed[best_idx])]

    # Ensure minimum separation between selected bands
    selected = []
    min_separation = 0.04
    for freq, val in peaks:
        if all(abs(freq - sf) > min_separation for sf, _ in selected):
            selected.append((freq, val))
        if len(selected) >= n_bands:
            break

    # If we didn't find enough peaks, fill with evenly-spaced high-survival regions
    if len(selected) < n_bands:
        for freq, val in sorted(zip(masked_radii, smoothed), key=lambda x: -x[1]):
            if all(abs(freq - sf) > min_separation for sf, _ in selected):
                selected.append((freq, val))
            if len(selected) >= n_bands:
                break

    return sorted(selected, key=lambda x: x[0])


def save_plot(radii, radial_values, current_bands, best_bands, output_path):
    """Save a visualization of the VAE passband."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(radii, radial_values, "b-", linewidth=1.5, label="VAE attenuation")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No attenuation")

    # Mark current ghost bands
    for band in current_bands:
        ax.axvline(x=band, color="red", linestyle="--", alpha=0.7)
    ax.axvline(x=current_bands[0], color="red", linestyle="--", alpha=0.7,
               label="Current ghost bands")

    # Mark recommended bands
    for freq, val in best_bands:
        ax.axvline(x=freq, color="green", linestyle="-", alpha=0.5)
    ax.axvline(x=best_bands[0][0], color="green", linestyle="-", alpha=0.5,
               label="Recommended bands")

    ax.set_xlabel("Normalized Frequency (fraction of Nyquist)")
    ax.set_ylabel("Power Survival Ratio (after/before)")
    ax.set_title("SD VAE Frequency Passband — Per-Frequency Power Attenuation")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, min(2.0, radial_values.max() * 1.1))
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vae, device = load_vae()

    print("Generating 50 diverse test images...")
    images = generate_diverse_images(n=50, size=(512, 512))

    print("Analyzing VAE passband...")
    radii, radial_values, attenuation_map = analyze_vae_passband(vae, device, images)

    # Save raw data
    np.save(OUTPUT_DIR / "vae_passband_radial.npy", np.stack([radii, radial_values]))
    np.save(OUTPUT_DIR / "vae_passband_2d.npy", attenuation_map)
    print(f"Saved radial profile and 2D map to {OUTPUT_DIR}/")

    # Current ghost bands
    current_bands = (0.5, 0.25, 0.125)

    # Find best bands
    best_bands = find_best_bands(radii, radial_values, n_bands=6)

    # Check survival at current bands
    print("\n" + "=" * 70)
    print("VAE PASSBAND ANALYSIS RESULTS")
    print("=" * 70)

    print("\nCurrent ghost bands — survival at each:")
    for band in current_bands:
        idx = np.argmin(np.abs(radii - band))
        survival = radial_values[idx]
        status = "GOOD" if survival > 0.5 else "WEAK" if survival > 0.2 else "DEAD"
        print(f"  {band:.3f} Nyquist: survival = {survival:.4f}  [{status}]")

    print(f"\nRecommended bands (top {len(best_bands)} VAE-transparent frequencies):")
    for freq, val in best_bands:
        print(f"  {freq:.3f} Nyquist: survival = {val:.4f}")

    # Compute overall passband statistics
    mid_mask = (radii > 0.05) & (radii < 0.45)
    if mid_mask.any():
        mid_vals = radial_values[mid_mask]
        print(f"\nMid-frequency statistics (0.05-0.45 Nyquist):")
        print(f"  Mean survival: {mid_vals.mean():.4f}")
        print(f"  Max survival:  {mid_vals.max():.4f} at {radii[mid_mask][np.argmax(mid_vals)]:.3f}")
        print(f"  Min survival:  {mid_vals.min():.4f} at {radii[mid_mask][np.argmin(mid_vals)]:.3f}")

    # Save report
    report_path = OUTPUT_DIR / "vae_passband_report.txt"
    with open(report_path, "w") as f:
        f.write("VAE Passband Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("Current ghost bands:\n")
        for band in current_bands:
            idx = np.argmin(np.abs(radii - band))
            f.write(f"  {band:.3f}: survival = {radial_values[idx]:.4f}\n")
        f.write(f"\nRecommended bands:\n")
        for freq, val in best_bands:
            f.write(f"  {freq:.3f}: survival = {val:.4f}\n")
        f.write(f"\nRecommended config:\n")
        band_tuple = tuple(round(freq, 3) for freq, _ in best_bands)
        f.write(f"  ghost_bands = {band_tuple}\n")
        f.write(f"  ghost_bandwidth = 0.05\n")
    print(f"\nReport saved to {report_path}")

    # Save plot
    save_plot(radii, radial_values, current_bands, best_bands,
              OUTPUT_DIR / "vae_passband_plot.png")

    # Print config recommendation
    band_tuple = tuple(round(freq, 3) for freq, _ in best_bands)
    print(f"\n{'=' * 70}")
    print(f"RECOMMENDED CONFIG CHANGE:")
    print(f"  ghost_bands = {band_tuple}")
    print(f"  ghost_bandwidth = 0.05")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
