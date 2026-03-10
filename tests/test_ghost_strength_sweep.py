"""Ghost strength sweep — find optimal ghost_strength_multiplier.

Sweeps ghost_strength_multiplier across a range, measuring:
- Visual quality (PSNR, SSIM)
- Ghost correlation on original watermarked images
- Ghost correlation after real VAE roundtrip
- Single-image detection rate after VAE

Requires: torch, diffusers (install via `pip install sigil-watermark[gpu-tests]`)
"""

import numpy as np
import pytest

try:
    import torch
    from diffusers import AutoencoderKL

    HAS_GPU_DEPS = True
except ImportError:
    HAS_GPU_DEPS = False

from sigil_watermark.config import SigilConfig
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.ghost.spectral_analysis import analyze_ghost_signature
from sigil_watermark.keygen import generate_author_keys

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_GPU_DEPS, reason="torch/diffusers not installed"),
]


def _make_test_images(rng, n=10, size=(512, 512), rgb=False):
    """Generate diverse test images."""
    images = []
    h, w = size
    for idx in range(n):
        img = np.zeros((h, w), dtype=np.float64)
        # Mix of frequencies
        for k in range(3):
            f1 = rng.uniform(8, 50)
            f2 = rng.uniform(8, 50)
            a = rng.uniform(15, 40)
            phase = rng.uniform(0, 2 * np.pi)
            yy = np.arange(h).reshape(-1, 1)
            xx = np.arange(w).reshape(1, -1)
            img += a * np.sin(yy / f1 + xx / f2 + phase)
        img += 128 + rng.normal(0, 8, (h, w))
        img = np.clip(img, 0, 255)
        if rgb:
            # Create RGB with slight channel variation
            r = np.clip(img + rng.normal(0, 3, (h, w)), 0, 255)
            g = np.clip(img + rng.normal(0, 3, (h, w)), 0, 255)
            b = np.clip(img + rng.normal(0, 3, (h, w)), 0, 255)
            img = np.stack([r, g, b], axis=-1)
        images.append(img)
    return images


def _vae_roundtrip(vae_model, image: np.ndarray) -> np.ndarray:
    """Pass an image through real SD VAE encode/decode.

    Accepts grayscale (H,W) or RGB (H,W,3). Returns same format.
    """
    device = next(vae_model.parameters()).device
    is_rgb = image.ndim == 3

    if is_rgb:
        h, w = image.shape[:2]
        img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        h, w = image.shape
        img_u8 = np.clip(image, 0, 255).astype(np.uint8)
        img_u8 = np.stack([img_u8, img_u8, img_u8], axis=2)

    tensor = torch.from_numpy(img_u8).permute(2, 0, 1).unsqueeze(0).float()
    tensor = (tensor / 127.5) - 1.0
    tensor = tensor.to(device)

    with torch.no_grad():
        latent = vae_model.encode(tensor).latent_dist.sample()
        decoded = vae_model.decode(latent).sample

    decoded_np = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
    decoded_np = (decoded_np + 1.0) * 127.5
    decoded_np = np.clip(decoded_np, 0, 255)[:h, :w, :]

    if is_rgb:
        return decoded_np
    else:
        return (
            0.299 * decoded_np[:, :, 0] + 0.587 * decoded_np[:, :, 1] + 0.114 * decoded_np[:, :, 2]
        )


def _psnr(original, modified):
    mse = np.mean((original - modified) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(255.0**2 / mse)


def _ssim_simple(a, b, window=7):
    """Simplified SSIM (no dependency on skimage)."""
    from scipy.ndimage import uniform_filter

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_a = uniform_filter(a, window)
    mu_b = uniform_filter(b, window)
    sigma_a2 = uniform_filter(a**2, window) - mu_a**2
    sigma_b2 = uniform_filter(b**2, window) - mu_b**2
    sigma_ab = uniform_filter(a * b, window) - mu_a * mu_b

    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sigma_a2 + sigma_b2 + c2)

    return np.mean(num / den)


@pytest.fixture(scope="module")
def vae():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="module")
def author_keys():
    return generate_author_keys(seed=b"ghost-strength-sweep-32-bytes!!")


@pytest.fixture(scope="module")
def test_images():
    rng = np.random.default_rng(42)
    return _make_test_images(rng, n=10, size=(512, 512))


@pytest.fixture(scope="module")
def test_images_rgb():
    rng = np.random.default_rng(42)
    return _make_test_images(rng, n=10, size=(512, 512), rgb=True)


STRENGTH_VALUES = [1.5, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0, 100.0, 200.0]


class TestGhostStrengthSweep:
    """Sweep ghost_strength_multiplier to find the optimal value."""

    def test_strength_sweep(self, vae, author_keys, test_images):
        """Main sweep: measure quality and detection across strength values."""
        results = []

        for strength in STRENGTH_VALUES:
            config = SigilConfig(ghost_strength_multiplier=strength)
            embedder = SigilEmbedder(config=config)

            psnrs = []
            ssims = []
            corr_originals = []
            corr_after_vae = []
            detected_after_vae = 0

            for img in test_images:
                wm = embedder.embed(img, author_keys)

                # Quality metrics
                psnrs.append(_psnr(img, wm))
                ssims.append(_ssim_simple(img, wm))

                # Ghost on original
                ghost_orig = analyze_ghost_signature(wm, author_keys.public_key, config)
                corr_originals.append(ghost_orig.correlation)

                # Ghost after VAE
                vae_img = _vae_roundtrip(vae, wm)
                ghost_vae = analyze_ghost_signature(vae_img, author_keys.public_key, config)
                corr_after_vae.append(ghost_vae.correlation)
                if ghost_vae.ghost_detected:
                    detected_after_vae += 1

            row = {
                "strength": strength,
                "psnr": np.mean(psnrs),
                "ssim": np.mean(ssims),
                "corr_original": np.mean(corr_originals),
                "corr_vae": np.mean(corr_after_vae),
                "detection_rate": detected_after_vae / len(test_images),
            }
            results.append(row)

            print(
                f"  strength={strength:5.1f}  "
                f"PSNR={row['psnr']:.1f}  "
                f"SSIM={row['ssim']:.4f}  "
                f"corr_orig={row['corr_original']:.4f}  "
                f"corr_vae={row['corr_vae']:.4f}  "
                f"det_rate={row['detection_rate']:.0%}"
            )

        # Summary
        print("\n" + "=" * 70)
        print("GHOST STRENGTH SWEEP SUMMARY")
        print("=" * 70)

        # Find best strength: highest VAE detection rate with PSNR > 35
        viable = [r for r in results if r["psnr"] > 35]
        if viable:
            best = max(viable, key=lambda r: r["corr_vae"])
            print(f"\nBest viable (PSNR>35): strength={best['strength']}")
            print(f"  PSNR={best['psnr']:.1f}, SSIM={best['ssim']:.4f}")
            print(f"  VAE correlation={best['corr_vae']:.4f}")
            print(f"  VAE detection rate={best['detection_rate']:.0%}")
        else:
            print("\nNo strength value achieves PSNR > 35")
            best = max(results, key=lambda r: r["corr_vae"])
            print(f"Best overall: strength={best['strength']}")

        # Find the crossover point where detection > 50%
        for r in results:
            if r["detection_rate"] >= 0.5:
                print(f"\nFirst ≥50% detection: strength={r['strength']}")
                break

        # Recommend
        quality_ok = [r for r in results if r["psnr"] > 38 and r["ssim"] > 0.95]
        if quality_ok:
            rec = max(quality_ok, key=lambda r: r["corr_vae"])
            print(f"\nRECOMMENDED (quality-constrained): strength={rec['strength']}")
        else:
            print("\nNo strength achieves PSNR>38 AND SSIM>0.95 — relax constraints")

    def test_strength_scaling_linearity(self, vae, author_keys, test_images):
        """Check if ghost correlation scales linearly with strength."""
        img = test_images[0]
        correlations = []

        for strength in STRENGTH_VALUES:
            config = SigilConfig(ghost_strength_multiplier=strength)
            embedder = SigilEmbedder(config=config)
            wm = embedder.embed(img, author_keys)
            ghost = analyze_ghost_signature(wm, author_keys.public_key, config)
            correlations.append(ghost.correlation)

        # Check monotonicity (higher strength → higher correlation)
        monotonic = all(
            correlations[i] <= correlations[i + 1] for i in range(len(correlations) - 1)
        )
        print(f"Correlations: {[f'{c:.4f}' for c in correlations]}")
        print(f"Monotonically increasing: {monotonic}")

        if not monotonic:
            print(
                "WARNING: Correlation does not increase monotonically with strength. "
                "This may indicate interference between ghost signal and other layers."
            )

    def test_rgb_multichannel_sweep(self, vae, author_keys, test_images_rgb):
        """Sweep with RGB images to measure multi-channel ghost improvement."""
        print("\nRGB MULTI-CHANNEL GHOST SWEEP")
        print("=" * 70)

        for strength in [8.0, 12.0, 20.0]:
            config = SigilConfig(ghost_strength_multiplier=strength)
            embedder = SigilEmbedder(config=config)

            corr_originals = []
            corr_after_vae = []
            detected_after_vae = 0

            for img in test_images_rgb:
                wm = embedder.embed(img, author_keys)

                # Ghost on original (RGB input → multi-channel analysis)
                ghost_orig = analyze_ghost_signature(wm, author_keys.public_key, config)
                corr_originals.append(ghost_orig.correlation)

                # Ghost after VAE (returns RGB)
                vae_img = _vae_roundtrip(vae, wm)
                ghost_vae = analyze_ghost_signature(vae_img, author_keys.public_key, config)
                corr_after_vae.append(ghost_vae.correlation)
                if ghost_vae.ghost_detected:
                    detected_after_vae += 1

            print(
                f"  strength={strength:5.1f}  "
                f"corr_orig={np.mean(corr_originals):.4f}  "
                f"corr_vae={np.mean(corr_after_vae):.4f}  "
                f"det_rate={detected_after_vae / len(test_images_rgb):.0%}"
            )

    def test_vae_attenuation_ratio(self, vae, author_keys, test_images):
        """Measure how much the VAE attenuates ghost signal at each strength."""
        img = test_images[0]
        print("\nVAE attenuation by strength:")

        for strength in STRENGTH_VALUES:
            config = SigilConfig(ghost_strength_multiplier=strength)
            embedder = SigilEmbedder(config=config)
            wm = embedder.embed(img, author_keys)

            ghost_before = analyze_ghost_signature(wm, author_keys.public_key, config)
            vae_img = _vae_roundtrip(vae, wm)
            ghost_after = analyze_ghost_signature(vae_img, author_keys.public_key, config)

            ratio = ghost_after.correlation / max(abs(ghost_before.correlation), 1e-10)
            print(
                f"  strength={strength:5.1f}: "
                f"before={ghost_before.correlation:.4f} "
                f"after={ghost_after.correlation:.4f} "
                f"ratio={ratio:.3f}"
            )
