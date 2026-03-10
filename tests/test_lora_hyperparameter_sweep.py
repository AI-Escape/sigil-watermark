"""LoRA hyperparameter sweep for ghost signal propagation.

Tests whether the ghost propagation ceiling is LoRA-capacity-limited or fundamental.
Sweeps:
  - LoRA rank: 8, 16, 32, 64
  - Training steps: 500, 1000, 2000
  - Uses real images (standard test images) instead of synthetic

Requires: torch, diffusers, peft (install via `pip install sigil-watermark[gpu-tests]`)
"""

import io
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTokenizer
    from PIL import Image

    HAS_GPU_DEPS = True
except ImportError:
    HAS_GPU_DEPS = False

from sigil_watermark.config import SigilConfig
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.keygen import generate_author_keys
from sigil_watermark.ghost.spectral_analysis import (
    batch_analyze_ghost,
    analyze_ghost_signature,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_GPU_DEPS, reason="torch/diffusers/peft not installed"),
    pytest.mark.skipif(
        HAS_GPU_DEPS and not torch.cuda.is_available(),
        reason="CUDA not available",
    ),
]

TEST_IMAGES_DIR = Path(__file__).parent / "test_images"

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DTYPE = torch.float16 if HAS_GPU_DEPS else None


def _load_real_images():
    """Load real test images as grayscale float64 arrays, resized to 512x512."""
    images = []
    if not TEST_IMAGES_DIR.exists():
        return images
    for path in sorted(TEST_IMAGES_DIR.glob("*.png")):
        img = Image.open(path).convert("L").resize((512, 512), Image.LANCZOS)
        images.append(np.array(img, dtype=np.float64))
    return images


def _augment_images(images, rng, target_count=30):
    """Augment a small set of images to reach target_count via transforms."""
    augmented = list(images)
    while len(augmented) < target_count:
        src = images[len(augmented) % len(images)].copy()
        # Random brightness/contrast shift
        brightness = rng.uniform(-15, 15)
        contrast = rng.uniform(0.85, 1.15)
        src = np.clip((src - 128) * contrast + 128 + brightness, 0, 255)
        # Random horizontal flip
        if rng.random() > 0.5:
            src = src[:, ::-1].copy()
        augmented.append(src)
    return augmented[:target_count]


def _train_lora_and_generate(
    watermarked_images, lora_rank, lora_alpha, num_steps, num_generate=50,
):
    """Train LoRA on watermarked images, generate new images.

    Returns list of generated grayscale images.
    """
    device = "cuda"

    # Prepare training tensors
    training_tensors = []
    for img in watermarked_images:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        rgb = np.stack([img_u8, img_u8, img_u8], axis=2)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensor = (tensor / 127.5) - 1.0
        training_tensors.append(tensor)

    # Load model
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=DTYPE)
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=DTYPE
    )
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE
    )
    scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    vae = vae.to(device)
    unet = unet.to(device)
    text_encoder = text_encoder.to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    # Encode training images to latent space
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-2)

    with torch.no_grad():
        latents = []
        for t in training_tensors:
            lat = vae.encode(t.unsqueeze(0).to(device, dtype=DTYPE)).latent_dist.sample()
            lat = lat * vae.config.scaling_factor
            latents.append(lat)

    # Text embedding
    with torch.no_grad():
        tokens = tokenizer(
            ["a painting"],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        text_emb = text_encoder(tokens)[0]

    # Training loop
    for step in range(num_steps):
        idx = step % len(latents)
        lat = latents[idx]

        t = torch.randint(
            0, scheduler.config.num_train_timesteps, (1,), device=device
        )
        noise = torch.randn_like(lat)
        noisy_lat = scheduler.add_noise(lat, noise, t)

        pred = unet(noisy_lat, t, encoder_hidden_states=text_emb).sample
        loss = torch.nn.functional.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Free training resources
    del optimizer, latents, text_emb
    torch.cuda.empty_cache()

    # Generate
    unet.eval()
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    generated = []
    for i in range(num_generate):
        with torch.no_grad():
            result = pipe(
                f"a painting, artwork {i}",
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=torch.Generator(device).manual_seed(42 + i),
            )
        img_np = np.array(result.images[0].convert("L"), dtype=np.float64)
        generated.append(img_np)

    # Cleanup
    del pipe, unet, vae, text_encoder
    torch.cuda.empty_cache()

    return generated


@pytest.fixture(scope="module")
def config():
    return SigilConfig()


@pytest.fixture(scope="module")
def author_keys():
    return generate_author_keys(seed=b"lora-hyperparam-sweep-32bytes!!")


@pytest.fixture(scope="module")
def wrong_keys():
    return generate_author_keys(seed=b"wrong-key-lora-sweep-32-bytes!!")


@pytest.fixture(scope="module")
def watermarked_training_images(author_keys, config):
    """Load real images, augment to 30, watermark them."""
    real_images = _load_real_images()
    assert len(real_images) >= 3, (
        f"Need at least 3 test images in {TEST_IMAGES_DIR}, found {len(real_images)}"
    )
    rng = np.random.default_rng(42)
    augmented = _augment_images(real_images, rng, target_count=30)

    embedder = SigilEmbedder(config=config)
    watermarked = [embedder.embed(img, author_keys) for img in augmented]
    return watermarked


# Each config: (rank, alpha, steps)
SWEEP_CONFIGS = [
    (8, 16, 500),     # baseline (same as previous test)
    (8, 16, 1000),    # more steps
    (8, 16, 2000),    # even more steps
    (16, 32, 500),    # higher rank
    (16, 32, 1000),
    (32, 64, 500),    # high rank
    (32, 64, 1000),
    (64, 128, 500),   # very high rank
    (64, 128, 1000),
]


class TestLoRAHyperparameterSweep:
    """Sweep LoRA rank and training steps to find ghost propagation ceiling."""

    def test_sweep(self, watermarked_training_images, author_keys, wrong_keys, config):
        """Main sweep across LoRA configs."""
        results = []

        for rank, alpha, steps in SWEEP_CONFIGS:
            print(f"\n{'='*60}")
            print(f"Training: rank={rank}, alpha={alpha}, steps={steps}")
            print(f"{'='*60}")

            generated = _train_lora_and_generate(
                watermarked_training_images,
                lora_rank=rank,
                lora_alpha=alpha,
                num_steps=steps,
                num_generate=50,
            )

            # Batch analysis
            correct = batch_analyze_ghost(
                generated, author_keys.public_key, config
            )
            wrong = batch_analyze_ghost(
                generated, wrong_keys.public_key, config
            )

            # Single-image detection
            detected = 0
            correlations = []
            for img in generated[:20]:
                r = analyze_ghost_signature(img, author_keys.public_key, config)
                correlations.append(r.correlation)
                if r.ghost_detected:
                    detected += 1

            row = {
                "rank": rank,
                "alpha": alpha,
                "steps": steps,
                "batch_corr_correct": correct.correlation,
                "batch_corr_wrong": wrong.correlation,
                "batch_gap": correct.correlation - wrong.correlation,
                "batch_p": correct.p_value,
                "single_det_rate": detected / 20,
                "single_mean_corr": np.mean(correlations),
                "single_std_corr": np.std(correlations),
            }
            results.append(row)

            print(
                f"  Batch: correct={row['batch_corr_correct']:.6f} "
                f"wrong={row['batch_corr_wrong']:.6f} "
                f"gap={row['batch_gap']:.6f} p={row['batch_p']:.4f}"
            )
            print(
                f"  Single: det_rate={row['single_det_rate']:.0%} "
                f"mean_corr={row['single_mean_corr']:.6f} "
                f"std={row['single_std_corr']:.6f}"
            )

        # Summary table
        print(f"\n{'='*80}")
        print("LORA HYPERPARAMETER SWEEP SUMMARY")
        print(f"{'='*80}")
        print(
            f"{'Rank':>4} {'Alpha':>5} {'Steps':>5} | "
            f"{'Batch Corr':>10} {'Gap':>8} {'P-value':>8} | "
            f"{'Det Rate':>8} {'Mean Corr':>10} {'Std':>8}"
        )
        print("-" * 80)
        for r in results:
            print(
                f"{r['rank']:>4} {r['alpha']:>5} {r['steps']:>5} | "
                f"{r['batch_corr_correct']:>10.6f} {r['batch_gap']:>8.6f} "
                f"{r['batch_p']:>8.4f} | "
                f"{r['single_det_rate']:>7.0%} "
                f"{r['single_mean_corr']:>10.6f} {r['single_std_corr']:>8.6f}"
            )

        # Find best config
        best = max(results, key=lambda r: r["single_det_rate"])
        print(f"\nBest config: rank={best['rank']}, steps={best['steps']}")
        print(f"  Detection rate: {best['single_det_rate']:.0%}")
        print(f"  Batch correlation: {best['batch_corr_correct']:.6f}")

        # Check if ceiling holds
        baseline = results[0]  # rank=8, steps=500
        best_det = best["single_det_rate"]
        baseline_det = baseline["single_det_rate"]

        if best_det >= 0.8:
            print(
                "\nCONCLUSION: ≥80% single-image detection achieved! "
                "Ghost propagation works with sufficient LoRA capacity."
            )
        elif best_det > baseline_det + 0.15:
            print(
                f"\nCONCLUSION: Improvement from {baseline_det:.0%} to {best_det:.0%}. "
                "Higher LoRA capacity helps but may not be sufficient alone."
            )
        else:
            print(
                f"\nCONCLUSION: Ceiling holds ({baseline_det:.0%} → {best_det:.0%}). "
                "Ghost propagation mechanism itself needs to change."
            )
