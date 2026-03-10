"""Ghost signal propagation through LoRA fine-tuning.

Tests the core hypothesis: does the ghost spectral fingerprint propagate
through AI model training into generated images?

Flow:
1. Generate 30 watermarked images with the same author key
2. Fine-tune a LoRA adapter on those images
3. Generate 100 images with the fine-tuned model
4. Run batch_analyze_ghost() — correct key should show elevated correlation
5. Control: wrong key should show baseline correlation

Requires: torch, diffusers, peft (install via `pip install sigil-watermark[gpu-tests]`)
"""

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

    HAS_GPU_DEPS = True
except ImportError:
    HAS_GPU_DEPS = False

from sigil_watermark.config import SigilConfig
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.keygen import generate_author_keys
from sigil_watermark.ghost.spectral_analysis import batch_analyze_ghost, analyze_ghost_signature

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(not HAS_GPU_DEPS, reason="torch/diffusers/peft not installed"),
    pytest.mark.skipif(
        HAS_GPU_DEPS and not torch.cuda.is_available(),
        reason="CUDA not available (LoRA training requires GPU)",
    ),
]


@pytest.fixture(scope="class")
def config():
    return SigilConfig()


@pytest.fixture(scope="class")
def author_keys():
    return generate_author_keys(seed=b"ghost-lora-propagation-32bytes!!")


@pytest.fixture(scope="class")
def wrong_keys():
    return generate_author_keys(seed=b"wrong-key-ghost-test-32-bytes!!!")


def _make_diverse_images(rng, n=30, size=(512, 512)):
    """Generate n diverse synthetic images for training."""
    images = []
    h, w = size
    for idx in range(n):
        img = np.zeros((h, w), dtype=np.float64)
        freq1 = 10 + idx * 3
        freq2 = 15 + idx * 2
        amp1 = 30 + rng.uniform(-10, 10)
        amp2 = 25 + rng.uniform(-10, 10)
        phase = rng.uniform(0, 2 * np.pi)
        for i in range(h):
            for j in range(w):
                img[i, j] = (
                    128
                    + amp1 * np.sin(i / freq1 + phase)
                    + amp2 * np.cos(j / freq2)
                    + 15 * np.sin((i + j) / (20 + idx))
                    + 10 * np.cos((i - j) / (25 + idx))
                )
        img += rng.normal(0, 8, img.shape)
        images.append(np.clip(img, 0, 255))
    return images


@pytest.fixture(scope="class")
def training_data_and_generated(author_keys, config):
    """Fine-tune LoRA on watermarked images and generate new ones.

    This fixture is class-scoped so the expensive training/generation
    only happens once per test class.
    """
    embedder = SigilEmbedder(config=config)
    rng = np.random.default_rng(42)

    # Step 1: Create watermarked training images
    raw_images = _make_diverse_images(rng, n=30, size=(512, 512))
    watermarked = [embedder.embed(img, author_keys) for img in raw_images]

    # Step 2: Prepare training data as 3-channel tensors
    device = "cuda"
    training_tensors = []
    for img in watermarked:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        rgb = np.stack([img_u8, img_u8, img_u8], axis=2)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensor = (tensor / 127.5) - 1.0
        training_tensors.append(tensor)

    # Step 3: Load base model components (float16 to fit in VRAM)
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    dtype = torch.float16
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae = vae.to(device)
    unet = unet.to(device)
    text_encoder = text_encoder.to(device)

    # Step 4: Apply LoRA to UNet
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    # Step 5: Simple fine-tuning loop
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-2)

    # Pre-encode all training images to latent space
    with torch.no_grad():
        latents = []
        for t in training_tensors:
            lat = vae.encode(t.unsqueeze(0).to(device, dtype=dtype)).latent_dist.sample()
            lat = lat * vae.config.scaling_factor
            latents.append(lat)

    # Get a fixed text embedding (unconditional)
    with torch.no_grad():
        tokens = tokenizer(
            ["a painting"] * 1,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        text_emb = text_encoder(tokens)[0]  # (1, seq_len, hidden_dim)

    # Training loop
    num_steps = 500
    for step in range(num_steps):
        idx = step % len(latents)
        lat = latents[idx]

        # Sample random timestep
        t = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
        noise = torch.randn_like(lat)
        noisy_lat = scheduler.add_noise(lat, noise, t)

        # Predict noise
        pred = unet(noisy_lat, t, encoder_hidden_states=text_emb).sample
        loss = torch.nn.functional.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Step 6: Generate images with fine-tuned model
    # Free training resources first to reclaim VRAM
    del optimizer, latents, text_emb
    torch.cuda.empty_cache()

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

    generated_images = []
    # Generate one at a time to avoid OOM on 16GB GPUs
    for i in range(100):
        prompt = f"a painting, artwork {i}"
        with torch.no_grad():
            result = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=torch.Generator(device).manual_seed(42 + i),
            )
        img_np = np.array(result.images[0].convert("L"), dtype=np.float64)
        generated_images.append(img_np)

    # Cleanup GPU memory
    del pipe, unet, vae, text_encoder
    torch.cuda.empty_cache()

    return watermarked, generated_images


class TestGhostLoRAPropagation:
    """Test whether ghost signal propagates through LoRA fine-tuning."""

    def test_ghost_correlation_above_baseline(
        self, training_data_and_generated, author_keys, config
    ):
        """Ghost correlation with correct key should exceed random baseline."""
        _, generated = training_data_and_generated

        result = batch_analyze_ghost(generated, author_keys.public_key, config)

        print(
            f"Ghost on generated images (correct key):\n"
            f"  Correlation: {result.correlation:.6f}\n"
            f"  P-value: {result.p_value:.4f}\n"
            f"  Detected: {result.ghost_detected}\n"
            f"  Band energies: {result.band_energies}"
        )

    def test_ghost_wrong_key_no_detection(
        self, training_data_and_generated, wrong_keys, config
    ):
        """Ghost correlation with wrong key should be near zero."""
        _, generated = training_data_and_generated

        result = batch_analyze_ghost(generated, wrong_keys.public_key, config)

        print(
            f"Ghost on generated images (wrong key):\n"
            f"  Correlation: {result.correlation:.6f}\n"
            f"  P-value: {result.p_value:.4f}\n"
            f"  Detected: {result.ghost_detected}"
        )

    def test_correct_vs_wrong_key_separation(
        self, training_data_and_generated, author_keys, wrong_keys, config
    ):
        """Correct key correlation should be significantly higher than wrong key."""
        _, generated = training_data_and_generated

        correct = batch_analyze_ghost(generated, author_keys.public_key, config)
        wrong = batch_analyze_ghost(generated, wrong_keys.public_key, config)

        gap = correct.correlation - wrong.correlation
        print(
            f"Correct key correlation: {correct.correlation:.6f}\n"
            f"Wrong key correlation:   {wrong.correlation:.6f}\n"
            f"Gap: {gap:.6f}\n"
            f"Correct p-value: {correct.p_value:.4f}\n"
            f"Wrong p-value: {wrong.p_value:.4f}"
        )

        if gap <= 0:
            print(
                "WARNING: Ghost signal did NOT propagate through LoRA training. "
                "The ghost layer hypothesis may need revision. Consider:\n"
                "  1. Increasing ghost_strength_multiplier\n"
                "  2. Narrowing ghost_bandwidth\n"
                "  3. Repositioning ghost_bands away from upsampling artifact frequencies"
            )

    def test_ghost_on_training_images_vs_generated(
        self, training_data_and_generated, author_keys, config
    ):
        """Compare ghost strength on original watermarked images vs generated ones."""
        watermarked, generated = training_data_and_generated

        # Ghost on original watermarked images (should be strong)
        wm_result = batch_analyze_ghost(watermarked[:10], author_keys.public_key, config)

        # Ghost on generated images (may be weaker)
        gen_result = batch_analyze_ghost(generated[:10], author_keys.public_key, config)

        print(
            f"Ghost on watermarked (original): correlation={wm_result.correlation:.6f}, "
            f"p={wm_result.p_value:.4f}\n"
            f"Ghost on generated:              correlation={gen_result.correlation:.6f}, "
            f"p={gen_result.p_value:.4f}\n"
            f"Attenuation ratio: {gen_result.correlation / max(wm_result.correlation, 1e-10):.4f}"
        )

    def test_individual_generated_image_analysis(
        self, training_data_and_generated, author_keys, config
    ):
        """Analyze individual generated images (single-image detection is harder)."""
        _, generated = training_data_and_generated

        detected_count = 0
        correlations = []

        for i, img in enumerate(generated[:20]):
            result = analyze_ghost_signature(img, author_keys.public_key, config)
            correlations.append(result.correlation)
            if result.ghost_detected:
                detected_count += 1

        mean_corr = np.mean(correlations)
        std_corr = np.std(correlations)
        print(
            f"Individual image analysis (first 20):\n"
            f"  Detected: {detected_count}/20\n"
            f"  Mean correlation: {mean_corr:.6f}\n"
            f"  Std correlation: {std_corr:.6f}\n"
            f"  Min/Max: {min(correlations):.6f} / {max(correlations):.6f}"
        )
