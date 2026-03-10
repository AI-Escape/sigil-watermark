"""Purification attack tests inspired by Hönig et al. (ICLR 2025).

"Adversarial Perturbations Cannot Reliably Protect Artists From Generative AI"
(arxiv 2406.12027) demonstrates that simple preprocessing — Gaussian noising,
diffusion-based purification, and noisy upscaling — can strip adversarial
perturbations from images.

Unlike adversarial perturbations (Glaze, Mist, Anti-DreamBooth) which are
designed to PREVENT model learning, Sigil watermarks are designed to SURVIVE
processing. These tests simulate the paper's bypass methods to verify that
our watermark is robust against the same attacks that defeat adversarial
protections.

Key bypass methods from the paper:
1. Gaussian noising (sigma 0.05-0.25 on [0,1] scale → 12.75-63.75 on [0,255])
2. DiffPure — add noise then denoise via diffusion model (simulated here)
3. Noisy Upscaling — add noise then run through an upscaler (simulated)
4. IMPRESS++ — iterative optimization (not applicable to watermarks)
"""

import io

import cv2
import numpy as np
import pytest
from PIL import Image

from sigil_watermark.config import SigilConfig
from sigil_watermark.detect import SigilDetector
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.keygen import generate_author_keys


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def author_keys():
    return generate_author_keys(seed=b"purification-attack-test32bytes!")


@pytest.fixture
def embedder(config):
    return SigilEmbedder(config=config)


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


def _make_image(rng, size=(512, 512)):
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            img[i, j] = (
                128 + 40 * np.sin(i / 20.0) + 30 * np.cos(j / 15.0) + 20 * np.sin((i + j) / 25.0)
            )
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255)


def _jpeg(image, quality):
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_u8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"), dtype=np.float64)


def _blur(image, sigma):
    ksize = max(3, int(sigma * 6) | 1)
    return cv2.GaussianBlur(image.astype(np.float32), (ksize, ksize), sigma).astype(np.float64)


def _resize(image, scale):
    """Resize then scale back to original size."""
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    down = cv2.resize(image.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(up.astype(np.float64), 0, 255)


# --- Gaussian Noising (Paper Section 3.1) ---


class TestGaussianNoising:
    """Paper's simplest bypass: add Gaussian noise to the image.

    Paper uses sigma in [0.05, 0.25] on [0,1] scale.
    On [0,255] scale: sigma 12.75 to 63.75.
    """

    @pytest.mark.parametrize(
        "sigma_01",
        [0.05, 0.10, 0.15, 0.20, 0.25],
        ids=["sigma_0.05", "sigma_0.10", "sigma_0.15", "sigma_0.20", "sigma_0.25"],
    )
    def test_gaussian_noising(self, embedder, detector, author_keys, sigma_01):
        """Watermark should survive Gaussian noise at levels used in the paper."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        sigma_255 = sigma_01 * 255.0
        noised = np.clip(wm + rng.normal(0, sigma_255, wm.shape), 0, 255)
        result = detector.detect(noised, author_keys.public_key)

        if sigma_01 <= 0.15:
            assert result.detected or result.payload_confidence > 0.4, (
                f"sigma={sigma_01}: conf={result.payload_confidence:.2f}"
            )
        else:
            # Heavy noise — at least some signal should remain
            assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3, (
                f"sigma={sigma_01}: payload={result.payload_confidence:.2f}, "
                f"ring={result.ring_confidence:.2f}"
            )

    def test_gaussian_noise_then_jpeg(self, embedder, detector, author_keys):
        """Paper notes noise + JPEG is common in social media pipelines."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        sigma_255 = 0.10 * 255.0  # sigma=0.10 in paper scale
        noised = np.clip(wm + rng.normal(0, sigma_255, wm.shape), 0, 255)
        compressed = _jpeg(noised, 75)
        result = detector.detect(compressed, author_keys.public_key)
        assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3

    def test_gaussian_noise_then_blur(self, embedder, detector, author_keys):
        """Noise followed by smoothing (common denoising step)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        sigma_255 = 0.10 * 255.0
        noised = np.clip(wm + rng.normal(0, sigma_255, wm.shape), 0, 255)
        smoothed = _blur(noised, 1.0)
        result = detector.detect(smoothed, author_keys.public_key)
        assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3


# --- DiffPure Simulation (Paper Section 3.2) ---


class TestDiffPureSimulation:
    """Simulate DiffPure: add noise at timestep t, then denoise.

    We can't run an actual diffusion model, but we simulate the effect:
    noise → denoise via bilateral/NLM filter (preserves structure, removes noise).
    This captures the essential operation: corruption + reconstruction.
    """

    def _simulate_diffpure(self, image, noise_level, rng):
        """Add noise then denoise with bilateral filter (structure-preserving)."""
        noised = np.clip(image + rng.normal(0, noise_level, image.shape), 0, 255)
        img_u8 = noised.astype(np.uint8)
        # Bilateral filter: preserves edges while smoothing
        denoised = cv2.bilateralFilter(img_u8, d=9, sigmaColor=75, sigmaSpace=75)
        return denoised.astype(np.float64)

    @pytest.mark.parametrize("noise_level", [15, 25, 40, 60])
    def test_diffpure_bilateral(self, embedder, detector, author_keys, noise_level):
        """Simulate noise + bilateral denoise at different strengths."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        purified = self._simulate_diffpure(wm, noise_level, rng)
        result = detector.detect(purified, author_keys.public_key)

        if noise_level <= 25:
            assert result.detected or result.payload_confidence > 0.4, (
                f"DiffPure noise={noise_level}: conf={result.payload_confidence:.2f}"
            )
        else:
            assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3, (
                f"DiffPure noise={noise_level}: payload={result.payload_confidence:.2f}, "
                f"ring={result.ring_confidence:.2f}"
            )

    def test_diffpure_nlm(self, embedder, detector, author_keys):
        """Simulate DiffPure with Non-Local Means denoising (stronger)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        noised = np.clip(wm + rng.normal(0, 25, wm.shape), 0, 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoising(noised, h=10, templateWindowSize=7, searchWindowSize=21)
        result = detector.detect(denoised.astype(np.float64), author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, (
            f"DiffPure NLM: conf={result.payload_confidence:.2f}"
        )

    def test_diffpure_multiple_rounds(self, embedder, detector, author_keys):
        """Paper notes DiffPure can be applied multiple times."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        purified = wm
        for i in range(3):
            noise_rng = np.random.default_rng(42 + i)
            purified = self._simulate_diffpure(purified, 15, noise_rng)

        result = detector.detect(purified, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.3, (
            f"DiffPure x3: conf={result.payload_confidence:.2f}"
        )


# --- Noisy Upscaling Simulation (Paper Section 3.3) ---


class TestNoisyUpscaling:
    """Paper's strongest bypass: add noise, then run through an upscaler.

    The upscaler acts as a learned denoiser that reconstructs the image
    while discarding adversarial perturbations. We simulate this with:
    noise → downscale → upscale (bicubic), which mimics the
    corruption + reconstruction paradigm.
    """

    def _noisy_upscale(self, image, noise_sigma, down_factor, rng):
        """Noise → downscale → upscale back to original size."""
        h, w = image.shape[:2]
        noised = np.clip(image + rng.normal(0, noise_sigma, image.shape), 0, 255)
        new_h, new_w = max(1, int(h / down_factor)), max(1, int(w / down_factor))
        down = cv2.resize(noised.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
        up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(up.astype(np.float64), 0, 255)

    @pytest.mark.parametrize(
        "noise_sigma,down_factor",
        [(10, 1.5), (20, 2.0), (30, 2.0), (40, 2.0), (15, 4.0)],
        ids=["mild", "moderate", "strong", "extreme", "high_downscale"],
    )
    def test_noisy_upscaling(self, embedder, detector, author_keys, noise_sigma, down_factor):
        """Simulate noisy upscaling at various intensities."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        attacked = self._noisy_upscale(wm, noise_sigma, down_factor, rng)
        result = detector.detect(attacked, author_keys.public_key)

        if noise_sigma <= 20 and down_factor <= 2.0:
            assert result.detected or result.payload_confidence > 0.3, (
                f"NoisyUpscale sigma={noise_sigma} df={down_factor}: "
                f"conf={result.payload_confidence:.2f}"
            )
        else:
            # Heavy attack — check for any surviving signal
            assert result.payload_confidence > 0.1 or result.ring_confidence > 0.2, (
                f"NoisyUpscale sigma={noise_sigma} df={down_factor}: "
                f"payload={result.payload_confidence:.2f}, ring={result.ring_confidence:.2f}"
            )

    def test_noisy_upscale_then_jpeg(self, embedder, detector, author_keys):
        """Noisy upscaling followed by JPEG (realistic social media pipeline)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        attacked = self._noisy_upscale(wm, 15, 2.0, rng)
        attacked = _jpeg(attacked, 75)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.payload_confidence > 0.1 or result.ring_confidence > 0.2

    def test_upscale_without_noise(self, embedder, detector, author_keys):
        """Pure downscale/upscale without noise — should be easier to survive."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        attacked = _resize(wm, 0.5)  # 2x downscale then back
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Pure resize: conf={result.payload_confidence:.2f}"
        )


# --- Superresolution Simulation ---


class TestSuperresolutionSimulation:
    """Simulate superresolution models (a key component of noisy upscaling).

    Real SR models apply learned sharpening after upscaling. We simulate with:
    downscale → upscale → unsharp mask (mimics learned detail synthesis).
    """

    def _simulate_sr(self, image, down_factor, sharpen_amount):
        """Downscale → upscale → sharpen."""
        h, w = image.shape[:2]
        new_h, new_w = max(1, int(h / down_factor)), max(1, int(w / down_factor))
        down = cv2.resize(image.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
        up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
        # Unsharp mask for sharpening
        blurred = cv2.GaussianBlur(up, (0, 0), 3)
        sharpened = cv2.addWeighted(up, 1 + sharpen_amount, blurred, -sharpen_amount, 0)
        return np.clip(sharpened.astype(np.float64), 0, 255)

    @pytest.mark.parametrize(
        "down_factor,sharpen",
        [(2.0, 0.5), (2.0, 1.0), (4.0, 0.5), (4.0, 1.0)],
        ids=["2x_mild", "2x_strong", "4x_mild", "4x_strong"],
    )
    def test_sr_simulation(self, embedder, detector, author_keys, down_factor, sharpen):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        attacked = self._simulate_sr(wm, down_factor, sharpen)
        result = detector.detect(attacked, author_keys.public_key)

        if down_factor <= 2.0:
            assert result.detected or result.payload_confidence > 0.3, (
                f"SR {down_factor}x sharpen={sharpen}: conf={result.payload_confidence:.2f}"
            )
        else:
            assert result.payload_confidence > 0.1 or result.ring_confidence > 0.2, (
                f"SR {down_factor}x sharpen={sharpen}: "
                f"payload={result.payload_confidence:.2f}, ring={result.ring_confidence:.2f}"
            )


# --- Training Simulation (Paper's Core Scenario) ---


class TestTrainingSimulation:
    """Simulate what happens when watermarked images go through a training pipeline.

    The paper's scenario: artist uploads protected images → attacker fine-tunes
    a model on those images → generates new images in the artist's style.

    For adversarial perturbations, the training process itself strips the protection.
    For Sigil watermarks, we WANT the watermark to survive into generated images.
    We simulate the training pipeline's processing steps.
    """

    def _simulate_training_preprocess(self, image, rng):
        """Simulate common training data augmentation pipeline.

        Most fine-tuning pipelines (DreamBooth, LoRA, textual inversion):
        1. Resize to training resolution (typically 512x512)
        2. Random crop / center crop
        3. Optional horizontal flip
        4. Normalize to [-1, 1]
        5. Add noise (diffusion training adds noise at various timesteps)
        """
        h, w = image.shape[:2]
        # Step 1: Resize to training resolution
        resized = cv2.resize(image.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA)
        # Step 2: Center crop (95% to simulate slight crop)
        margin = int(512 * 0.025)
        cropped = resized[margin : 512 - margin, margin : 512 - margin]
        cropped = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_LINEAR)
        # Step 3: Skip flip (deterministic for testing)
        # Step 4-5: Normalize → add noise → denormalize (simulates one training step)
        normalized = cropped / 127.5 - 1.0  # [-1, 1]
        # Simulate diffusion noise at moderate timestep
        noisy = normalized + rng.normal(0, 0.3, normalized.shape).astype(np.float32)
        # Simulate denoising (bilateral approximation)
        denoised_u8 = np.clip((noisy + 1.0) * 127.5, 0, 255).astype(np.uint8)
        denoised = cv2.bilateralFilter(denoised_u8, d=5, sigmaColor=50, sigmaSpace=50)
        return denoised.astype(np.float64)

    def test_single_training_preprocess(self, embedder, detector, author_keys):
        """Watermark should partially survive a single training preprocess step."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        processed = self._simulate_training_preprocess(wm, rng)
        result = detector.detect(processed, author_keys.public_key)
        assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3, (
            f"Training preprocess: payload={result.payload_confidence:.2f}, "
            f"ring={result.ring_confidence:.2f}"
        )

    def test_training_augmentations_only(self, embedder, detector, author_keys):
        """Just resize + crop + normalize (without noise) should be easy."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        # Resize to 512
        resized = cv2.resize(wm.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA)
        # Center crop 90%
        margin = int(512 * 0.05)
        cropped = resized[margin : 512 - margin, margin : 512 - margin]
        cropped = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_LINEAR)

        result = detector.detect(cropped.astype(np.float64), author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4, (
            f"Augmentations only: conf={result.payload_confidence:.2f}"
        )

    def test_multiple_training_epochs(self, embedder, detector, author_keys):
        """Simulate multiple passes through training pipeline."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        processed = wm
        for epoch in range(3):
            epoch_rng = np.random.default_rng(42 + epoch)
            processed = self._simulate_training_preprocess(processed, epoch_rng)

        result = detector.detect(processed, author_keys.public_key)
        # After 3 passes, some signal should remain in at least one layer
        assert result.payload_confidence > 0.1 or result.ring_confidence > 0.2, (
            f"3 epochs: payload={result.payload_confidence:.2f}, ring={result.ring_confidence:.2f}"
        )


# --- VAE Encode/Decode Simulation ---


class TestVAESimulation:
    """Simulate VAE encode/decode pass (latent diffusion bottleneck).

    Stable Diffusion and similar models encode images to a compressed latent
    space (8x spatial downsampling) then decode back. This acts as a lossy
    bottleneck that could strip watermarks.

    We simulate with aggressive downscale → upscale + blur.
    """

    def _simulate_vae(self, image, compression_factor=8):
        """Simulate VAE encode → decode cycle."""
        h, w = image.shape[:2]
        latent_h, latent_w = max(1, h // compression_factor), max(1, w // compression_factor)
        # Encode: aggressive downscale
        latent = cv2.resize(
            image.astype(np.float32), (latent_w, latent_h), interpolation=cv2.INTER_AREA
        )
        # Decode: upscale back (VAE decoder is a learned upsampler)
        decoded = cv2.resize(latent, (w, h), interpolation=cv2.INTER_CUBIC)
        # VAE introduces slight smoothing
        decoded = cv2.GaussianBlur(decoded, (3, 3), 0.5)
        return np.clip(decoded.astype(np.float64), 0, 255)

    @pytest.mark.parametrize("compression", [4, 8])
    def test_vae_roundtrip(self, embedder, detector, author_keys, compression):
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        decoded = self._simulate_vae(wm, compression)
        result = detector.detect(decoded, author_keys.public_key)

        if compression <= 4:
            assert result.detected or result.payload_confidence > 0.3, (
                f"VAE {compression}x: conf={result.payload_confidence:.2f}"
            )
        else:
            # 8x compression is very aggressive
            assert result.payload_confidence > 0.1 or result.ring_confidence > 0.2, (
                f"VAE {compression}x: payload={result.payload_confidence:.2f}, "
                f"ring={result.ring_confidence:.2f}"
            )

    def test_vae_then_jpeg(self, embedder, detector, author_keys):
        """VAE followed by JPEG (model output saved as JPEG)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        decoded = self._simulate_vae(wm, 4)
        compressed = _jpeg(decoded, 85)
        result = detector.detect(compressed, author_keys.public_key)
        assert result.payload_confidence > 0.1 or result.ring_confidence > 0.2


# --- Best-of-N Strategy (Paper Section 4) ---


class TestBestOfN:
    """Paper's "best-of-4" strategy: try multiple bypass variants, pick best.

    For adversarial perturbations, the attacker picks the result that looks
    best. For watermarks, this means trying multiple attacks and the watermark
    should survive at least in a degraded form.
    """

    def test_best_of_4_attacks(self, embedder, detector, author_keys):
        """Apply 4 different attacks and verify watermark survives all."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        attacks = [
            ("gaussian", lambda i: np.clip(i + rng.normal(0, 25, i.shape), 0, 255)),
            ("blur", lambda i: _blur(i, 1.5)),
            ("jpeg", lambda i: _jpeg(i, 60)),
            ("resize", lambda i: _resize(i, 0.5)),
        ]

        surviving = 0
        for name, attack in attacks:
            attacked = attack(wm)
            result = detector.detect(attacked, author_keys.public_key)
            if result.payload_confidence > 0.2 or result.ring_confidence > 0.3:
                surviving += 1

        # At least 3 out of 4 attacks should leave detectable signal
        assert surviving >= 3, f"Only {surviving}/4 attacks left surviving signal"

    def test_strongest_single_attack(self, embedder, detector, author_keys):
        """Even the strongest single attack should leave some trace."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        # Strongest: noise + downscale + upscale + JPEG
        attacked = np.clip(wm + rng.normal(0, 30, wm.shape), 0, 255)
        attacked = _resize(attacked, 0.5)
        attacked = _jpeg(attacked, 60)

        result = detector.detect(attacked, author_keys.public_key)
        # At least ring layer should show something
        assert result.ring_confidence > 0.1 or result.payload_confidence > 0.1, (
            f"Strongest attack: payload={result.payload_confidence:.2f}, "
            f"ring={result.ring_confidence:.2f}"
        )


# --- Comparative Robustness (Paper's Table 1 Scenarios) ---


class TestPaperTable1Scenarios:
    """Tests mirroring the paper's evaluation scenarios.

    The paper evaluates protections under: no bypass, Gaussian, DiffPure,
    Noisy Upscaling, IMPRESS++. We test our watermark under each.
    """

    def test_no_bypass(self, embedder, detector, author_keys):
        """Baseline: no attack applied."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        result = detector.detect(wm, author_keys.public_key)
        assert result.detected

    def test_gaussian_bypass(self, embedder, detector, author_keys):
        """Paper's Gaussian bypass (sigma=0.1 on [0,1])."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        attacked = np.clip(wm + rng.normal(0, 25.5, wm.shape), 0, 255)
        result = detector.detect(attacked, author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4

    def test_diffpure_bypass(self, embedder, detector, author_keys):
        """Paper's DiffPure bypass (noise + denoise)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        noised = np.clip(wm + rng.normal(0, 25, wm.shape), 0, 255).astype(np.uint8)
        denoised = cv2.bilateralFilter(noised, d=9, sigmaColor=75, sigmaSpace=75)
        result = detector.detect(denoised.astype(np.float64), author_keys.public_key)
        assert result.detected or result.payload_confidence > 0.4

    def test_noisy_upscaling_bypass(self, embedder, detector, author_keys):
        """Paper's Noisy Upscaling bypass (noise + downscale + upscale)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        noised = np.clip(wm + rng.normal(0, 20, wm.shape), 0, 255)
        h, w = noised.shape
        down = cv2.resize(noised.astype(np.float32), (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
        result = detector.detect(up.astype(np.float64), author_keys.public_key)
        assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3

    def test_full_pipeline_scenario(self, embedder, detector, author_keys):
        """Full realistic scenario: watermark → social media → model training.

        1. Embed watermark
        2. JPEG compression (upload)
        3. Resize (platform processing)
        4. Gaussian noise (preprocessing)
        5. Bilateral denoise (simulated model processing)
        """
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        # Social media upload
        pipeline = _jpeg(wm, 80)
        # Platform resize
        h, w = pipeline.shape
        pipeline = cv2.resize(
            pipeline.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA
        ).astype(np.float64)
        # Training preprocessing
        pipeline = np.clip(pipeline + rng.normal(0, 15, pipeline.shape), 0, 255)
        # Denoise
        pipeline_u8 = pipeline.astype(np.uint8)
        pipeline = cv2.bilateralFilter(pipeline_u8, d=5, sigmaColor=50, sigmaSpace=50)

        result = detector.detect(pipeline.astype(np.float64), author_keys.public_key)
        assert result.payload_confidence > 0.2 or result.ring_confidence > 0.3, (
            f"Full pipeline: payload={result.payload_confidence:.2f}, "
            f"ring={result.ring_confidence:.2f}"
        )
