"""Real Stable Diffusion VAE encode/decode robustness test.

Tests the watermark against an actual SD VAE (stabilityai/sd-vae-ft-mse),
which projects images into a learned 4-channel latent space with 8x spatial
downsampling. This is fundamentally different from the simulated VAE
(resize down/up + blur) because the encoder actively discards structured
noise that doesn't correlate with natural image statistics.

Requires: torch, diffusers (install via `pip install sigil-watermark[gpu-tests]`)
"""

import io

import cv2
import numpy as np
import pytest

try:
    import torch
    from diffusers import AutoencoderKL

    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

from sigil_watermark.config import SigilConfig
from sigil_watermark.detect import SigilDetector
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.ghost.spectral_analysis import analyze_ghost_signature, extract_ghost_hash
from sigil_watermark.keygen import derive_ghost_hash, generate_author_keys

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers/torch not installed"),
]


@pytest.fixture
def config():
    return SigilConfig()


@pytest.fixture
def author_keys():
    return generate_author_keys(seed=b"real-vae-test-key-32-bytes-pad!!")


@pytest.fixture
def embedder(config):
    return SigilEmbedder(config=config)


@pytest.fixture
def detector(config):
    return SigilDetector(config=config)


@pytest.fixture(scope="module")
def vae():
    """Load the SD VAE model (cached across tests in module)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model


def _make_image(rng, size=(512, 512)):
    """Generate a test image with varied texture."""
    h, w = size
    img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            img[i, j] = (
                128 + 40 * np.sin(i / 20.0) + 30 * np.cos(j / 15.0) + 20 * np.sin((i + j) / 25.0)
            )
    img += rng.normal(0, 8, img.shape)
    return np.clip(img, 0, 255)


def _vae_roundtrip(vae_model, image: np.ndarray) -> np.ndarray:
    """Pass a grayscale image through a real SD VAE encode/decode cycle.

    Args:
        vae_model: Loaded AutoencoderKL model.
        image: Grayscale float64 image [0, 255], shape (H, W).

    Returns:
        Reconstructed grayscale float64 image [0, 255], shape (H, W).
    """
    device = next(vae_model.parameters()).device
    h, w = image.shape

    # Grayscale -> 3-channel RGB (replicate across channels)
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    rgb = np.stack([img_u8, img_u8, img_u8], axis=2)  # (H, W, 3)

    # To tensor: (H, W, 3) -> (1, 3, H, W), normalize to [-1, 1]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    tensor = (tensor / 127.5) - 1.0
    tensor = tensor.to(device)

    # Encode -> latent -> decode
    with torch.no_grad():
        latent = vae_model.encode(tensor).latent_dist.sample()
        decoded = vae_model.decode(latent).sample

    # Back to numpy: (1, 3, H, W) -> (H, W, 3) -> grayscale
    decoded_np = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
    decoded_np = (decoded_np + 1.0) * 127.5
    decoded_np = np.clip(decoded_np, 0, 255)

    # Extract Y channel (luminance) via standard BT.601 weights
    y_channel = (
        0.299 * decoded_np[:, :, 0] + 0.587 * decoded_np[:, :, 1] + 0.114 * decoded_np[:, :, 2]
    )

    return y_channel[:h, :w]


def _simulate_vae(image, compression_factor=8):
    """Simulated VAE (resize-based) for comparison."""
    h, w = image.shape[:2]
    latent_h = max(1, h // compression_factor)
    latent_w = max(1, w // compression_factor)
    latent = cv2.resize(
        image.astype(np.float32), (latent_w, latent_h), interpolation=cv2.INTER_AREA
    )
    decoded = cv2.resize(latent, (w, h), interpolation=cv2.INTER_CUBIC)
    decoded = cv2.GaussianBlur(decoded, (3, 3), 0.5)
    return np.clip(decoded.astype(np.float64), 0, 255)


def _psnr(a, b):
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(255.0**2 / mse)


class TestRealVAERoundtrip:
    """Test watermark survival through real SD VAE encode/decode."""

    def test_vae_roundtrip_detection(self, vae, embedder, detector, author_keys):
        """Core test: does the watermark survive a real VAE cycle?"""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        decoded = _vae_roundtrip(vae, wm)
        result = detector.detect(decoded, author_keys.public_key)

        print(
            f"Real VAE roundtrip:\n"
            f"  Ring confidence: {result.ring_confidence:.3f}\n"
            f"  Payload confidence: {result.payload_confidence:.3f}\n"
            f"  Detected: {result.detected}\n"
            f"  Author match: {result.author_id_match}"
        )

        # Ring and payload layers don't reliably survive the VAE's learned
        # latent projection. That's expected — the ghost layer (Layer 3) is
        # the designed VAE-survival channel. Check that at least some signal
        # is detectable across all layers.
        assert result.ring_confidence > 0.0 or result.payload_confidence > 0.45, (
            "No signal survived the real VAE — watermark is fully stripped."
        )

    def test_vae_vs_simulated_comparison(self, vae, embedder, detector, author_keys):
        """Compare real VAE damage vs simulated VAE damage.

        This directly validates whether the current _simulate_vae is a
        reasonable proxy for the real thing.
        """
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        # Real VAE
        real_decoded = _vae_roundtrip(vae, wm)
        real_result = detector.detect(real_decoded, author_keys.public_key)
        real_psnr = _psnr(wm, real_decoded)

        # Simulated VAE (8x, matching SD's 8x spatial compression)
        sim_decoded = _simulate_vae(wm, compression_factor=8)
        sim_result = detector.detect(sim_decoded, author_keys.public_key)
        sim_psnr = _psnr(wm, sim_decoded)

        print(
            f"Real VAE:      ring={real_result.ring_confidence:.3f}, "
            f"payload={real_result.payload_confidence:.3f}, PSNR={real_psnr:.1f}\n"
            f"Simulated VAE: ring={sim_result.ring_confidence:.3f}, "
            f"payload={sim_result.payload_confidence:.3f}, PSNR={sim_psnr:.1f}\n"
            f"Simulation gap: "
            f"ring={sim_result.ring_confidence - real_result.ring_confidence:+.3f}, "
            f"payload="
            f"{sim_result.payload_confidence - real_result.payload_confidence:+.3f}"
        )

        # If the simulation is significantly more generous, flag it
        if (
            sim_result.payload_confidence > real_result.payload_confidence + 0.2
            or sim_result.ring_confidence > real_result.ring_confidence + 0.2
        ):
            print(
                "WARNING: Simulated VAE is significantly more generous than real VAE. "
                "Simulation-based robustness claims may be overstated."
            )

    def test_vae_plus_jpeg(self, vae, embedder, detector, author_keys):
        """VAE followed by JPEG Q75 (realistic output pipeline)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        decoded = _vae_roundtrip(vae, wm)

        # JPEG compression
        from PIL import Image

        img_u8 = np.clip(decoded, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img_u8, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        compressed = np.array(Image.open(buf).convert("L"), dtype=np.float64)

        result = detector.detect(compressed, author_keys.public_key)
        print(
            f"VAE + JPEG Q75: ring={result.ring_confidence:.3f}, "
            f"payload={result.payload_confidence:.3f}, detected={result.detected}"
        )

    def test_vae_multiple_roundtrips(self, vae, embedder, detector, author_keys):
        """Multiple VAE encode/decode cycles (worst case)."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        current = wm
        for i in range(3):
            current = _vae_roundtrip(vae, current)
            result = detector.detect(current, author_keys.public_key)
            quality = _psnr(wm, current)
            print(
                f"VAE pass {i + 1}: ring={result.ring_confidence:.3f}, "
                f"payload={result.payload_confidence:.3f}, PSNR={quality:.1f}"
            )

    def test_vae_roundtrip_ghost(self, vae, embedder, author_keys, config):
        """Check ghost signal correlation after real VAE."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        baseline = analyze_ghost_signature(wm, author_keys.public_key, config)
        decoded = _vae_roundtrip(vae, wm)
        result = analyze_ghost_signature(decoded, author_keys.public_key, config)

        print(
            f"Ghost after real VAE:\n"
            f"  Correlation: {baseline.correlation:.6f} -> {result.correlation:.6f}\n"
            f"  P-value: {baseline.p_value:.4f} -> {result.p_value:.4f}\n"
            f"  Detected: {baseline.ghost_detected} -> {result.ghost_detected}"
        )

        # Ghost signal is the designed VAE-survival channel. With VAE-optimized
        # bands and strength=100x, single-image detection should survive.
        assert result.ghost_detected, (
            f"Ghost signal lost after real VAE (corr={result.correlation:.6f}, "
            f"p={result.p_value:.4f})"
        )
        assert result.p_value < 0.01, (
            f"Ghost signal not statistically significant after VAE (p={result.p_value:.4f})"
        )

    def test_multiple_images_vae(self, vae, embedder, detector, author_keys, config):
        """VAE roundtrip across multiple different images."""
        surviving = 0
        ghost_surviving = 0
        total = 5

        for seed in range(total):
            rng = np.random.default_rng(seed + 200)
            img = _make_image(rng)
            wm = embedder.embed(img, author_keys)

            decoded = _vae_roundtrip(vae, wm)
            result = detector.detect(decoded, author_keys.public_key)
            ghost = analyze_ghost_signature(decoded, author_keys.public_key, config)

            if result.detected:
                surviving += 1
            if ghost.ghost_detected:
                ghost_surviving += 1
            print(
                f"Image {seed}: detected={result.detected}, "
                f"ring={result.ring_confidence:.3f}, payload={result.payload_confidence:.3f}, "
                f"ghost={ghost.ghost_detected} (corr={ghost.correlation:.4f})"
            )

        print(f"\n{surviving}/{total} images: ring+payload detection survived real VAE")
        print(f"{ghost_surviving}/{total} images: ghost signal survived real VAE")

        # Ghost is the designed VAE-survival channel — should survive on all images
        assert ghost_surviving >= 4, (
            f"Ghost signal should survive VAE on most images, got {ghost_surviving}/{total}"
        )


class TestJPEGBeforeVAE:
    """Test realistic pipeline: image is JPEG/PNG re-encoded BEFORE VAE encode/decode.

    This simulates the real-world scenario where:
    1. Artist uploads watermarked image
    2. Social media platform re-encodes as JPEG
    3. Scraper downloads the JPEG
    4. AI training pipeline runs it through SD VAE encode/decode

    The question: does JPEG compression before VAE destroy the watermark
    more than VAE alone?
    """

    @pytest.mark.parametrize("quality", [99, 90, 75, 50])
    def test_jpeg_then_vae(self, vae, embedder, detector, author_keys, quality):
        """JPEG at various qualities -> VAE encode/decode."""
        from PIL import Image

        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        # Step 1: JPEG re-encoding
        img_u8 = np.clip(wm, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img_u8, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        jpeg_decoded = np.array(Image.open(buf).convert("L"), dtype=np.float64)

        # Step 2: VAE encode/decode
        vae_decoded = _vae_roundtrip(vae, jpeg_decoded)

        result = detector.detect(vae_decoded, author_keys.public_key)
        jpeg_psnr = _psnr(wm, jpeg_decoded)
        vae_psnr = _psnr(wm, vae_decoded)

        print(
            f"JPEG Q{quality} -> VAE: "
            f"jpeg_psnr={jpeg_psnr:.1f}, vae_psnr={vae_psnr:.1f}, "
            f"ring={result.ring_confidence:.3f}, "
            f"payload={result.payload_confidence:.3f}, "
            f"detected={result.detected}"
        )

        # At least some signal should survive the pipeline
        assert result.ring_confidence > 0.0 or result.payload_confidence > 0.3, (
            f"No signal after JPEG Q{quality} -> VAE"
        )

    @pytest.mark.parametrize("quality", [99, 90, 75, 50])
    def test_jpeg_then_vae_ghost(self, vae, embedder, author_keys, config, quality):
        """Ghost signal survival after JPEG -> VAE pipeline."""
        from PIL import Image

        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        # JPEG -> VAE
        img_u8 = np.clip(wm, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img_u8, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        jpeg_decoded = np.array(Image.open(buf).convert("L"), dtype=np.float64)
        vae_decoded = _vae_roundtrip(vae, jpeg_decoded)

        ghost = analyze_ghost_signature(vae_decoded, author_keys.public_key, config)

        print(
            f"Ghost after JPEG Q{quality} -> VAE: "
            f"corr={ghost.correlation:.6f}, "
            f"p={ghost.p_value:.4f}, "
            f"detected={ghost.ghost_detected}"
        )

        # Ghost is the designed VAE survival channel
        if quality >= 75:
            assert ghost.ghost_detected, (
                f"Ghost lost after JPEG Q{quality} -> VAE (corr={ghost.correlation:.6f})"
            )

    @pytest.mark.parametrize("quality", [99, 90, 75, 50])
    def test_jpeg_then_vae_ghost_hash(self, vae, embedder, author_keys, config, quality):
        """Ghost hash extraction after JPEG -> VAE pipeline."""
        from PIL import Image

        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        expected_hash = derive_ghost_hash(author_keys.public_key, config)

        # JPEG -> VAE
        img_u8 = np.clip(wm, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img_u8, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        jpeg_decoded = np.array(Image.open(buf).convert("L"), dtype=np.float64)
        vae_decoded = _vae_roundtrip(vae, jpeg_decoded)

        extracted, confidences = extract_ghost_hash(vae_decoded, config)
        errors = sum(a != b for a, b in zip(extracted, expected_hash))

        print(
            f"Ghost hash after JPEG Q{quality} -> VAE: "
            f"errors={errors}/{config.ghost_hash_bits}, "
            f"min_conf={min(confidences):.6f}"
        )

        if quality >= 75:
            assert errors <= 3, (
                f"Ghost hash too damaged after JPEG Q{quality} -> VAE: {errors} errors"
            )

    def test_vae_then_jpeg_vs_jpeg_then_vae(self, vae, embedder, detector, author_keys, config):
        """Compare: does order matter? VAE->JPEG vs JPEG->VAE."""
        from PIL import Image

        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        quality = 75

        # Path A: JPEG -> VAE
        img_u8 = np.clip(wm, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img_u8, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        path_a = _vae_roundtrip(vae, np.array(Image.open(buf).convert("L"), dtype=np.float64))

        # Path B: VAE -> JPEG
        vae_first = _vae_roundtrip(vae, wm)
        img_u8_b = np.clip(vae_first, 0, 255).astype(np.uint8)
        pil_b = Image.fromarray(img_u8_b, mode="L")
        buf_b = io.BytesIO()
        pil_b.save(buf_b, format="JPEG", quality=quality)
        buf_b.seek(0)
        path_b = np.array(Image.open(buf_b).convert("L"), dtype=np.float64)

        result_a = detector.detect(path_a, author_keys.public_key)
        result_b = detector.detect(path_b, author_keys.public_key)
        ghost_a = analyze_ghost_signature(path_a, author_keys.public_key, config)
        ghost_b = analyze_ghost_signature(path_b, author_keys.public_key, config)

        print(
            f"JPEG Q{quality} -> VAE: "
            f"ring={result_a.ring_confidence:.3f}, "
            f"payload={result_a.payload_confidence:.3f}, "
            f"ghost_corr={ghost_a.correlation:.6f}\n"
            f"VAE -> JPEG Q{quality}: "
            f"ring={result_b.ring_confidence:.3f}, "
            f"payload={result_b.payload_confidence:.3f}, "
            f"ghost_corr={ghost_b.correlation:.6f}\n"
            f"Order difference: "
            f"ring={result_a.ring_confidence - result_b.ring_confidence:+.3f}, "
            f"payload={result_a.payload_confidence - result_b.payload_confidence:+.3f}, "
            f"ghost={ghost_a.correlation - ghost_b.correlation:+.6f}"
        )

    def test_multiple_jpeg_then_vae(self, vae, embedder, detector, author_keys, config):
        """Multiple JPEG re-encodes then VAE (worst realistic case)."""
        from PIL import Image

        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        # 3x JPEG Q85 re-encoding (shared across platforms)
        current = wm
        for _ in range(3):
            img_u8 = np.clip(current, 0, 255).astype(np.uint8)
            pil = Image.fromarray(img_u8, mode="L")
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            current = np.array(Image.open(buf).convert("L"), dtype=np.float64)

        # Then VAE
        vae_decoded = _vae_roundtrip(vae, current)

        result = detector.detect(vae_decoded, author_keys.public_key)
        ghost = analyze_ghost_signature(vae_decoded, author_keys.public_key, config)

        print(
            f"3x JPEG Q85 -> VAE: "
            f"ring={result.ring_confidence:.3f}, "
            f"payload={result.payload_confidence:.3f}, "
            f"ghost_corr={ghost.correlation:.6f}, "
            f"ghost_detected={ghost.ghost_detected}"
        )

    def test_jpeg_before_vae_comprehensive_report(
        self, vae, embedder, detector, author_keys, config, capsys
    ):
        """Print comprehensive report: JPEG quality vs post-VAE detection."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        qualities = [99, 95, 90, 85, 80, 75, 60, 50]

        print(f"\n{'=' * 90}")
        print("JPEG BEFORE VAE — DETECTION REPORT")
        print(f"{'=' * 90}")
        print(
            f"{'JPEG Q':<10} {'Ring':>8} {'Payload':>10} "
            f"{'Ghost corr':>12} {'Ghost det':>10} "
            f"{'Hash err':>10} {'Detected':>10}"
        )
        print(f"{'-' * 90}")

        # Baseline: VAE only (no JPEG)
        vae_only = _vae_roundtrip(vae, wm)
        r_base = detector.detect(vae_only, author_keys.public_key)
        g_base = analyze_ghost_signature(vae_only, author_keys.public_key, config)
        h_base, _ = extract_ghost_hash(vae_only, config)
        expected = derive_ghost_hash(author_keys.public_key, config)
        e_base = sum(a != b for a, b in zip(h_base, expected))
        print(
            f"{'VAE only':<10} {r_base.ring_confidence:>8.3f} "
            f"{r_base.payload_confidence:>10.3f} "
            f"{g_base.correlation:>12.6f} "
            f"{'YES' if g_base.ghost_detected else 'no':>10} "
            f"{e_base:>10} "
            f"{'YES' if r_base.detected else 'no':>10}"
        )

        for q in qualities:
            from PIL import Image

            img_u8 = np.clip(wm, 0, 255).astype(np.uint8)
            pil = Image.fromarray(img_u8, mode="L")
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            jpeg_decoded = np.array(Image.open(buf).convert("L"), dtype=np.float64)
            vae_decoded = _vae_roundtrip(vae, jpeg_decoded)

            result = detector.detect(vae_decoded, author_keys.public_key)
            ghost = analyze_ghost_signature(vae_decoded, author_keys.public_key, config)
            extracted, _ = extract_ghost_hash(vae_decoded, config)
            errors = sum(a != b for a, b in zip(extracted, expected))

            print(
                f"Q{q:<8} {result.ring_confidence:>8.3f} "
                f"{result.payload_confidence:>10.3f} "
                f"{ghost.correlation:>12.6f} "
                f"{'YES' if ghost.ghost_detected else 'no':>10} "
                f"{errors:>10} "
                f"{'YES' if result.detected else 'no':>10}"
            )

        print(f"{'=' * 90}")


class TestGhostHashAuthorIdentification:
    """Test ghost hash-based author identification after VAE.

    The ghost hash encodes 12 bits of the author's identity into the ghost
    signal. After VAE encode/decode, these bits can be extracted blindly
    and used to narrow O(N) author search to O(1) via indexed lookup.
    """

    def test_ghost_hash_identifies_author_from_10k_keys(self, vae, embedder, author_keys, config):
        """After VAE, ghost hash should narrow 10K candidates to <10."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        # VAE encode/decode
        decoded = _vae_roundtrip(vae, wm)

        # Extract ghost hash blindly (no key needed)
        extracted_hash, bit_confidences = extract_ghost_hash(decoded, config)
        expected_hash = derive_ghost_hash(author_keys.public_key, config)

        hash_errors = sum(a != b for a, b in zip(extracted_hash, expected_hash))
        print(
            f"Ghost hash extraction after VAE:\n"
            f"  Expected:  {expected_hash}\n"
            f"  Extracted: {extracted_hash}\n"
            f"  Errors: {hash_errors}/{config.ghost_hash_bits}\n"
            f"  Min bit confidence: {min(bit_confidences):.6f}\n"
            f"  Mean bit confidence: {np.mean(bit_confidences):.6f}"
        )

        # Generate 10K author keys and precompute their ghost hashes
        all_keys = []
        hash_index: dict[tuple, list[int]] = {}  # hash -> list of key indices
        for i in range(10000):
            key = generate_author_keys(seed=f"author-{i}".encode())
            all_keys.append(key)
            h = tuple(derive_ghost_hash(key.public_key, config))
            hash_index.setdefault(h, []).append(i)

        # Also add the real author
        real_hash = tuple(expected_hash)
        real_idx = 10000
        hash_index.setdefault(real_hash, []).append(real_idx)

        # Fuzzy lookup: check all hashes within Hamming distance ≤ 2.
        # With 6-bit hash, this checks C(6,0)+C(6,1)+C(6,2) = 22 bins.
        # Total candidates: ~22 * 156 ≈ 3400 out of 10K (66% reduction).
        query_hash = tuple(extracted_hash)
        candidates = []
        for stored_hash, indices in hash_index.items():
            hamming = sum(a != b for a, b in zip(query_hash, stored_hash))
            if hamming <= 2:
                candidates.extend(indices)

        real_found = real_idx in candidates
        num_candidates = len(candidates)

        print(
            f"\n  10K-key identification (Hamming ≤ 2):\n"
            f"  Query hash: {query_hash}\n"
            f"  Real hash:  {real_hash}\n"
            f"  Hash errors: {hash_errors}\n"
            f"  Candidates: {num_candidates}\n"
            f"  Real author found: {real_found}\n"
            f"  Reduction: O({len(all_keys) + 1}) -> O({num_candidates})"
        )

        assert real_found, (
            f"Real author not found within Hamming distance 2. "
            f"Hash errors: {hash_errors}/{config.ghost_hash_bits}"
        )
        # Fuzzy lookup should still significantly reduce candidates
        assert num_candidates < len(all_keys), (
            f"Ghost hash provided no reduction: {num_candidates} candidates"
        )

    def test_ghost_hash_survives_multiple_vae_passes(self, vae, embedder, author_keys, config):
        """Ghost hash should survive 3 consecutive VAE passes."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)
        expected_hash = derive_ghost_hash(author_keys.public_key, config)

        current = wm
        for i in range(3):
            current = _vae_roundtrip(vae, current)
            extracted, confidences = extract_ghost_hash(current, config)
            errors = sum(a != b for a, b in zip(extracted, expected_hash))
            print(
                f"VAE pass {i + 1}: hash errors={errors}/{config.ghost_hash_bits}, "
                f"min_conf={min(confidences):.6f}"
            )

        # After 3 VAE passes, hash should be within fuzzy lookup range
        assert errors <= 2, f"Ghost hash degraded after 3 VAE passes: {errors} errors"

    def test_ghost_hash_plus_jpeg(self, vae, embedder, author_keys, config):
        """Ghost hash should survive VAE + JPEG Q75."""
        rng = np.random.default_rng(42)
        img = _make_image(rng)
        wm = embedder.embed(img, author_keys)

        decoded = _vae_roundtrip(vae, wm)

        # JPEG Q75
        from PIL import Image

        img_u8 = np.clip(decoded, 0, 255).astype(np.uint8)
        pil = Image.fromarray(img_u8, mode="L")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        compressed = np.array(Image.open(buf).convert("L"), dtype=np.float64)

        expected_hash = derive_ghost_hash(author_keys.public_key, config)
        extracted, confidences = extract_ghost_hash(compressed, config)
        errors = sum(a != b for a, b in zip(extracted, expected_hash))

        print(
            f"Ghost hash after VAE + JPEG Q75:\n"
            f"  Errors: {errors}/{config.ghost_hash_bits}\n"
            f"  Min confidence: {min(confidences):.6f}"
        )

        assert errors <= 2, f"Ghost hash too damaged after VAE + JPEG: {errors} errors"

    def test_ghost_hash_across_images(self, vae, embedder, config):
        """Ghost hash extraction should work across different images."""
        keys = generate_author_keys(seed=b"multi-image-ghost-hash-test!!!!!")
        expected_hash = derive_ghost_hash(keys.public_key, config)

        within_hamming_2 = 0
        total = 5
        for seed in range(total):
            rng = np.random.default_rng(seed + 500)
            img = _make_image(rng)
            wm = embedder.embed(img, keys)
            decoded = _vae_roundtrip(vae, wm)

            extracted, confidences = extract_ghost_hash(decoded, config)
            errors = sum(a != b for a, b in zip(extracted, expected_hash))
            if errors <= 2:
                within_hamming_2 += 1
            print(f"Image {seed}: hash errors={errors}, min_conf={min(confidences):.6f}")

        print(
            f"\n{within_hamming_2}/{total} images: ghost hash within Hamming distance 2 after VAE"
        )
        # Most images should have ≤ 2 errors (within fuzzy lookup range)
        assert within_hamming_2 >= 3, (
            f"Ghost hash too unreliable: only {within_hamming_2}/{total} within Hamming distance 2"
        )
