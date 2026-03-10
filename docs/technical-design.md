# Sigil Watermark System

Technical documentation for the Sigil invisible watermark, Signarture's proprietary artist ownership watermark.

## Overview

Sigil is a three-layer, three-tier invisible watermark designed for artist ownership verification. It embeds a cryptographic identity into images that survives JPEG compression, cropping, rotation, scaling, print-scan cycles, and common image manipulation attacks.

**Design principles:**
- Pure signal processing (no neural networks, no training infrastructure)
- Cryptographic identity via Ed25519 keypairs (not user-typed text)
- QR-code-level robustness through redundancy at every layer
- Hardened against targeted removal by key-aware attackers
- Perceptually invisible on all content (35-47 dB on real photos/art with adaptive ring strength)

## Architecture

### Three Embedding Layers

```
Input Image
    |
    +- Layer 1: DFT Ring Anchor ------- Geometric compass (rotation-invariant)
    |
    +- Layer 2: DWT Tiled Payload ----- RS-encoded identity (crop-robust)
    |
    +- Layer 3: Ghost Signal ---------- Spectral bias at VAE-transparent frequencies
    |
Output Image (watermarked)
```

**Layer 1 — DFT Ring Anchor** (`transforms.py: embed_dft_rings`)
- Embeds up to 8 concentric ring peaks in the Fourier spectrum (4 key-derived + 2 content-dependent + 2 sentinel)
- Uses multiplicative embedding: `magnitude += alpha * ring_mask * capped_mag`, tying the watermark to local spectrum energy so a simple notch filter destroys image content
- Soft ceiling caps the multiplicative boost at 2x median magnitude to prevent distortion on high-energy spectrum peaks
- Per-ring strength scales by `4 / num_rings` to keep total energy constant regardless of ring count
- Ring profiles are broad Gaussians (sigma=0.04 in normalized frequency space) — forces attackers to notch wider bands, increasing their quality cost
- Key-derived phase modulation at each ring frequency adds a verification dimension that pure magnitude analysis cannot recover
- Ring radii are derived from the author's public key via HKDF
- Rings are rotation-invariant: they survive any rotation without correction

Three types of rings provide defense-in-depth:

| Ring Type | Count | Position Source | Purpose |
|-----------|-------|----------------|---------|
| Key-derived | 4 | HKDF from public key | Primary detection |
| Content-dependent | 2 | Image hash + public key | Attacker confusion only (embedded but not checked during detection — positions require the original image to derive, so the detector cannot verify them without it) |
| Sentinel | 2 | Server secret | Tampering detection: if sentinels survive but key rings are gone, targeted removal is suspected |

**Layer 2 — Fractal Sigil Tiling** (`tiling.py: tile_embed`, `embed.py`)
- Carries the actual payload: 8-bit beacon + 20-bit author index + 48-bit author ID
- Payload is Reed-Solomon encoded (nsym=8) for error correction: 76 raw bits -> ~144 encoded bits
- DWT decomposition (db4 wavelet, 3 levels) of the Y channel
- Spread-spectrum CDMA embedding in LH and HL subbands
- Each subband is tiled into independent blocks (best-fit from 32/64/128/256)
- Every tile embeds the same full payload with the same PN sequence
- Any single surviving tile can recover the full identity -> extreme crop robustness
- Spreading factor: 256 chips per payload bit (adaptive reduction for small tiles)
- Perceptual masking scales embedding strength by local image complexity

**Layer 3 — Training Ghost Signal** (`embed.py: _embed_ghost_signal`)
- Adds faint spectral energy at frequencies empirically measured to survive the Stable Diffusion VAE encode/decode cycle
- Ghost bands at (0.163, 0.218, 0.286, 0.343, 0.425) of Nyquist frequency, selected via `scripts/vae_passband_analysis.py` against `stabilityai/sd-vae-ft-mse`
- VAE survival ratios: 0.343 -> 55x, 0.425 -> 26x, 0.218 -> 24x, 0.163 -> 14x, 0.286 -> 9x
- Bandwidth: 0.05 (wider than original 0.03 for more robust detection)
- Strength multiplier: 200x (spectrally narrow signal has zero quality impact — PSNR stays >46.9 dB)
- Ghost hash: 8-bit author fingerprint encoded into the ghost signal via orthogonal universal PN sequences, enabling blind author binning after VAE (see Ghost Hash section)
- Embedded in Y channel only — multi-channel (RGB) embedding was tested but hurts VAE survival because the SD VAE mixes channels in its latent space, destroying per-channel PN coherence
- Single-image ghost detection after real SD VAE: 100% (up from 40% before optimization)
- Ghost scaling capped at local_median=2000 to prevent unbounded distortion

### Three Detection Tiers

```
Input Image
    |
    +- Tier 1: Beacon Check ---- "Is this a Signarture watermark?" (8 bits)
    |
    +- Tier 2: Author Index ---- "Whose watermark is it?" (20 bits, blind scan)
    |
    +- Tier 3: Author Verify --- "Cryptographic proof of authorship" (48 bits)
    |
Detection Result
```

All three tiers are extracted from a single combined RS-encoded payload:

| Field | Bits | Purpose |
|-------|------|---------|
| Beacon | 8 | All-ones marker — universal Signarture identifier |
| Author Index | 20 | Derived from public key — enables blind scanning without key |
| Author ID | 48 | Truncated SHA-256 of public key — cryptographic verification |

**Tier 1 — Universal Beacon** (no key required)
- First 8 bits of decoded payload should all be 1
- If >=70% match -> beacon found -> image contains a Signarture watermark
- Fast check: detect_beacon() runs the full extraction but only checks beacon bits

**Tier 2 — Author Index** (no key required)
- Next 20 bits identify the author (2^20 = ~1M unique slots)
- Used for blind scanning: "which artist watermarked this image?"
- Can be matched against a database of known public keys by precomputing their indices

**Tier 3 — Author Verification** (requires candidate public key)
- Final 48 bits are compared against the expected author ID derived from the public key
- BER < 15% -> author verified (with RS correction, typically exact match)
- Provides cryptographic proof: only the holder of the Ed25519 private key could have produced this watermark

### Detection Pipeline

```python
detector = SigilDetector(config)
result = detector.detect(image, candidate_public_key)

# result.detected            -> True/False
# result.beacon_found        -> Tier 1
# result.author_index        -> Tier 2 (20-bit list)
# result.author_id_match     -> Tier 3
# result.confidence          -> Combined confidence (0-1)
# result.ring_confidence     -> Layer 1 DFT ring detection strength
# result.payload_confidence  -> Layer 2 payload correlation strength
# result.ghost_confidence    -> Layer 3 ghost signal strength (scaled by hash match)
# result.ghost_hash          -> Extracted 8-bit author hash (blind, no key needed)
# result.ghost_hash_match    -> Whether ghost hash matches the candidate key
# result.tampering_suspected -> Sentinel rings present but key rings removed
```

### Ring Detection — Spectral Whitening

Ring detection uses **spectral whitening** to handle natural images with 1/f spectral profiles. Without whitening, the natural radial magnitude falloff (~3.4x across the ring region in typical photographs) completely masks the ring signal, producing 0% ring confidence on all real photographs despite working perfectly on synthetic test images.

The detection algorithm:
1. Compute the 2D DFT magnitude spectrum
2. Build a radial profile: mean magnitude per thin annular bin (200+ bins from 0.02 to 0.95 of Nyquist)
3. Fit a degree-2 polynomial in log-log space to model the smooth 1/f spectral slope
4. Divide the 2D magnitude by this radial model (**spectral whitening**) — removes the 1/f gradient while preserving the localized ring peaks
5. Compute NCC between the whitened spectrum and the expected ring template in the ring region
6. Map NCC to confidence: `(ncc - 0.03) / 0.09`, calibrated so unwatermarked images (NCC < 0.05) score near 0 and watermarked images (NCC 0.06-0.28) score 0.5-1.0

**Known limitation:** Strongly anisotropic images (e.g., landscapes with strong horizon lines) produce weaker ring detection because their spectral structure isn't well-modeled by a radial polynomial. The payload (Layer 2) and ghost signal (Layer 3) compensate — overall detection still succeeds.

Detection includes automatic geometric correction:
1. First attempt: detect without correction
2. If confidence is high (>0.7), return immediately
3. If ring confidence is decent (>0.3) but payload is weak (<0.5), try rotations
4. Brute-force angles: [0, 90, 180, 270, 1, -1, 2, -2, 5, -5] degrees
5. Pick the rotation that gives the best payload confidence

### Tampering Detection

The detector checks sentinel rings independently from key-derived rings. If sentinel rings are present (confidence > 0.5) but key-derived rings are absent (confidence < 0.2), `tampering_suspected` is set to `True`. This indicates that someone with access to the author's public key attempted targeted ring removal but didn't know about the sentinel ring positions (which are derived from a server secret).

## Ring Hardening

Layer 1 was hardened against a key-aware attacker who knows the author's public key and can derive the exact ring positions. Five changes were made:

### 1. Multiplicative Embedding
**Old:** `magnitude += scaled_strength * ring_mask` (additive, constant boost regardless of content)
**New:** `magnitude += alpha * ring_mask * capped_mag` (proportional to local spectrum energy)

An attacker can't subtract a constant to remove the ring — they'd need to know the original magnitude at each frequency bin. The soft ceiling (2x median) prevents runaway distortion on high-energy peaks.

### 2. Broader Ring Profiles
**Old:** Ring width sigma=0.015 (narrow, cheap to notch)
**New:** Ring width sigma=0.04 (wide, attacker must destroy broader frequency bands)

The quality cost of notch filtering increased from PSNR > 35 dB to ~34.5 dB for the attacker. Natural content in the notched bands is also destroyed, making the attack visible.

### 3. Phase Modulation
Key-derived phase offsets (limited to +/-0.3 radians to avoid destructive interference) are applied at ring frequencies. Phase modulation serves a security purpose — it makes the embedded ring signal key-dependent, so a generic ring template won't match. However, detection relies solely on magnitude-based NCC (not phase coherence). Empirical testing showed phase coherence at ring locations is ~0.01 even on watermarked images (essentially noise-floor), which was diluting magnitude confidence by ~30% when weighted in. JPEG compression further degrades phase, making it unreliable as a detection signal.

### 4. Content-Dependent Ring Positions
2 of the 8 rings derive their positions from a coarse hash (8x8 block means) of the image content combined with the author's key. An attacker can't predict these positions without access to the original image. The hash is quantized for robustness to minor pixel changes.

### 5. Sentinel Rings
2 rings at positions derived from a server secret. The attacker has no way to find or remove them. If key-derived rings are removed but sentinels remain, the system flags `tampering_suspected`.

## Adaptive Ring Strength

Without adaptation, multiplicative ring embedding produces dramatically different PSNR across image types: 42–50 dB on synthetic images but 24–35 dB on real photographs and artwork. This happens because `magnitude += alpha * ring_mask * capped_mag` adds energy proportional to existing spectral content — real photographs have rich spectra, so the multiplicative boost produces larger pixel-domain changes.

### Parseval's Theorem Approach

Adaptive ring strength uses **Parseval's theorem** to predict the ring layer's pixel-domain MSE directly in the frequency domain, then scales alpha down if the predicted PSNR falls below a target:

```
MSE_pixel = (1 / h*w) * sum(|delta_F|^2)
          = (alpha^2 / h*w) * sum_rings(|ring_mask * capped_mag|^2)
```

This gives an exact PSNR prediction without trial embedding — zero additional cost beyond one pass over ring frequencies (already computed).

If the predicted PSNR < `ring_target_psnr`, alpha is reduced:
```python
target_mse = 255^2 / 10^(target_psnr/10)
max_alpha = sqrt(target_mse * h * w / sum_sq)
alpha = max(max_alpha, base_alpha * min_alpha_fraction)
```

The floor (`min_alpha_fraction`) prevents alpha from dropping so low that ring detection becomes impossible.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adaptive_ring_strength` | `True` | Enable Parseval-based adaptive scaling |
| `ring_target_psnr` | 36.0 | Target PSNR for the ring layer alone (dB). Overall PSNR is ~2-4 dB lower due to DWT + ghost layers. |
| `ring_min_alpha_fraction` | 0.30 | Floor: alpha never drops below 30% of nominal. Ensures ring_confidence >= ~0.16 on all tested images. |

### Impact on Detection

With adaptation, ring_confidence drops from ~0.7 to 0.15–0.30 on spectrally rich images. The ring layer shifts from **primary detection signal** to **supporting evidence and geometric compass**. Detection now primarily relies on:

- `author_id_match` (payload layer) — the strongest signal
- `beacon_found` (payload layer) — blind detection
- Ring confidence still contributes to overall confidence blend (0.35 weight) and serves as geometric correction anchor

The `detected` condition is unchanged: `payload_confidence > 0.5 AND (beacon_found OR ring_confidence > 0.5 OR author_id_match)`. On adapted images, `ring_confidence > 0.5` rarely triggers — `author_id_match` or `beacon_found` drive detection instead.

### Tradeoffs

| Upside | Downside |
|--------|----------|
| +10 dB avg PSNR on real photos (29.7 → 40.1 dB) | Ring layer alone can no longer drive detection |
| No visible artifacts on any tested image (min 34.9 dB) | Reduced noise robustness under combined attacks (sigma=10 + key-alpha combos may produce `detected=False`) |
| Mathematically exact, zero-cost prediction | Spectrally rich images get weaker ring protection against notch filtering |
| 100% author_id_match on all clean images | ~14% of sigma=5-10 noise key/image combos retain signal but don't cross strict detection threshold |

### Quality Results with Adaptive Strength

See "Measured Quality — Real Photographs & Artwork" section below for full table.

## Cryptographic Identity

### Ed25519 Keypairs (`keygen.py`)

Each artist has one Ed25519 keypair:
- **Private key**: 32 bytes, stored encrypted in the database, used only during embedding
- **Public key**: 32 bytes, stored in the database, used for detection/verification

Key generation: `generate_author_keys(seed=None)` — random by default, or deterministic from a seed via HKDF.

### Key Derivation

All watermark parameters are deterministically derived from the public key using HKDF-SHA256:

| Parameter | Salt | Info | Purpose |
|-----------|------|------|---------|
| PN sequence | `signarture-sigil-v1-pn` | `pn-sequence` | Spread-spectrum chips |
| Ring radii | `signarture-sigil-v1-rings` | `ring-radii` | DFT ring positions |
| Ring phase offsets | `signarture-sigil-v1-rings` | `ring-phase-offsets` | Phase modulation |
| Content ring radii | `signarture-sigil-v1-content-rings` | `content-ring-radii` | Content-dependent positions |
| Sentinel ring radii | `signarture-sigil-v1-sentinel` | `sentinel-ring-radii` | Tampering detection |
| Author ID | SHA-256 of `signarture-author-id-v1 + pubkey` | — | 48-bit identity |
| Author index | SHA-256 of `signarture-author-index-v1 + pubkey` | — | 20-bit blind index |

Universal (key-independent) parameters:
| Parameter | Seed | Purpose |
|-----------|------|---------|
| Beacon PN | `signarture-universal-beacon-v1` | Blind payload extraction |
| Index PN | `signarture-universal-pn-v1` | Author index extraction |

## Error Correction

### Reed-Solomon FEC (`fec.py`)

The combined payload (76 bits) is RS-encoded before embedding:
- Bits are packed into bytes (MSB-first)
- `reedsolo.RSCodec(nsym=8)` adds 8 error-correction symbols
- Result is unpacked back to bits (~144 encoded bits)
- Can correct up to `nsym/2 = 4` symbol errors in the payload

This provides resilience against bit errors from JPEG compression, noise, and partial tile corruption.

## Robustness Features

### Fractal Tiling (`tiling.py`)

Each DWT subband is divided into tiles. Every tile independently embeds the full payload:

```
+--------+--------+--------+--------+
| Tile 1 | Tile 2 | Tile 3 | Tile 4 |  Each tile = full payload
+--------+--------+--------+--------+
| Tile 5 | Tile 6 | Tile 7 | Tile 8 |  Majority vote across tiles
+--------+--------+--------+--------+
| Tile 9 | Tile10 | Tile11 | Tile12 |  Any 1 surviving tile
+--------+--------+--------+--------+  can recover the payload
| Tile13 | Tile14 | Tile15 | Tile16 |
+--------+--------+--------+--------+
```

Tile size selection: `best_tile_size()` picks the largest tile from `(32, 64, 128, 256)` that fits in the subband while providing enough spreading capacity.

Detection extracts bits from every tile, then majority-votes across all tiles for the final answer.

### Color Support (`color.py`)

- RGB images are converted to YCbCr
- Watermark is embedded in the Y (luminance) channel only
- Y survives JPEG subsampling, color shifts, and print-scan best
- Cb/Cr channels are preserved and recombined after embedding

### Geometric Auto-Correction (`geometric.py`)

**Fourier-Mellin Transform** for rotation/scale estimation:
1. Compute DFT magnitude of test and reference images
2. Apply highpass filter to suppress DC
3. Convert to log-polar coordinates (rotation -> x-shift, scale -> y-shift)
4. Phase correlation gives (angle, scale) estimates
5. Apply inverse transform to correct the image

**Brute-force rotation** (`try_rotations`):
- When no reference image is available, tries common angles: [0, 90, 180, 270, 1, -1, 2, -2, 5, -5]
- Picks the rotation that maximizes detection confidence
- Integrated into `SigilDetector.detect()` automatically

### Perceptual Masking (`perceptual.py`)

Embedding strength adapts to local image complexity:
- Gradient magnitude analysis detects edges and texture
- Noise estimation via median filter identifies smooth vs. textured regions
- JND (Just Noticeable Difference) threshold prevents visible artifacts
- Result: stronger embedding in textured regions, weaker in flat regions

## Ghost Signal — VAE Optimization

The ghost signal was empirically optimized for maximum survival through the Stable Diffusion VAE (`stabilityai/sd-vae-ft-mse`):

### VAE Passband Analysis
`scripts/vae_passband_analysis.py` measures per-frequency attenuation across 50 diverse images. The top VAE-transparent bands were identified:

| Frequency (Nyquist) | VAE Survival Ratio | Selected? |
|---------------------|-------------------|-----------|
| 0.343 | 55x | Yes |
| 0.425 | 26x | Yes |
| 0.218 | 24x | Yes |
| 0.163 | 14x | Yes |
| 0.286 | 9x | Yes |
| 0.500 (old) | 5x | No |
| 0.250 (old) | ~5x | No |
| 0.125 (old) | ~9x | No |

### Perceptual Masking
Ghost modulation depth is scaled by the perceptual mask's mean value — full strength in textured/detailed images, reduced in smooth images (sky, flat gradients). This prevents subtle banding artifacts on smooth regions that PSNR/SSIM wouldn't capture. The mask is the same local-variance JND mask used for Layer 2 embedding.

### Strength vs. Quality
The ghost signal is spectrally narrow enough that increasing strength has negligible quality impact:

| Multiplier | PSNR | SSIM | VAE Detection Rate |
|-----------|------|------|--------------------|
| 1.5 (old) | 47.2 | 0.995 | 40% |
| 50 | 47.1 | 0.994 | 80% |
| 100 | 47.0 | 0.994 | 100% |
| 200 (current) | 46.9 | 0.994 | 100% |

### Multi-Channel Embedding (tested, rejected)
RGB ghost embedding was tested — embedding independent PN sequences per channel. Through the VAE, per-channel coherence is destroyed because the SD VAE mixes channels in its learned latent space. Y-only embedding is definitively better.

### Ghost Hash — Author Binning After VAE

The ghost signal carries an 8-bit author fingerprint that survives the SD VAE encode/decode cycle, enabling blind author identification without knowing the candidate key.

**How it works:**
1. Derive 8 hash bits from the author's public key via SHA-256
2. Generate 8 universal (key-independent) PN sequences from fixed seeds
3. Build a composite PN: `(1/sqrt(8)) * sum(sign_i * pn_i)` where `sign_i = +1 if bit_i=1, -1 if bit_i=0`
4. Embed the composite into all ghost bands (same energy as single-PN approach)
5. To extract: correlate with each universal PN, sign of correlation = hash bit

**Performance after real SD VAE:**
- Per-bit error rate: ~1-2 bits out of 8 after VAE encode/decode
- Fuzzy lookup (Hamming distance ≤ 2): finds correct author in 5/5 test images
- 10K artist database → ~1,400 candidates (86% reduction, vs O(10K) brute force)
- Postgres can scan 1,400 candidates in <100ms with ghost composite correlation

**Lookup flow:**
1. Extract ghost hash blindly (no key needed): `extract_ghost_hash(image, config)`
2. Query database: `SELECT * FROM artists WHERE hamming_distance(ghost_hash, $query) <= 2`
3. For each candidate (~1,400), verify with full detection pipeline
4. The ghost hash is stored as an indexed integer column for fast lookup

**Why not more bits?** Per-bit SNR drops by `1/sqrt(N)` since N composite PNs share the same ghost energy. With 8 bits, the weakest bits have confidence ~0.003, near the noise floor. More bits would cause too many extraction errors after VAE.

### Measured Quality — Real Photographs & Artwork

Tested on 8 public domain images (paintings + photographs) at various resolutions with **adaptive ring strength** enabled (`ring_target_psnr=36.0`, `ring_min_alpha_fraction=0.30`). These numbers represent what artists will actually experience:

| Image | Size | PSNR | Max Dev | Ring Conf | Detected | Author Match |
|-------|------|------|---------|-----------|----------|-------------|
| Girl with Pearl Earring | 800x937 | 44.7 dB | 26 | 0.18 | YES | YES |
| Water Lilies (Monet) | 1280x1230 | 41.2 dB | 33 | 0.19 | YES | YES |
| Photo Architecture (Pyramids) | 1280x851 | 40.5 dB | 40 | 0.22 | YES | YES |
| Photo Landscape (Yosemite) | 1280x835 | 39.8 dB | 30 | 0.17 | YES | YES |
| Persistence of Memory (Dali) | 368x271 | 39.2 dB | 28 | 0.16 | YES | YES |
| Starry Night (Van Gogh) | 1280x1014 | 38.4 dB | 34 | 0.20 | YES | YES |
| Photo Urban (Times Square) | 1280x853 | 38.1 dB | 42 | 0.26 | YES | YES |
| Great Wave (Hokusai) | 1280x860 | **34.9 dB** | **48** | 0.18 | YES | YES |
| **Average** | | **40.1 dB** | **35** | | | |

**Before adaptive strength (for comparison):** avg 29.7 dB, min 24.4 dB (Great Wave had max_dev=124 — visually noticeable). Adaptive strength raised the floor by +10 dB average.

**Why ring_confidence is lower:** With adaptive strength, alpha is reduced on spectrally rich images. Ring confidence drops from ~0.7 to 0.15-0.30, but NCC-based ring detection is strength-agnostic (normalized). The ring signal is still present — just weaker. Detection relies on `author_id_match` rather than `ring_confidence > 0.5`.

**Key-dependent variation:** PSNR varies by 1-3 dB across different author keys because each key places rings at different frequencies. Some frequency placements hit stronger spectral peaks in specific images.

### JPEG Re-Encoding Robustness

Systematic JPEG quality sweep (grayscale natural scene, embed_strength=3.0, ring_strength=20.0, adaptive ring strength enabled):

**Note:** With adaptive ring strength, ring confidence on synthetic images drops from ~0.7 to ~0.2-0.3 (alpha is scaled down based on spectral energy). Detection is driven by `author_id_match` from the payload layer.

**Key findings:**
- Full detection (detected + author_id_match) survives down to **JPEG Q60**
- Ring layer retains signal (confidence ~0.2) with adaptive strength
- Author ID match fails below Q50 (RS-encoded payload accumulates too many bit errors)
- Ghost signal remains detectable (>0.25) down to Q20
- Sequential re-encoding (3x Q75, Q99→Q85→Q75 chain) survives with full detection

RGB JPEG follows the same pattern (tested Q50-Q99, full detection through Q60).

### JPEG Before VAE (Social Media → AI Training Pipeline)

Tests the realistic scenario where an image is JPEG-compressed (social media upload) before being fed through the SD VAE encoder/decoder (AI training). This is tested in `test_real_vae_attack.py::TestJPEGBeforeVAE` (GPU required).

Test matrix: JPEG Q50/Q75/Q90/Q99 → VAE, plus VAE→JPEG vs JPEG→VAE order comparison, and 3x JPEG Q85 → VAE worst case.

### LoRA Propagation (tested, fundamental limitation)
LoRA hyperparameter sweep (rank 8-64, steps 500-2000) with real test images proved the ghost propagation mechanism fundamentally doesn't scale:
- More training steps overwrites the ghost signal
- Higher LoRA rank doesn't help capture more signal
- Ceiling holds at all configurations tested
- The mechanism itself needs fundamental change (style-entangled approach needed) for reliable single-image detection of LoRA-generated images

## Configuration

All parameters in `config.py: SigilConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_rings` | 4 | Key-derived DFT ring count |
| `num_content_rings` | 2 | Content-dependent ring count |
| `num_sentinel_rings` | 2 | Sentinel ring count |
| `ring_radius_min/max` | 0.1 / 0.35 | Ring frequency range (fraction of Nyquist) |
| `ring_strength` | 20.0 | Ring embedding strength (multiplicative alpha = strength/50) |
| `ring_width` | 0.04 | Ring Gaussian sigma (fraction of Nyquist) |
| `tile_sizes` | (32, 64, 128, 256) | Available tile sizes |
| `wavelet` | db4 | DWT wavelet family |
| `dwt_levels` | 3 | DWT decomposition depth |
| `embed_subbands` | (LH, HL) | Which detail subbands to embed in |
| `embed_strength` | 3.0 | Base spread-spectrum strength |
| `spreading_factor` | 256 | CDMA chips per payload bit |
| `ghost_bands` | (0.163, 0.218, 0.286, 0.343, 0.425) | VAE-optimized ghost frequency positions |
| `ghost_bandwidth` | 0.05 | Ghost band Gaussian width |
| `ghost_strength_multiplier` | 200.0 | Ghost signal relative strength |
| `ghost_hash_bits` | 8 | Ghost hash bits for author binning |
| `adaptive_ring_strength` | True | Enable Parseval-based adaptive ring scaling |
| `ring_target_psnr` | 36.0 | Target PSNR for ring layer alone (dB) |
| `ring_min_alpha_fraction` | 0.30 | Floor: alpha never below 30% of nominal |
| `beacon_bits` | 8 | Beacon payload size |
| `author_index_bits` | 20 | Blind scan index size |
| `author_id_bits` | 48 | Cryptographic identity size |
| `rs_nsym` | 8 | Reed-Solomon error correction symbols |
| `target_psnr_db` | 40.0 | Visual quality target (actual: 35-47 dB with adaptive ring strength) |
| `target_ssim` | 0.98 | Structural similarity target |

## Strength Tuning and Backwards Compatibility

Embedding strength parameters (`ring_strength`, `embed_strength`) only affect the embedding pipeline.
The detection pipeline is **fully strength-agnostic** — it uses normalized cross-correlation (rings)
and sign-of-correlation bit decisions (payload), neither of which reference the embedding strength.

This means:
- **Changing strength defaults only affects newly embedded images.** Images embedded at prior
  strength settings (e.g., `ring_strength=50`, `embed_strength=5`) will still be detected normally,
  and will produce higher confidence scores due to their stronger signal.
- **No migration or re-embedding needed.** Old and new images coexist seamlessly.
- **The structural config must match.** Detection depends on `wavelet`, `dwt_levels`,
  `embed_subbands`, `tile_sizes`, `spreading_factor`, `rs_nsym`, and payload bit sizes. These
  define where to look and how to decode — changing any of these would break detection of old images.

### Strength history

| Date | `ring_strength` | `embed_strength` | Notes |
|------|----------------|-----------------|-------|
| Initial | 50.0 | 5.0 | Original defaults, additive rings, sigma=0.015 |
| 2026-03-05 | 20.0 | 3.0 | Reduced for imperceptibility |
| 2026-03-07 | 20.0 | 3.0 | Multiplicative rings, sigma=0.04, phase modulation, content+sentinel rings |
| 2026-03-07 | 20.0 | 3.0 | Ring detection fix: spectral whitening (was 0% on all real photographs) |
| 2026-03-07 | 20.0 | 3.0 | Ghost hash (8-bit author binning), ghost strength 100→200× |
| 2026-03-08 | 20.0 | 3.0 | Adaptive ring strength via Parseval's theorem (avg PSNR 29.7→40.1 dB on real photos) |

## Comparison with Adversarial Perturbation Approaches

### Background: Honig et al. (ICLR 2025)

"Adversarial Perturbations Cannot Reliably Protect Artists From Generative AI" (arxiv 2406.12027) evaluates tools like Glaze, Mist, and Anti-DreamBooth that add adversarial perturbations to artist images to prevent AI models from learning an artist's style. The paper finds that simple preprocessing bypasses all these protections:

| Bypass Method | How It Works | Effectiveness |
|---------------|-------------|---------------|
| Gaussian Noising | Add noise (sigma 0.05-0.25 on [0,1]) | Breaks most protections |
| DiffPure | Add noise -> denoise via SDXL | Strong bypass |
| Noisy Upscaling | Add noise -> run through SD Upscaler | Best bypass (>40% median success) |
| IMPRESS++ | White-box iterative optimization | Strongest but requires model access |

Key finding: "protections are broken from day one" — the bypass methods existed before the protections were created. A best-of-4 strategy (try multiple bypasses, pick best result) defeats all evaluated protections.

### Why Sigil Takes a Fundamentally Different Approach

The adversarial perturbation tools and Sigil watermarks solve **different problems**:

| | Adversarial Perturbations (Glaze etc.) | Sigil Watermark |
|---|---|---|
| **Goal** | PREVENT model from learning artist's style | SURVIVE training, propagate into generated images |
| **Mechanism** | Adversarial noise targeting model gradients | Signal processing in frequency domain |
| **Failure mode** | Preprocessing strips the perturbation -> model learns normally | Processing degrades signal -> detection confidence drops |
| **Fundamental limit** | Perturbation must be strong enough to mislead but weak enough to be invisible — a contradictory requirement | Signal only needs to be detectable, not to mislead anything |

The paper's core insight — that adversarial perturbations are inherently brittle because any preprocessing that restores perceptual quality also strips the perturbation — **does not apply to Sigil**. Our watermark is embedded in the frequency domain structure of the image, not as an adversarial gradient-based signal. The watermark doesn't need to deceive any model; it just needs to survive as a detectable signal.

### Sigil's Performance Against Paper's Bypass Methods

We tested Sigil against simulations of all the paper's bypass methods (`test_purification_attacks.py`):

| Attack | Sigil Survives? | Notes |
|--------|----------------|-------|
| Gaussian noise (sigma 0.05-0.15) | Full detection | Payload + ring both survive |
| Gaussian noise (sigma 0.20-0.25) | Partial detection | Payload survives; ring degraded (adaptive alpha) |
| DiffPure (bilateral denoise) | Full detection | Structure-preserving denoise doesn't remove frequency signal |
| DiffPure x3 (repeated) | Detection survives | Multi-round purification still leaves signal |
| Noisy Upscaling (2x) | Partial detection | Payload or ring layer survives |
| Noisy Upscaling (4x) | Partial detection | Heavy downsampling destroys payload tiles; ring persistence depends on adaptive alpha |
| VAE encode/decode (real SD 1.5) | Ghost layer survives | Rings fully destroyed (conf=0.002); ghost hash match + payload partial signal |
| VAE encode/decode (simulated) | Partial detection | Simulated bottleneck is more generous than real VAE |
| Training preprocessing | Partial detection | Resize + crop + noise + denoise |
| Full pipeline (social -> training) | Partial detection | JPEG -> resize -> noise -> denoise chain |

**Layer survival by attack class:**
- **Pixel-space attacks** (noise, JPEG, resize, denoise): Layers 1+2 survive. Ring and payload both contribute to detection.
- **Latent-space attacks** (VAE encode/decode): Layer 3 (ghost) survives. Rings are fully destroyed. Ghost hash enables blind author binning.
- **Targeted key-aware attacks**: Attacker can remove key-derived rings and payload, but sentinel rings and content-dependent rings survive (positions unknown to attacker).

### Implications for Signarture's Strategy

1. **We don't need to prevent learning.** Unlike Glaze/Mist, we WANT models to learn the artist's style. Our watermark rides along with the training data.

2. **Our watermark is orthogonal to purification attacks.** The paper's bypass methods target adversarial gradient signals. Our frequency-domain watermark occupies a different part of the signal space.

3. **Multi-layer redundancy matters.** Layers 1+2 survive pixel-space attacks; Layer 3 (ghost) survives latent-space attacks (VAE). No single layer needs to survive everything.

4. **The training ghost signal (Layer 3) addresses the remaining gap.** For the scenario where an image goes through full model training and generation, the ghost signal is designed to propagate spectral bias into the model's outputs, enabling batch-level detection even when individual-image detection fails.

## Test Coverage

~1,900 tests (+ GPU/slow tests excluded from default runs) covering:

### Test Matrix
All core tests are parameterized across:
- **7 image types**: gradient, texture, edges, highfreq, photo_like, dark, natural_scene
- **3 author keys**: key-alpha, key-bravo, key-charlie (different seeds)

This gives 21 (image × key) combinations per test, catching key-dependent and content-dependent failures that single-image/single-key tests miss.

### Image Categories
Tests distinguish between **realistic** and **pathological** images:
- **Realistic** (texture, photo_like, natural_scene): Full detection expected — detected=True, author_id_match=True, confidence>0.5
- **Pathological** (gradient, edges, highfreq, dark): Relaxed thresholds — these extreme synthetic images lack spectral diversity or dynamic range for reliable full-pipeline watermarking but are included to document known limitations

### Coverage Areas
- Roundtrip embedding/detection (grayscale + RGB, all image types × all keys)
- JPEG compression quality sweep (Q20-Q99, grayscale + RGB) across realistic image types
- Sequential re-encoding (2x, 3x, 5x at Q75/Q90; JPEG→PNG→JPEG; descending Q99→Q85→Q75)
- JPEG before VAE pipeline (Q50/Q75/Q90/Q99 → SD VAE, order comparison, multi-JPEG → VAE)
- Cropping (10%, 25%, 50%)
- Rotation (1, 2, 5, 90, 180, 270 degrees)
- Scaling (0.75x downscale/upscale)
- Gaussian noise (sigma 5-65) across realistic image types
- Brightness/contrast adjustments across realistic image types
- Perspective/affine transforms
- Print-scan simulation
- Median filter, histogram equalization, gamma correction
- Bit depth reduction
- Color attacks (hue rotation, desaturation, white balance, vignetting)
- Combined attack chains (3/4/5-step, repeated, ordering)
- 18 diverse procedurally generated image types
- False positive verification (random keys, wrong keys, patterns, confidence distributions)
- Analytical false positive rate verification (< 1e-30 with RS, < 1e-6 fallback)
- Social media simulation (Instagram, Twitter, Facebook, WhatsApp, Pinterest, TikTok)
- Small/non-square/odd-dimension images (64x64 to 1024x1024)
- Configuration sensitivity (strength, wavelet, subbands, spreading factor)
- Ghost signal robustness (JPEG, noise, blur, brightness, contrast, gamma) across all image types × keys
- Ghost hash extraction with known error thresholds per image type
- Ghost key discrimination across all image types
- Tile destruction and partial corruption
- Purification attacks from Honig et al. (Gaussian noising, DiffPure, noisy upscaling)
- Training pipeline simulation (augmentation, VAE bottleneck, multi-epoch)
- Superresolution simulation
- Targeted removal attacks (ring notch filter, PN subtraction, ghost band removal, combined, quality-constrained)
- Natural image ring detection (1/f photos, portraits, dark images, high contrast, anisotropic landscapes)
- Spectral whitening discrimination (watermarked vs unwatermarked gap verification)
- Cross-key false positive verification (20 keys, no cross-detection) on realistic images
- Production JPEG/PNG roundtrip mirroring the exact encode pipeline
- Re-encoding robustness (test_reencoding.py): JPEG Q20-Q99 sweep, PNG, sequential, cross-codec chains
- Real photographs & artwork (test_real_photos.py): 8 public domain images (paintings + photos), RGB, multi-key, PSNR/detection/JPEG/ghost, comprehensive quality report
- Quality metrics (PSNR > 37 dB, max deviation < 30) across all image types

### Test Infrastructure

**Shared fixtures** (`conftest.py`):
- Vectorized image generators (numpy-only, no pixel-by-pixel loops)
- `NATURAL_IMAGE_GENERATORS` dict registry for parameterization
- `PATHOLOGICAL_IMAGES` set for known-limitation documentation
- `realistic_image` fixture for tests requiring strong detection
- `natural_image` fixture for broad coverage including edge cases
- `multi_author_keys` fixture for multi-key testing (3 keys per test)
- Shared JPEG/PNG roundtrip utilities

**Parallel execution** (`pytest-xdist`):
- `uv run pytest -n auto` runs tests across all CPU cores
- ~2.5x speedup with 4 workers (e.g., 2:23 vs ~5+ minutes sequential)
- Tests are fully independent (no shared mutable state)

**Running tests:**
```bash
# Default (excludes GPU/slow):
uv run python -m pytest tests/

# Parallel:
uv run python -m pytest tests/ -n auto

# Single file:
uv run python -m pytest tests/test_pipeline.py -v

# GPU tests (requires torch/diffusers):
uv run python -m pytest -m gpu
```

GPU/slow tests (run with `uv run pytest -m gpu` or `uv run pytest -m slow`):
- Real SD VAE roundtrip attack (`test_real_vae_attack.py`)
- Ghost strength sweep across VAE (`test_ghost_strength_sweep.py`)
- LoRA ghost propagation (`test_ghost_lora_propagation.py`)
- LoRA hyperparameter sweep (`test_lora_hyperparameter_sweep.py`)

### Known Limitations (documented by tests)
- **Gradient images**: Ghost hash extraction unreliable (3+ bit errors vs 2-error threshold). Ghost confidence low. Pathological: pure diagonal gradient lacks spectral diversity for ghost PN correlation.
- **Edge images**: Binary content (240/15 values) causes ghost hash failure and beacon non-detection on some keys. Saturation clips reduce effective dynamic range.
- **High-frequency images**: Dense noise interferes with ring detection and beacon. Author ID match fails on some key combinations.
- **Dark images**: Very low dynamic range (30±15) makes watermark fragile under noise. Sigma=5 Gaussian noise can break detection.
- **JPEG Q50**: Author ID match degrades across multiple image types. Detection survives but cryptographic verification weakens.

## Usage

```python
from sigil_watermark.keygen import generate_author_keys
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.detect import SigilDetector

# Generate keys (once per artist)
keys = generate_author_keys()

# Embed
embedder = SigilEmbedder()
watermarked = embedder.embed(image, keys)  # image: np.ndarray (H,W) or (H,W,3)

# Detect
detector = SigilDetector()
result = detector.detect(watermarked, keys.public_key)
assert result.detected
assert result.author_id_match

# Tampering check
if result.tampering_suspected:
    print("WARNING: Targeted ring removal detected")
```

## References

The robustness testing and comparative analysis in this project was inspired by:

> Robert Honig, Javier Rando, Nicholas Carlini, and Florian Tramer. "Adversarial Perturbations Cannot Reliably Protect Artists From Generative AI." In *The Thirteenth International Conference on Learning Representations (ICLR)*, 2025. https://openreview.net/forum?id=yfW1x7uBS5

Their finding that adversarial perturbations (Glaze, Mist, Anti-DreamBooth) are fundamentally brittle against simple preprocessing informed our decision to pursue frequency-domain watermarking rather than adversarial protection. See the "Comparison with Adversarial Perturbation Approaches" section and `tests/test_purification_attacks.py` for details.
