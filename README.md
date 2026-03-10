# Sigil Watermark

[![PyPI version](https://img.shields.io/pypi/v/sigil-watermark.svg)](https://pypi.org/project/sigil-watermark/)
[![CI](https://github.com/AI-Escape/sigil-watermark/actions/workflows/ci.yml/badge.svg)](https://github.com/AI-Escape/sigil-watermark/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Invisible, crypto-verified, AI-training-resilient image watermarks. A pure signal-processing system — no neural networks, no training infrastructure, no GPU required.

## Features

- **Invisible**: 40+ dB PSNR on real photographs and artwork (adaptive embedding strength)
- **Crypto-verified**: Ed25519 keypairs with HKDF-derived parameters — each watermark is cryptographically unique
- **Three independent layers**: DFT ring anchor, DWT spread-spectrum payload, ghost spectral signal
- **Robust**: Survives JPEG compression (Q60+), resize, crop, rotation, noise, bilateral filtering
- **VAE-resilient**: Ghost signal layer survives Stable Diffusion VAE encode/decode
- **False positive rate**: < 10^-30 (analytically proven, empirically validated)
- **Fast**: Pure NumPy/SciPy — embeds a 1024x1024 image in ~200ms on CPU
- **Typed**: Full PEP 561 compliance with type annotations

## Installation

```bash
pip install sigil-watermark
```

## Quick Start

```python
import numpy as np
from sigil_watermark import SigilEmbedder, SigilDetector, generate_author_keys

# Generate author keypair (or derive from a seed for reproducibility)
keys = generate_author_keys()

# Load your image as a float64 array (0-255), grayscale or RGB
image = np.array(Image.open("artwork.png").convert("RGB"), dtype=np.float64)

# Embed watermark
embedder = SigilEmbedder()
watermarked = embedder.embed(image, keys)

# Save (the watermark survives JPEG, PNG, and other formats)
Image.fromarray(np.clip(watermarked, 0, 255).astype(np.uint8)).save("watermarked.png")

# Detect watermark
detector = SigilDetector()
result = detector.detect(watermarked, keys.public_key)

print(f"Detected: {result.detected}")           # True
print(f"Author match: {result.author_id_match}") # True
print(f"Confidence: {result.confidence:.3f}")     # ~0.7
```

## Architecture

Sigil embeds three independent watermark layers, each targeting a different attack class:

```
                    ┌─────────────────────────────────┐
                    │         Input Image              │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  Layer 1: DFT Ring Anchor        │
                    │  8 concentric frequency rings     │
                    │  (key-derived + sentinel + content)│
                    │  Survives: rotation, crop, scale  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  Layer 2: DWT Spread-Spectrum    │
                    │  RS-encoded payload in wavelets   │
                    │  Tiled for crop robustness        │
                    │  Survives: JPEG, noise, resize    │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │  Layer 3: Ghost Spectral Signal  │
                    │  Multiplicative modulation at     │
                    │  VAE-transparent frequency bands   │
                    │  Survives: VAE encode/decode      │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │       Watermarked Image           │
                    │       PSNR: 35-47 dB              │
                    └─────────────────────────────────┘
```

Detection is three-tier:

1. **Beacon** — "Is this a Sigil watermark?" (universal marker)
2. **Author Index** — "Whose watermark is it?" (20-bit index for blind scanning)
3. **Author Verification** — "Cryptographic proof of authorship" (48-bit author ID)

## Documentation

- [Getting Started](https://ai-escape.github.io/sigil-watermark/getting-started/) — Tutorial with examples
- [API Reference](https://ai-escape.github.io/sigil-watermark/api/) — Full API documentation
- [Technical Design](https://ai-escape.github.io/sigil-watermark/technical-design/) — Deep dive into the signal processing

## Development

```bash
git clone https://github.com/AI-Escape/sigil-watermark.git
cd sigil-watermark
uv sync --extra dev
uv run pytest              # ~1,900 tests (GPU/slow excluded by default)
uv run pytest -n auto      # parallel execution
uv run ruff check .        # lint
uv run ruff format --check # format check
```

### GPU tests

Some tests require a GPU with PyTorch and diffusers installed:

```bash
uv sync --extra gpu
uv run pytest -m gpu       # real SD VAE encode/decode, LoRA propagation
```

### Building docs locally

```bash
uv sync --extra docs
uv run mkdocs serve        # http://127.0.0.1:8000
```

## Origin

Sigil Watermark was originally developed as part of the [Signarture](https://signarture.ai) artist protection platform.

## License

Apache-2.0 — see [LICENSE](LICENSE) for details.

