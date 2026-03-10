# Sigil Watermark

Invisible, crypto-verified, AI-training-resilient image watermarks. A pure signal-processing system -- no neural networks, no training infrastructure, no GPU required.

## Features

- **Invisible**: 40+ dB PSNR on real photographs and artwork
- **Crypto-verified**: Ed25519 keypairs with HKDF-derived parameters
- **Three independent layers**: DFT ring anchor, DWT spread-spectrum payload, ghost spectral signal
- **Robust**: Survives JPEG (Q60+), resize, crop, rotation, noise, bilateral filtering
- **VAE-resilient**: Ghost signal layer survives Stable Diffusion VAE encode/decode
- **False positive rate**: < 10^-30 (analytically proven)
- **Fast**: Pure NumPy/SciPy -- embeds a 1024x1024 image in ~200ms on CPU

## Installation

```bash
pip install sigil-watermark
```

## Quick Start

```python
import numpy as np
from PIL import Image
from sigil_watermark import SigilEmbedder, SigilDetector, generate_author_keys

# Generate author keypair
keys = generate_author_keys()

# Load image as float64 (0-255), grayscale or RGB
image = np.array(Image.open("artwork.png").convert("RGB"), dtype=np.float64)

# Embed watermark
embedder = SigilEmbedder()
watermarked = embedder.embed(image, keys)

# Detect watermark
detector = SigilDetector()
result = detector.detect(watermarked, keys.public_key)

print(f"Detected: {result.detected}")            # True
print(f"Author match: {result.author_id_match}")  # True
```

See the [Getting Started](getting-started.md) guide for more examples, or the [API Reference](api/index.md) for full documentation.
