# Getting Started

## Installation

```bash
pip install sigil-watermark
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add sigil-watermark
```

## Key Generation

Every author needs an Ed25519 keypair. The public key is shared openly; the private key is kept secret.

```python
from sigil_watermark import generate_author_keys

# Random keypair
keys = generate_author_keys()

# Deterministic keypair from a 32-byte seed (reproducible)
keys = generate_author_keys(seed=b"my-secret-seed-exactly-32-bytes!")

# Access the raw key bytes
print(len(keys.public_key))   # 32
print(len(keys.private_key))  # 32
```

## Embedding a Watermark

### RGB Images

```python
import numpy as np
from PIL import Image
from sigil_watermark import SigilEmbedder, generate_author_keys

keys = generate_author_keys()
embedder = SigilEmbedder()

# Load as float64 RGB (0-255)
image = np.array(Image.open("photo.jpg").convert("RGB"), dtype=np.float64)

# Embed -- returns float64 array same shape as input
watermarked = embedder.embed(image, keys)

# Save
output = np.clip(watermarked, 0, 255).astype(np.uint8)
Image.fromarray(output).save("watermarked.png")
```

### Grayscale Images

```python
image = np.array(Image.open("drawing.png").convert("L"), dtype=np.float64)
watermarked = embedder.embed(image, keys)  # works with 2D arrays too
```

### Image Requirements

- **Dtype**: `float64`, values in `[0, 255]`
- **Shape**: `(H, W)` for grayscale or `(H, W, 3)` for RGB
- **Minimum size**: 64x64 (larger images produce stronger watermarks)
- **Even dimensions**: Recommended for best DWT performance (odd dims are handled but may reduce quality)

## Detecting a Watermark

```python
from sigil_watermark import SigilDetector

detector = SigilDetector()

# Detect with a candidate public key
result = detector.detect(watermarked, keys.public_key)
```

### Understanding DetectionResult

```python
result.detected            # bool: watermark found?
result.author_id_match     # bool: author ID matches the provided key?
result.confidence          # float: overall confidence (0-1)
result.ring_confidence     # float: Layer 1 (DFT rings) signal strength
result.payload_confidence  # float: Layer 2 (DWT payload) signal strength
result.ghost_confidence    # float: Layer 3 (ghost signal) strength
result.beacon_found        # bool: universal Sigil beacon detected?
result.tampering_suspected # bool: sentinel rings intact but key rings removed?
result.ghost_hash          # list[int] | None: extracted ghost hash bits
result.ghost_hash_match    # bool: ghost hash matches candidate key?
```

A typical detection on an uncompressed watermarked image:

| Field | Value |
|-------|-------|
| `detected` | `True` |
| `author_id_match` | `True` |
| `confidence` | `0.7` |
| `ring_confidence` | `0.3-0.8` |
| `payload_confidence` | `0.85-1.0` |
| `ghost_confidence` | `0.3-0.6` |

## JPEG Robustness

The watermark survives standard web compression:

```python
import io
from PIL import Image

# Embed
watermarked = embedder.embed(image, keys)

# JPEG roundtrip at Q75
buf = io.BytesIO()
Image.fromarray(np.clip(watermarked, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=75)
buf.seek(0)
compressed = np.array(Image.open(buf).convert("RGB"), dtype=np.float64)

# Still detectable
result = detector.detect(compressed, keys.public_key)
print(result.detected)          # True
print(result.author_id_match)   # True
```

## Custom Configuration

```python
from sigil_watermark import SigilConfig, SigilEmbedder, SigilDetector

config = SigilConfig(
    ring_strength=25.0,           # Stronger rings (default: 20.0)
    embed_strength=4.0,           # Stronger payload (default: 3.0)
    adaptive_ring_strength=True,  # Adapt to image content (default: True)
    ring_target_psnr=38.0,        # Higher quality target (default: 36.0)
)

# Both embedder and detector must use the same config
embedder = SigilEmbedder(config=config)
detector = SigilDetector(config=config)
```

See the [Configuration API](api/config.md) for all available parameters.
