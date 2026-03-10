"""Sigil Watermark -- invisible, crypto-verified, AI-training-resilient image watermarks.

A pure signal-processing watermark system with three independent embedding layers
and three-tier detection. No neural networks required.

Example:
    >>> from sigil_watermark import SigilEmbedder, SigilDetector, generate_author_keys
    >>> keys = generate_author_keys()
    >>> embedder = SigilEmbedder()
    >>> detector = SigilDetector()
    >>> watermarked = embedder.embed(image, keys)
    >>> result = detector.detect(watermarked, keys.public_key)
    >>> result.detected
    True
"""

__version__ = "0.2.0"

from sigil_watermark.config import DEFAULT_CONFIG, SigilConfig
from sigil_watermark.detect import DetectionResult, SigilDetector
from sigil_watermark.embed import SigilEmbedder
from sigil_watermark.keygen import AuthorKeys, generate_author_keys

__all__ = [
    "__version__",
    "AuthorKeys",
    "DEFAULT_CONFIG",
    "DetectionResult",
    "SigilConfig",
    "SigilDetector",
    "SigilEmbedder",
    "generate_author_keys",
]
