#!/usr/bin/env python3
"""Download public domain photographs and artwork for watermark testing.

Downloads a diverse set of real images to tests/test_images/real/:
- Paintings (various styles, periods, textures)
- Photographs (landscapes, portraits, urban, nature)
- Various resolutions and aspect ratios

All images are public domain (PD-old, PD-US, CC0) from Wikimedia Commons.

Usage:
    uv run python scripts/download_real_images.py
"""

import sys
import urllib.request
from pathlib import Path

# Each entry: (filename, url, description)
# All URLs are from Wikimedia Commons for public domain works.
IMAGES = [
    # === Paintings ===
    (
        "starry_night.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/"
        "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/"
        "1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "Van Gogh - Starry Night (1889). Rich swirling texture, mixed frequencies.",
    ),
    (
        "great_wave.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/"
        "Tsunami_by_hokusai_19th_century.jpg/"
        "1280px-Tsunami_by_hokusai_19th_century.jpg",
        "Hokusai - The Great Wave (1831). High contrast woodblock print.",
    ),
    (
        "girl_pearl_earring.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/"
        "Meisje_met_de_parel.jpg/"
        "800px-Meisje_met_de_parel.jpg",
        "Vermeer - Girl with a Pearl Earring (1665). Smooth tonal gradations.",
    ),
    (
        "mona_lisa.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/"
        "Mona_Lisa.jpg/"
        "800px-Mona_Lisa.jpg",
        "Da Vinci - Mona Lisa (1503). Sfumato technique, subtle gradients.",
    ),
    (
        "water_lilies.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/"
        "Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg/"
        "1280px-Claude_Monet_-_Water_Lilies_-_1906%2C_Ryerson.jpg",
        "Monet - Water Lilies (1906). Impressionist brushwork, organic texture.",
    ),
    (
        "persistence_of_memory.jpg",
        "https://upload.wikimedia.org/wikipedia/en/d/dd/The_Persistence_of_Memory.jpg",
        "Dali - Persistence of Memory (1931). Surreal, smooth + detailed.",
    ),
    # === Photographs ===
    (
        "photo_landscape.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/"
        "Tunnel_View%2C_Yosemite_Valley%2C_Yosemite_NP_-_Diliff.jpg/"
        "1280px-Tunnel_View%2C_Yosemite_Valley%2C_Yosemite_NP_-_Diliff.jpg",
        "Yosemite Valley landscape. Mountains, trees, sky — strong anisotropy.",
    ),
    (
        "photo_portrait.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/"
        "Migrant_Mother_by_Dorothea_Lange.jpg/"
        "800px-Migrant_Mother_by_Dorothea_Lange.jpg",
        "Dorothea Lange - Migrant Mother (1936). Portrait, grainy texture.",
    ),
    (
        "photo_urban.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
        "New_york_times_square-terabass.jpg/"
        "1280px-New_york_times_square-terabass.jpg",
        "Times Square, NYC. High detail, bright colors, mixed content.",
    ),
    (
        "photo_nature.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/"
        "Macaw_at_Chester_Zoo.jpg/"
        "1280px-Macaw_at_Chester_Zoo.jpg",
        "Macaw parrot. Fine feather detail, saturated colors.",
    ),
    (
        "photo_architecture.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/"
        "All_Gizah_Pyramids.jpg/"
        "1280px-All_Gizah_Pyramids.jpg",
        "Pyramids of Giza. Strong geometric edges, desert texture.",
    ),
    (
        "photo_dark.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/"
        "Hubble_ultra_deep_field.jpg/"
        "1024px-Hubble_ultra_deep_field.jpg",
        "Hubble Ultra Deep Field. Very dark, faint point sources.",
    ),
]


def download_image(url: str, dest: Path) -> bool:
    """Download an image, return True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SigilWatermarkTest/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    output_dir = Path(__file__).parent.parent / "tests" / "test_images" / "real"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(IMAGES)} public domain images to {output_dir}/\n")

    success = 0
    for filename, url, description in IMAGES:
        dest = output_dir / filename
        if dest.exists():
            print(f"  [skip] {filename} (already exists)")
            success += 1
            continue

        print(f"  [download] {filename}")
        print(f"    {description}")
        if download_image(url, dest):
            size_kb = dest.stat().st_size / 1024
            print(f"    OK ({size_kb:.0f} KB)")
            success += 1

    print(f"\nDone: {success}/{len(IMAGES)} images available.")
    if success < len(IMAGES):
        print("Some downloads failed. Re-run to retry, or add images manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()
