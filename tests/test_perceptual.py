"""Tests for the perceptual masking module."""

import numpy as np
import pytest

from sigil_watermark.perceptual import compute_perceptual_mask
from sigil_watermark.config import SigilConfig


@pytest.fixture
def config():
    return SigilConfig()


class TestPerceptualMask:
    def test_output_shape_matches_input(self, medium_gray_image, config):
        mask = compute_perceptual_mask(medium_gray_image, config=config)
        assert mask.shape == medium_gray_image.shape

    def test_values_non_negative(self, medium_gray_image, config):
        mask = compute_perceptual_mask(medium_gray_image, config=config)
        assert np.all(mask >= 0)

    def test_minimum_floor(self, flat_image, config):
        """Even flat regions should have at least mask_floor strength."""
        mask = compute_perceptual_mask(flat_image, config=config)
        assert np.all(mask >= config.mask_floor - 1e-6)

    def test_textured_regions_higher_than_flat(self, config):
        """Textured regions should allow stronger embedding than flat ones."""
        img = np.zeros((256, 256), dtype=np.float64)
        # Left half: flat
        img[:, :128] = 128
        # Right half: textured
        rng = np.random.default_rng(42)
        img[:, 128:] = 128 + rng.normal(0, 30, (256, 128))
        img = np.clip(img, 0, 255)

        mask = compute_perceptual_mask(img, config=config)
        flat_mean = mask[:, :128].mean()
        textured_mean = mask[:, 128:].mean()
        assert textured_mean > flat_mean, (
            f"Textured mean ({textured_mean:.3f}) should be > flat mean ({flat_mean:.3f})"
        )

    def test_deterministic(self, medium_gray_image, config):
        m1 = compute_perceptual_mask(medium_gray_image, config=config)
        m2 = compute_perceptual_mask(medium_gray_image, config=config)
        np.testing.assert_array_equal(m1, m2)

    def test_scales_with_local_variance(self, config):
        """Mask should be proportional to local image activity."""
        rng = np.random.default_rng(42)
        # Create image with known variance regions
        img = np.zeros((256, 256), dtype=np.float64)
        img[:128, :] = 128  # Low variance
        img[128:, :] = 128 + rng.normal(0, 50, (128, 256))  # High variance
        img = np.clip(img, 0, 255)

        mask = compute_perceptual_mask(img, config=config)
        low_var_mask = mask[:128, :].mean()
        high_var_mask = mask[128:, :].mean()
        assert high_var_mask > low_var_mask * 1.5

    def test_works_on_various_sizes(self, config):
        for size in [(128, 128), (256, 512), (512, 256), (1024, 1024)]:
            img = np.random.default_rng(42).normal(128, 30, size)
            img = np.clip(img, 0, 255)
            mask = compute_perceptual_mask(img, config=config)
            assert mask.shape == size
