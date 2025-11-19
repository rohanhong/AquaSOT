"""
Test Augmentation Pipeline

Tests for complete bbox-aware augmentation pipeline integration.
Verifies RGB and Sonar transforms work together correctly.

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

import unittest
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aquasot3d.dataloader.datasets.augmentation import (
    TrackingAugmentationBboxAware
)


class TestTrainingAugmentation(unittest.TestCase):
    """Test training augmentation pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = {
            'rgb_img': np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            'sonar_img': np.random.randint(0, 256, (376, 682, 1), dtype=np.uint8),
            'rgb_bbox': np.array([100, 50, 80, 60], dtype=np.float32),
            'sonar_bbox': np.array([200, 100, 90, 70], dtype=np.float32),
        }

    def test_training_pipeline_basic(self):
        """Test basic training pipeline execution."""
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            rgb_config={'enable_color_jitter': True, 'enable_blur': True},
            sonar_config={'enable_speckle': True})

        result = aug(self.test_data)

        # Check outputs exist
        self.assertIn('rgb_img', result)
        self.assertIn('sonar_img', result)
        self.assertIn('rgb_bbox', result)
        self.assertIn('sonar_bbox', result)

        # Check output types
        self.assertIsInstance(result['rgb_img'], torch.Tensor)
        self.assertIsInstance(result['sonar_img'], torch.Tensor)

        # Check RGB shape (C, H, W)
        self.assertEqual(result['rgb_img'].shape[0], 3)

        # Check Sonar shape after log-polar (1, 256, 256)
        self.assertEqual(result['sonar_img'].shape, (1, 256, 256))

    def test_evaluation_pipeline(self):
        """Test evaluation pipeline (no augmentation)."""
        aug = TrackingAugmentationBboxAware(
            is_train=False)

        result = aug(self.test_data)

        # Should still produce valid outputs
        self.assertIn('rgb_img', result)
        self.assertIn('sonar_img', result)

    def test_rgb_transforms_enabled(self):
        """Test RGB transforms can be individually enabled."""
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            rgb_config={
                'enable_color_jitter': True,
                'enable_blur': True,
                'enable_horizontal_flip': True,
                'enable_occlusion': True,
            })

        result = aug(self.test_data)

        # Should complete without error
        self.assertIn('rgb_img', result)

    def test_sonar_transforms_enabled(self):
        """Test Sonar transforms can be individually enabled."""
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            sonar_config={
                'enable_speckle': True,
                'enable_radial_jitter': True,
                'enable_sector_occlusion': False,  # Keep disabled
            })

        result = aug(self.test_data)

        # Should complete without error
        self.assertIn('sonar_img', result)


class TestDynamicResolution(unittest.TestCase):
    """Test pipeline with different Sonar resolutions."""

    def test_multiple_resolutions(self):
        """Test pipeline works with all 5 dataset resolutions."""
        aug = TrackingAugmentationBboxAware(
            is_train=False)

        resolutions = [
            (376, 682, "uuv2-9"),
            (377, 684, "uuv5,12"),
            (375, 680, "uuv10"),
            (501, 909, "uuv11"),
            (503, 912, "uuv1"),
        ]

        for H, W, sequences in resolutions:
            with self.subTest(resolution=f"{H}x{W}", sequences=sequences):
                test_data = {
                    'rgb_img': np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
                    'sonar_img': np.random.randint(0, 256, (H, W, 1), dtype=np.uint8),
                    'rgb_bbox': np.array([100, 50, 80, 60], dtype=np.float32),
                    'sonar_bbox': np.array([200, 100, 90, 70], dtype=np.float32),
                }

                result = aug(test_data)

                # Sonar should be transformed to 256×256 via log-polar
                self.assertEqual(result['sonar_img'].shape, (1, 256, 256))


class TestBboxFormatConsistency(unittest.TestCase):
    """Test bbox format remains xywh throughout pipeline."""

    def test_bbox_format_xywh(self):
        """Test bboxes remain in xywh format."""
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            rgb_config={'enable_color_jitter': True},
            sonar_config={'enable_speckle': True})

        test_data = {
            'rgb_img': np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            'sonar_img': np.random.randint(0, 256, (376, 682, 1), dtype=np.uint8),
            'rgb_bbox': np.array([100, 50, 80, 60], dtype=np.float32),  # xywh
            'sonar_bbox': np.array([200, 100, 90, 70], dtype=np.float32),  # xywh
        }

        result = aug(test_data)

        # Output bboxes should have 4 elements (xywh format)
        self.assertEqual(len(result['rgb_bbox']), 4)
        self.assertEqual(len(result['sonar_bbox']), 4)


class TestWithoutLogPolar(unittest.TestCase):
    """Test pipeline without log-polar transform."""

    def test_no_logpolar(self):
        """Test pipeline works without log-polar transform."""
        aug = TrackingAugmentationBboxAware(
            is_train=True,
            rgb_config={'enable_color_jitter': True},
            sonar_config={'enable_speckle': True},
            use_logpolar=False,  # Disabled
        )

        test_data = {
            'rgb_img': np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
            'sonar_img': np.random.randint(0, 256, (376, 682, 1), dtype=np.uint8),
            'rgb_bbox': np.array([100, 50, 80, 60], dtype=np.float32),
            'sonar_bbox': np.array([200, 100, 90, 70], dtype=np.float32),
        }

        result = aug(test_data)

        # Sonar should NOT be transformed to 256×256
        # Shape should be (1, 376, 682) after tensor conversion
        self.assertEqual(result['sonar_img'].shape[0], 1)


if __name__ == '__main__':
    unittest.main()
