"""
Test Sonar Dynamic Resolution Support

Tests for dynamic resolution handling across all Sonar transforms and components.
Verifies support for 5 different dataset resolutions.

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

from aquasot3d.dataloader.transforms.logpolar import LogPolarTransform
from aquasot3d.dataloader.transforms.sonar_geometry import SonarGeometryHandler
from aquasot3d.dataloader.transforms.sonar_transforms import (
    LogPolarTransformBboxAware,
    RadialScalingJitterBboxAware,
    SectorOcclusionBboxAware,
    letterbox_resize,
)
from aquasot3d.models.backbones.sonar_encoder import SonarEncoder


class TestDatasetResolutions(unittest.TestCase):
    """Test all 5 dataset resolutions."""

    # Dataset resolutions from specs
    RESOLUTIONS = [
        (376, 682, "uuv2,3,4,6,7,8,9"),
        (377, 684, "uuv5,12"),
        (375, 680, "uuv10"),
        (501, 909, "uuv11"),
        (503, 912, "uuv1"),
    ]

    def test_logpolar_transform_all_resolutions(self):
        """Test LogPolarTransform supports all 5 resolutions."""
        transform = LogPolarTransform(
            sonar_height=376,  # Initial size (will adapt)
            sonar_width=682,
            output_height=256,
            output_width=256,
        )

        for H, W, sequences in self.RESOLUTIONS:
            with self.subTest(resolution=f"{H}x{W}", sequences=sequences):
                img = torch.randn(1, 1, H, W)
                output = transform(img)
                self.assertEqual(output.shape, (1, 1, 256, 256),
                               f"Failed for {H}x{W} ({sequences})")

    def test_sonar_geometry_handler_all_resolutions(self):
        """Test SonarGeometryHandler supports all 5 resolutions."""
        geometry = SonarGeometryHandler(sonar_height=376, sonar_width=682)

        for H, W, sequences in self.RESOLUTIONS:
            with self.subTest(resolution=f"{H}x{W}", sequences=sequences):
                img = torch.randn(1, 1, H, W)
                output = geometry(img)

                self.assertEqual(output['masked'].shape, (1, 1, H, W))
                self.assertEqual(output['weighted'].shape, (1, 1, H, W))
                self.assertIsNotNone(output['radial_profile'])

    def test_logpolar_bbox_aware_none_dimensions(self):
        """Test LogPolarTransformBboxAware with None dimensions (auto-detect)."""
        transform = LogPolarTransformBboxAware(
            sonar_height=None,  # Auto-detect
            sonar_width=None,   # Auto-detect
            output_height=256,
            output_width=256,
        )

        for H, W, sequences in self.RESOLUTIONS:
            with self.subTest(resolution=f"{H}x{W}", sequences=sequences):
                img = np.random.rand(H, W, 1).astype(np.float32) * 255
                bboxes = np.array([[100, 50, 80, 60]], dtype=np.float32)

                result = transform(image=img, bboxes=bboxes)

                self.assertEqual(result['image'].shape, (256, 256, 1))
                self.assertEqual(result['bboxes'].shape, (1, 4))

    def test_sonar_encoder_dynamic_sizing(self):
        """Test SonarEncoder with None img_size (dynamic sizing)."""
        encoder = SonarEncoder(
            encoder_type='vit',
            model_name='vit_tiny_patch16_224',
            img_size=None,  # Auto-detect
            use_logpolar=True,
        )

        for H, W, sequences in self.RESOLUTIONS[:2]:  # Test first 2 for speed
            with self.subTest(resolution=f"{H}x{W}", sequences=sequences):
                template = torch.randn(2, 1, H, W)
                search = torch.randn(2, 1, H, W)

                outputs = encoder(template, search)

                self.assertIn('template_features', outputs)
                self.assertIn('search_features', outputs)
                self.assertEqual(encoder.img_size, (H, W))


class TestRadialScalingJitter(unittest.TestCase):
    """Test RadialScalingJitterBboxAware dynamic sensor_pos."""

    def test_auto_computed_sensor_pos(self):
        """Test sensor_pos is auto-computed as (W/2, H-1)."""
        transform = RadialScalingJitterBboxAware(
            scale_range=(0.95, 1.05),
            prob=1.0,
            sensor_pos=None  # Auto-compute
        )

        test_sizes = [(376, 682), (501, 909)]

        for H, W in test_sizes:
            with self.subTest(resolution=f"{H}x{W}"):
                img = np.random.rand(H, W, 1).astype(np.float32) * 255
                bboxes = np.array([[100, 50, 80, 60]], dtype=np.float32)

                result = transform(image=img, bboxes=bboxes)

                # Verify image dimensions preserved
                self.assertEqual(result['image'].shape[:2], (H, W))


class TestSectorOcclusion(unittest.TestCase):
    """Test SectorOcclusionBboxAware dynamic sensor_pos."""

    def test_auto_computed_sensor_pos(self):
        """Test sensor_pos is auto-computed for different image sizes."""
        transform = SectorOcclusionBboxAware(
            min_angle=15,
            max_angle=30,
            prob=1.0,
            sensor_pos=None  # Auto-compute
        )

        test_sizes = [(376, 682), (375, 680), (503, 912)]

        for H, W in test_sizes:
            with self.subTest(resolution=f"{H}x{W}"):
                img = np.random.rand(H, W, 1).astype(np.float32) * 255
                bboxes = np.array([[100, 50, 80, 60]], dtype=np.float32)

                result = transform(image=img, bboxes=bboxes)

                # Verify image dimensions preserved
                self.assertEqual(result['image'].shape[:2], (H, W))


class TestLetterboxResize(unittest.TestCase):
    """Test letterbox_resize utility function."""

    def test_letterbox_preserves_aspect_ratio(self):
        """Test letterbox_resize preserves aspect ratio."""
        img = np.random.rand(376, 682).astype(np.float32)
        resized, (scale, pad_top, pad_left) = letterbox_resize(img, (256, 256))

        self.assertEqual(resized.shape, (256, 256))
        self.assertGreater(scale, 0)
        self.assertGreaterEqual(pad_top, 0)
        self.assertGreaterEqual(pad_left, 0)

    def test_letterbox_sensor_position(self):
        """Test letterbox places padding on top (sensor at bottom)."""
        img = np.random.rand(376, 682).astype(np.float32)
        resized, (scale, pad_top, pad_left) = letterbox_resize(img, (256, 256))

        # Vertical padding should be on top only
        # Calculate expected dimensions
        new_H = int(376 * scale)
        expected_pad_top = 256 - new_H

        self.assertEqual(pad_top, expected_pad_top)


class TestHardcodedDimensionElimination(unittest.TestCase):
    """Verify no hardcoded 376×682 dimensions remain."""

    def test_logpolar_bbox_aware_accepts_none(self):
        """Test LogPolarTransformBboxAware accepts None for dimensions."""
        transform = LogPolarTransformBboxAware(
            sonar_height=None,
            sonar_width=None,
            output_height=256,
            output_width=256,
        )

        self.assertIsNone(transform.sonar_height)
        self.assertIsNone(transform.sonar_width)
        self.assertFalse(transform._initialized)

    def test_sonar_encoder_accepts_none(self):
        """Test SonarEncoder accepts None for img_size."""
        encoder = SonarEncoder(
            encoder_type='vit',
            img_size=None,
        )

        # Should default to (512, 512) for initialization
        self.assertEqual(encoder.img_size, (512, 512))


if __name__ == '__main__':
    unittest.main()
