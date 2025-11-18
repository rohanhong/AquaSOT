"""
Test RGB Bbox-Aware Transforms

Tests for rgb_transforms.py bbox-aware transform implementations.
Verifies bbox synchronization, format handling, and backward compatibility.

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

from aquasot3d.dataloader.transforms.rgb_transforms import (
    ColorJitterBboxAware,
    RandomBlurBboxAware,
    NormalizeBboxAware,
    RandomHorizontalFlipBboxAware,
    RandomVerticalFlipBboxAware,
    RandomOcclusionBboxAware,
    ToTensor,
    # Test backward compatibility aliases
    ColorJitter,
    RandomBlur,
    Normalize,
)


class TestPhotometricTransforms(unittest.TestCase):
    """Test photometric transforms (bbox-preserving)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        self.test_bboxes = np.array([
            [100, 50, 80, 60],
            [300, 200, 100, 80],
        ], dtype=np.float32)

    def test_color_jitter_preserves_bboxes(self):
        """Test ColorJitterBboxAware preserves bbox coordinates."""
        transform = ColorJitterBboxAware(brightness=0.2, prob=1.0)
        result = transform(image=self.test_img, bboxes=self.test_bboxes)

        self.assertIn('image', result)
        self.assertIn('bboxes', result)
        np.testing.assert_array_equal(result['bboxes'], self.test_bboxes)

    def test_random_blur_preserves_bboxes(self):
        """Test RandomBlurBboxAware preserves bbox coordinates."""
        transform = RandomBlurBboxAware(kernel_sizes=[3, 5], prob=1.0)
        result = transform(image=self.test_img, bboxes=self.test_bboxes)

        np.testing.assert_array_equal(result['bboxes'], self.test_bboxes)

    def test_normalize_preserves_bboxes(self):
        """Test NormalizeBboxAware preserves bbox coordinates."""
        transform = NormalizeBboxAware()
        result = transform(image=self.test_img, bboxes=self.test_bboxes)

        np.testing.assert_array_equal(result['bboxes'], self.test_bboxes)
        # Check normalization applied
        self.assertTrue(result['image'].dtype == np.float32)


class TestGeometricTransforms(unittest.TestCase):
    """Test geometric transforms (bbox-aware)."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        self.test_bbox = np.array([[100, 50, 80, 60]], dtype=np.float32)  # xywh

    def test_horizontal_flip_transforms_bbox(self):
        """Test RandomHorizontalFlipBboxAware correctly transforms bbox."""
        transform = RandomHorizontalFlipBboxAware(prob=1.0, bbox_format='xywh')
        result = transform(image=self.test_img, bboxes=self.test_bbox)

        # Verify bbox was transformed
        self.assertFalse(np.array_equal(result['bboxes'], self.test_bbox))

        # Verify math: x_flipped = W - x - w = 640 - 100 - 80 = 460
        expected_x = 640 - self.test_bbox[0, 0] - self.test_bbox[0, 2]
        np.testing.assert_almost_equal(result['bboxes'][0, 0], expected_x)

        # Verify y, w, h unchanged
        np.testing.assert_almost_equal(result['bboxes'][0, 1], self.test_bbox[0, 1])
        np.testing.assert_almost_equal(result['bboxes'][0, 2], self.test_bbox[0, 2])
        np.testing.assert_almost_equal(result['bboxes'][0, 3], self.test_bbox[0, 3])

    def test_vertical_flip_transforms_bbox(self):
        """Test RandomVerticalFlipBboxAware correctly transforms bbox."""
        transform = RandomVerticalFlipBboxAware(prob=1.0, bbox_format='xywh')
        result = transform(image=self.test_img, bboxes=self.test_bbox)

        # Verify math: y_flipped = H - y - h = 480 - 50 - 60 = 370
        expected_y = 480 - self.test_bbox[0, 1] - self.test_bbox[0, 3]
        np.testing.assert_almost_equal(result['bboxes'][0, 1], expected_y)

    def test_occlusion_filters_bboxes(self):
        """Test RandomOcclusionBboxAware can filter heavily occluded bboxes."""
        transform = RandomOcclusionBboxAware(
            min_size=0.8,  # Large occlusion
            max_size=0.9,
            prob=1.0,
            occlusion_threshold=0.3
        )
        result = transform(image=self.test_img, bboxes=self.test_bbox)

        # Result should have <= original bbox count
        self.assertLessEqual(len(result['bboxes']), len(self.test_bbox))


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility aliases."""

    def test_legacy_names_work(self):
        """Test that legacy names (ColorJitter, etc.) still work."""
        # These should not raise ImportError
        self.assertIsNotNone(ColorJitter)
        self.assertIsNotNone(RandomBlur)
        self.assertIsNotNone(Normalize)

    def test_aliases_point_to_bbox_aware(self):
        """Test that aliases point to BboxAware versions."""
        self.assertEqual(ColorJitter.__name__, 'ColorJitterBboxAware')
        self.assertEqual(RandomBlur.__name__, 'RandomBlurBboxAware')
        self.assertEqual(Normalize.__name__, 'NormalizeBboxAware')


class TestToTensor(unittest.TestCase):
    """Test ToTensor utility."""

    def test_to_tensor_hwc_to_chw(self):
        """Test ToTensor converts (H, W, C) to (C, H, W)."""
        img = np.random.rand(480, 640, 3).astype(np.float32)
        to_tensor = ToTensor()
        result = to_tensor(img)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 480, 640))

    def test_to_tensor_grayscale(self):
        """Test ToTensor handles grayscale images."""
        img = np.random.rand(480, 640).astype(np.float32)
        to_tensor = ToTensor()
        result = to_tensor(img)

        self.assertEqual(result.shape, (1, 480, 640))


class TestBboxFormatHandling(unittest.TestCase):
    """Test different bbox format handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    def test_xywh_format(self):
        """Test transforms work with xywh format."""
        bbox = np.array([[100, 50, 80, 60]], dtype=np.float32)
        transform = RandomHorizontalFlipBboxAware(prob=1.0, bbox_format='xywh')
        result = transform(image=self.test_img, bboxes=bbox)
        self.assertEqual(result['bboxes'].shape, (1, 4))

    def test_cxcywh_format(self):
        """Test transforms work with cxcywh format."""
        bbox = np.array([[140, 80, 80, 60]], dtype=np.float32)  # cx, cy, w, h
        transform = RandomHorizontalFlipBboxAware(prob=1.0, bbox_format='cxcywh')
        result = transform(image=self.test_img, bboxes=bbox)
        self.assertEqual(result['bboxes'].shape, (1, 4))

    def test_xyxy_format(self):
        """Test transforms work with xyxy format."""
        bbox = np.array([[100, 50, 180, 110]], dtype=np.float32)  # x1, y1, x2, y2
        transform = RandomHorizontalFlipBboxAware(prob=1.0, bbox_format='xyxy')
        result = transform(image=self.test_img, bboxes=bbox)
        self.assertEqual(result['bboxes'].shape, (1, 4))


if __name__ == '__main__':
    unittest.main()
