"""
Test Augmentation None Bbox Handling (Audit 2025112003 - Bug Fix)

Regression tests to ensure augmentation correctly handles None bboxes
in single-modality training mode.

Critical bugs fixed:
- augmentation.py Line 228-229: AttributeError on None.copy()
- augmentation.py Line 235, 265: AttributeError on None.reshape()

Author: AquaSOT-3D Team
Date: 2025-11-20
"""

import pytest
import torch
import numpy as np

from aquasot3d.dataloader.datasets.augmentation import (
    TrackingAugmentation,
    NoAugmentation
)


class TestTrackingAugmentationNoneBbox:
    """Test TrackingAugmentation handles None bbox correctly"""

    def test_rgb_only_mode_none_sonar_bbox(self):
        """Test RGB-only mode with None sonar_bbox (Sonar disabled)"""
        aug = TrackingAugmentation(is_train=False)

        data = {
            'rgb_img': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'sonar_img': None,  # Sonar disabled
            'rgb_bbox': np.array([100, 100, 50, 50], dtype=np.float32),
            'sonar_bbox': None,  # ❌ This used to crash with AttributeError
            'is_valid_rgb': True,
            'is_valid_sonar': None
        }

        # Should not crash
        result = aug(data)

        # Verify RGB processed correctly
        assert result['rgb_img'] is not None
        assert isinstance(result['rgb_img'], torch.Tensor)
        assert result['rgb_bbox'] is not None

        # Verify Sonar remains None
        assert result['sonar_img'] is None
        assert result['sonar_bbox'] is None
        assert result['is_valid_sonar'] is None

    def test_sonar_only_mode_none_rgb_bbox(self):
        """Test Sonar-only mode with None rgb_bbox (RGB disabled)"""
        aug = TrackingAugmentation(is_train=False)

        data = {
            'rgb_img': None,  # RGB disabled
            'sonar_img': np.random.randint(0, 255, (376, 682), dtype=np.uint8),
            'rgb_bbox': None,  # ❌ This used to crash with AttributeError
            'sonar_bbox': np.array([200, 150, 80, 60], dtype=np.float32),
            'is_valid_rgb': None,
            'is_valid_sonar': True
        }

        # Should not crash
        result = aug(data)

        # Verify Sonar processed correctly
        assert result['sonar_img'] is not None
        assert isinstance(result['sonar_img'], torch.Tensor)
        assert result['sonar_bbox'] is not None

        # Verify RGB remains None
        assert result['rgb_img'] is None
        assert result['rgb_bbox'] is None
        assert result['is_valid_rgb'] is None

    def test_rgb_only_mode_with_augmentation(self):
        """Test RGB-only mode with augmentation enabled"""
        aug = TrackingAugmentation(
            is_train=True,
            rgb_config={
                'enable_color_jitter': True,
                'enable_blur': True,
                'enable_occlusion': False  # Disable to avoid bbox filtering
            }
        )

        data = {
            'rgb_img': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'sonar_img': None,
            'rgb_bbox': np.array([100, 100, 50, 50], dtype=np.float32),
            'sonar_bbox': None,
            'is_valid_rgb': True,
            'is_valid_sonar': None
        }

        result = aug(data)

        # Verify augmentation applied to RGB
        assert result['rgb_img'] is not None
        assert result['rgb_bbox'] is not None
        assert result['is_valid_rgb'] == True

        # Verify Sonar not processed
        assert result['sonar_img'] is None
        assert result['sonar_bbox'] is None

    def test_sonar_only_mode_with_augmentation(self):
        """Test Sonar-only mode with augmentation enabled"""
        aug = TrackingAugmentation(
            is_train=True,
            sonar_config={
                'enable_speckle': True,
                'enable_occlusion': False  # Disable to avoid bbox filtering
            }
        )

        data = {
            'rgb_img': None,
            'sonar_img': np.random.randint(0, 255, (376, 682), dtype=np.uint8),
            'rgb_bbox': None,
            'sonar_bbox': np.array([200, 150, 80, 60], dtype=np.float32),
            'is_valid_rgb': None,
            'is_valid_sonar': True
        }

        result = aug(data)

        # Verify augmentation applied to Sonar
        assert result['sonar_img'] is not None
        assert result['sonar_bbox'] is not None
        assert result['is_valid_sonar'] == True

        # Verify RGB not processed
        assert result['rgb_img'] is None
        assert result['rgb_bbox'] is None

    def test_dual_modality_unaffected(self):
        """Test dual-modality mode still works correctly (no regression)"""
        aug = TrackingAugmentation(is_train=False)

        data = {
            'rgb_img': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'sonar_img': np.random.randint(0, 255, (376, 682), dtype=np.uint8),
            'rgb_bbox': np.array([100, 100, 50, 50], dtype=np.float32),
            'sonar_bbox': np.array([200, 150, 80, 60], dtype=np.float32),
            'is_valid_rgb': True,
            'is_valid_sonar': True
        }

        result = aug(data)

        # Verify both modalities processed
        assert result['rgb_img'] is not None
        assert result['sonar_img'] is not None
        assert result['rgb_bbox'] is not None
        assert result['sonar_bbox'] is not None
        assert result['is_valid_rgb'] == True
        assert result['is_valid_sonar'] == True


class TestNoAugmentationNoneBbox:
    """Test NoAugmentation handles None bbox correctly"""

    def test_rgb_only_mode_none_sonar_bbox(self):
        """Test RGB-only mode with None sonar_bbox in evaluation"""
        aug = NoAugmentation()

        data = {
            'rgb_img': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'sonar_img': None,
            'rgb_bbox': np.array([100, 100, 50, 50], dtype=np.float32),
            'sonar_bbox': None  # Should not crash
        }

        result = aug(data)

        # Verify RGB processed
        assert result['rgb_img'] is not None
        assert isinstance(result['rgb_img'], torch.Tensor)
        assert result['rgb_bbox'] is not None

        # Verify Sonar remains None
        assert result['sonar_img'] is None
        assert result['sonar_bbox'] is None

    def test_sonar_only_mode_none_rgb_bbox(self):
        """Test Sonar-only mode with None rgb_bbox in evaluation"""
        aug = NoAugmentation()

        data = {
            'rgb_img': None,
            'sonar_img': np.random.randint(0, 255, (376, 682), dtype=np.uint8),
            'rgb_bbox': None,  # Should not crash
            'sonar_bbox': np.array([200, 150, 80, 60], dtype=np.float32)
        }

        result = aug(data)

        # Verify Sonar processed
        assert result['sonar_img'] is not None
        assert isinstance(result['sonar_img'], torch.Tensor)
        assert result['sonar_bbox'] is not None

        # Verify RGB remains None
        assert result['rgb_img'] is None
        assert result['rgb_bbox'] is None

    def test_dual_modality_unaffected(self):
        """Test dual-modality mode still works correctly (no regression)"""
        aug = NoAugmentation()

        data = {
            'rgb_img': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'sonar_img': np.random.randint(0, 255, (376, 682), dtype=np.uint8),
            'rgb_bbox': np.array([100, 100, 50, 50], dtype=np.float32),
            'sonar_bbox': np.array([200, 150, 80, 60], dtype=np.float32)
        }

        result = aug(data)

        # Verify both modalities processed
        assert result['rgb_img'] is not None
        assert result['sonar_img'] is not None
        assert result['rgb_bbox'] is not None
        assert result['sonar_bbox'] is not None


class TestBboxCopyAndReshapeSafety:
    """Test specific operations that caused crashes"""

    def test_none_bbox_copy_safe(self):
        """Test that None bbox .copy() is handled safely"""
        aug = TrackingAugmentation(is_train=False)

        data = {
            'rgb_img': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'sonar_img': None,
            'rgb_bbox': np.array([100, 100, 50, 50], dtype=np.float32),
            'sonar_bbox': None,  # Line 229 used to call None.copy()
            'is_valid_rgb': True,
            'is_valid_sonar': None
        }

        # This used to raise: AttributeError: 'NoneType' object has no attribute 'copy'
        result = aug(data)
        assert result['sonar_bbox'] is None

    def test_none_bbox_reshape_safe(self):
        """Test that None bbox .reshape() is handled safely"""
        aug = TrackingAugmentation(is_train=False)

        data = {
            'rgb_img': None,
            'sonar_img': np.random.randint(0, 255, (376, 682), dtype=np.uint8),
            'rgb_bbox': None,  # Line 235 used to call None.reshape()
            'sonar_bbox': np.array([200, 150, 80, 60], dtype=np.float32),
            'is_valid_rgb': None,
            'is_valid_sonar': True
        }

        # This used to raise: AttributeError: 'NoneType' object has no attribute 'reshape'
        result = aug(data)
        assert result['rgb_bbox'] is None

    def test_none_bbox_tensor_conversion_safe(self):
        """Test that None bbox tensor conversion is handled safely"""
        aug = TrackingAugmentation(is_train=False)

        data = {
            'rgb_img': np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
            'sonar_img': None,
            'rgb_bbox': np.array([100, 100, 50, 50], dtype=np.float32),
            'sonar_bbox': None,
            'is_valid_rgb': True,
            'is_valid_sonar': None
        }

        result = aug(data)

        # Verify RGB bbox converted to tensor
        assert isinstance(result['rgb_bbox'], (torch.Tensor, np.ndarray))

        # Verify Sonar bbox remains None (not converted)
        assert result['sonar_bbox'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
