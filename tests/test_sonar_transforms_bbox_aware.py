"""
Unit Tests for Bbox-Aware Sonar Transforms

Tests Phase 2 and Phase 3 transforms from Audit 2025111801 implementation.

Author: AquaSOT-3D Team
Date: 2025-11-18
"""

import pytest
import numpy as np
import torch

from aquasot3d.dataloader.transforms.sonar_transforms_bbox_aware import (
    SpeckleNoiseBboxAware,
    IntensityNormalizationBboxAware,
    FanMaskApplicationBboxAware,
    RadialScalingJitterBboxAware,
    SectorOcclusionBboxAware,
)


# ==============================================================================
# Phase 2: Photometric Transform Tests
# ==============================================================================

class TestSpeckleNoiseBboxAware:
    """Test speckle noise photometric transform."""

    def test_preserves_bboxes(self):
        """Test that speckle noise does not modify bboxes."""
        transform = SpeckleNoiseBboxAware(noise_level=0.1, prob=1.0)

        image = np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        # Bboxes should be unchanged
        np.testing.assert_array_equal(result['bboxes'], bboxes)

    def test_adds_noise_to_image(self):
        """Test that speckle noise modifies image."""
        transform = SpeckleNoiseBboxAware(noise_level=0.1, prob=1.0)

        image = np.ones((376, 682, 1), dtype=np.uint8) * 128
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        # Image should be different (noise added)
        assert not np.array_equal(result['image'], image)

    def test_probabilistic_execution(self):
        """Test that probability parameter works."""
        transform = SpeckleNoiseBboxAware(noise_level=0.1, prob=0.0)

        image = np.ones((376, 682, 1), dtype=np.uint8) * 128
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        # With p=0, image should be unchanged
        np.testing.assert_array_equal(result['image'], image)

    def test_output_range_valid(self):
        """Test that output remains in valid range [0, 255]."""
        transform = SpeckleNoiseBboxAware(noise_level=0.5, prob=1.0)

        image = np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        assert np.min(result['image']) >= 0
        assert np.max(result['image']) <= 255


class TestIntensityNormalizationBboxAware:
    """Test intensity normalization photometric transform."""

    def test_preserves_bboxes(self):
        """Test that normalization does not modify bboxes."""
        transform = IntensityNormalizationBboxAware(mode='adaptive')

        image = np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        np.testing.assert_array_equal(result['bboxes'], bboxes)

    def test_adaptive_normalization(self):
        """Test adaptive normalization adjusts mean and std."""
        transform = IntensityNormalizationBboxAware(
            mode='adaptive', target_mean=128.0, target_std=50.0
        )

        image = np.random.randint(50, 200, (376, 682, 1), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        # Check normalized statistics are close to target
        result_mean = np.mean(result['image'])
        result_std = np.std(result['image'])

        np.testing.assert_allclose(result_mean, 128.0, rtol=0.1)
        np.testing.assert_allclose(result_std, 50.0, rtol=0.3)

    def test_none_mode_passthrough(self):
        """Test that mode='none' returns unchanged image."""
        transform = IntensityNormalizationBboxAware(mode='none')

        image = np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        np.testing.assert_array_equal(result['image'], image)


class TestFanMaskApplicationBboxAware:
    """Test fan mask photometric transform."""

    def test_preserves_bboxes(self):
        """Test that fan mask does not modify bboxes."""
        transform = FanMaskApplicationBboxAware(fov_angle=130)

        image = np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        np.testing.assert_array_equal(result['bboxes'], bboxes)

    def test_zeros_outside_fov(self):
        """Test that regions outside FOV are zeroed."""
        transform = FanMaskApplicationBboxAware(fov_angle=130)

        image = np.ones((376, 682, 1), dtype=np.uint8) * 255
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        # Result should have zeros (outside FOV)
        assert np.min(result['image']) == 0

        # Center region (inside FOV) should have non-zero values
        cx, cy = 682 // 2, 376 - 1
        center_region = result['image'][cy - 50:cy, cx - 50:cx + 50]
        assert np.mean(center_region) > 0

    def test_mask_caching(self):
        """Test that mask is cached for repeated use."""
        transform = FanMaskApplicationBboxAware(fov_angle=130)

        image1 = np.ones((376, 682, 1), dtype=np.uint8) * 255
        image2 = np.ones((376, 682, 1), dtype=np.uint8) * 200
        bboxes = np.array([[100, 200, 50, 80]])

        result1 = transform(image=image1, bboxes=bboxes)
        result2 = transform(image=image2, bboxes=bboxes)

        # Cache should have entry for (376, 682)
        assert (376, 682) in transform._mask_cache


# ==============================================================================
# Phase 3: Geometric Transform Tests
# ==============================================================================

class TestRadialScalingJitterBboxAware:
    """Test radial scaling jitter geometric transform (Audit Finding 2)."""

    def test_transforms_both_image_and_bboxes(self):
        """Test that both image and bboxes are transformed consistently."""
        transform = RadialScalingJitterBboxAware(
            scale_range=(1.1, 1.1), prob=1.0  # Fixed scale for deterministic test
        )

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        # Bboxes should be modified (scaled)
        assert not np.array_equal(result['bboxes'], bboxes)

        # Bbox size should increase by ~10% (scale 1.1)
        np.testing.assert_allclose(
            result['bboxes'][0, 2:], bboxes[0, 2:] * 1.1, rtol=0.15
        )

    def test_bbox_image_synchronization(self):
        """
        Test that bbox-image synchronization is maintained (Audit Finding 2).

        This is the critical test for the ~19% IoU loss bug fix.
        """
        transform = RadialScalingJitterBboxAware(
            scale_range=(1.05, 1.05), prob=1.0, sensor_pos=(341, 375)
        )

        # Create synthetic image with target at bbox location
        image = np.zeros((376, 682, 3), dtype=np.uint8)
        bbox = np.array([[250, 120, 80, 60]])

        # Draw target in bbox
        x, y, w, h = bbox[0].astype(int)
        image[y:y+h, x:x+w, :] = 255

        result = transform(image=image, bboxes=bbox)

        # Extract transformed bbox region
        x_new, y_new, w_new, h_new = result['bboxes'][0].astype(int)
        bbox_region = result['image'][y_new:y_new+h_new, x_new:x_new+w_new, :]

        # Bbox region should contain the transformed target (high intensity)
        mean_intensity = np.mean(bbox_region)
        assert mean_intensity > 100, f"Bbox desynchronized: mean intensity {mean_intensity} < 100"

    def test_batch_transform(self):
        """Test transforming batch of bboxes."""
        transform = RadialScalingJitterBboxAware(
            scale_range=(1.1, 1.1), prob=1.0
        )

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([
            [100, 200, 50, 80],
            [300, 150, 60, 90],
        ])

        result = transform(image=image, bboxes=bboxes)

        # Both bboxes should be transformed
        assert len(result['bboxes']) == 2
        assert not np.array_equal(result['bboxes'], bboxes)

    def test_probabilistic_execution(self):
        """Test that probability parameter works."""
        transform = RadialScalingJitterBboxAware(
            scale_range=(1.1, 1.1), prob=0.0
        )

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        # With p=0, both should be unchanged
        np.testing.assert_array_equal(result['image'], image)
        np.testing.assert_array_equal(result['bboxes'], bboxes)

    def test_torch_tensor_bboxes(self):
        """Test that torch tensor bboxes are handled correctly."""
        transform = RadialScalingJitterBboxAware(
            scale_range=(1.1, 1.1), prob=1.0
        )

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = torch.tensor([[100.0, 200.0, 50.0, 80.0]])

        result = transform(image=image, bboxes=bboxes)

        # Result should be torch tensor
        assert isinstance(result['bboxes'], torch.Tensor)


class TestSectorOcclusionBboxAware:
    """Test sector occlusion geometric transform (Audit Finding 3)."""

    def test_filters_occluded_bboxes(self):
        """
        Test that heavily occluded bboxes are filtered out (Audit Finding 3).
        """
        transform = SectorOcclusionBboxAware(
            min_angle=30, max_angle=30, prob=1.0, occlusion_threshold=0.5
        )

        image = np.ones((376, 682, 3), dtype=np.uint8) * 255
        # Place bbox in center (likely to be occluded by random sector)
        bboxes = np.array([[320, 150, 80, 100]])

        # Run multiple times to hit occluded case
        occluded_count = 0
        for _ in range(20):
            result = transform(image=image, bboxes=bboxes)
            if len(result['bboxes']) < len(bboxes):
                occluded_count += 1

        # At least some runs should filter the bbox
        assert occluded_count > 0, "Bbox should be occluded in some runs"

    def test_preserves_non_occluded_bboxes(self):
        """Test that non-occluded bboxes are preserved."""
        # Place bbox at extreme left (unlikely to be occluded)
        transform = SectorOcclusionBboxAware(
            min_angle=10, max_angle=20, prob=1.0, occlusion_threshold=0.8
        )

        image = np.ones((376, 682, 3), dtype=np.uint8) * 255
        bboxes = np.array([
            [50, 300, 40, 50],    # Left edge
            [600, 300, 40, 50],   # Right edge
        ])

        # Run multiple times
        for _ in range(10):
            result = transform(image=image, bboxes=bboxes)
            # At least one bbox should remain
            assert len(result['bboxes']) >= 1

    def test_occlusion_applied_to_image(self):
        """Test that sector occlusion zeros out image region."""
        transform = SectorOcclusionBboxAware(
            min_angle=30, max_angle=30, prob=1.0
        )

        image = np.ones((376, 682, 3), dtype=np.uint8) * 255
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        # Result should have zeros (occluded sector)
        assert np.min(result['image']) == 0

    def test_batch_filtering(self):
        """Test filtering batch of bboxes based on occlusion."""
        transform = SectorOcclusionBboxAware(
            min_angle=40, max_angle=40, prob=1.0, occlusion_threshold=0.5
        )

        image = np.ones((376, 682, 3), dtype=np.uint8) * 255
        bboxes = np.array([
            [100, 150, 50, 80],
            [300, 200, 60, 90],
            [500, 250, 40, 70],
        ])

        # Run multiple times
        filtered_counts = []
        for _ in range(10):
            result = transform(image=image, bboxes=bboxes)
            filtered_counts.append(len(result['bboxes']))

        # Some runs should filter bboxes
        assert min(filtered_counts) < len(bboxes)

    def test_probabilistic_execution(self):
        """Test that probability parameter works."""
        transform = SectorOcclusionBboxAware(
            min_angle=30, max_angle=30, prob=0.0
        )

        image = np.ones((376, 682, 3), dtype=np.uint8) * 255
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        # With p=0, both should be unchanged
        np.testing.assert_array_equal(result['image'], image)
        np.testing.assert_array_equal(result['bboxes'], bboxes)


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegrationScenarios:
    """Integration tests for real-world augmentation scenarios."""

    def test_photometric_pipeline(self):
        """Test composing multiple photometric transforms."""
        from aquasot3d.dataloader.transforms.base import ComposeTransforms

        pipeline = ComposeTransforms([
            SpeckleNoiseBboxAware(noise_level=0.1, prob=1.0),
            IntensityNormalizationBboxAware(mode='adaptive'),
            FanMaskApplicationBboxAware(fov_angle=130),
        ])

        image = np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = pipeline(image=image, bboxes=bboxes)

        # Bboxes should be unchanged (all photometric)
        np.testing.assert_array_equal(result['bboxes'], bboxes)

        # Image should be modified
        assert not np.array_equal(result['image'], image)

    def test_geometric_pipeline(self):
        """Test composing photometric and geometric transforms."""
        from aquasot3d.dataloader.transforms.base import ComposeTransforms

        pipeline = ComposeTransforms([
            SpeckleNoiseBboxAware(noise_level=0.1, prob=1.0),
            RadialScalingJitterBboxAware(scale_range=(1.05, 1.05), prob=1.0),
            FanMaskApplicationBboxAware(fov_angle=130),
        ])

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = pipeline(image=image, bboxes=bboxes)

        # Bboxes should be modified (geometric transform)
        assert not np.array_equal(result['bboxes'], bboxes)

    def test_metadata_passthrough(self):
        """Test that metadata is preserved through pipeline."""
        transform = SpeckleNoiseBboxAware(noise_level=0.1, prob=1.0)

        image = np.random.randint(0, 255, (376, 682, 1), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])
        metadata = {
            'frame_id': 42,
            'sequence_name': 'uuv1',
            'pad_x': 0,
            'pad_y': 10,
        }

        result = transform(image=image, bboxes=bboxes, **metadata)

        # Metadata should be preserved
        assert result['frame_id'] == 42
        assert result['sequence_name'] == 'uuv1'
        assert result['pad_x'] == 0
        assert result['pad_y'] == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
