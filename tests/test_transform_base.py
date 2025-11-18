"""
Unit Tests for Bbox-Aware Transform Base Classes

Tests the base transform architecture introduced in Audit 2025111801 Phase 1
to solve bbox-image desynchronization bugs.

Author: AquaSOT-3D Team
Date: 2025-11-18
"""

import pytest
import numpy as np
import torch
import cv2

from aquasot3d.dataloader.transforms.base import (
    BboxFormat,
    validate_bboxes,
    transform_bboxes_affine,
    BboxAwareTransform,
    PhotometricTransform,
    LegacyTransformWrapper,
    ComposeTransforms,
    compose_transforms,
)


class TestBboxFormat:
    """Test BboxFormat dataclass and standard format definitions."""

    def test_xywh_format(self):
        """Test XYWH format specification."""
        fmt = BboxFormat.XYWH
        assert fmt.name == 'xywh'
        assert '[x, y, w, h]' in fmt.description
        assert 'Top-left' in fmt.description

    def test_cxcywh_format(self):
        """Test CXCYWH format specification."""
        fmt = BboxFormat.CXCYWH
        assert fmt.name == 'cxcywh'
        assert '[cx, cy, w, h]' in fmt.description
        assert 'Center' in fmt.description

    def test_xyxy_format(self):
        """Test XYXY format specification."""
        fmt = BboxFormat.XYXY
        assert fmt.name == 'xyxy'
        assert '[x1, y1, x2, y2]' in fmt.description
        assert 'corners' in fmt.description.lower()

    def test_custom_format(self):
        """Test creating custom bbox format."""
        custom = BboxFormat('custom', 'Custom format description')
        assert custom.name == 'custom'
        assert custom.description == 'Custom format description'


class TestValidateBboxes:
    """Test bbox validation and boundary clipping."""

    def test_validate_xywh_within_bounds(self):
        """Test validating bboxes already within image bounds."""
        bboxes = np.array([[100, 200, 50, 80]])
        image_shape = (376, 682)

        result = validate_bboxes(bboxes, image_shape, bbox_format='xywh')

        np.testing.assert_array_equal(result, bboxes)

    def test_clip_out_of_bounds_xywh(self):
        """Test clipping bboxes that extend beyond image boundaries."""
        bboxes = np.array([[650, 300, 100, 100]])  # Extends beyond width=682
        image_shape = (376, 682)

        result = validate_bboxes(bboxes, image_shape, bbox_format='xywh', clip=True)

        # Should be clipped to fit within [0, 682) x [0, 376)
        assert result[0, 0] >= 0
        assert result[0, 1] >= 0
        assert result[0, 0] + result[0, 2] <= 682
        assert result[0, 1] + result[0, 3] <= 376

    def test_clip_negative_coordinates(self):
        """Test clipping bboxes with negative coordinates."""
        bboxes = np.array([[-10, -20, 50, 80]])
        image_shape = (376, 682)

        result = validate_bboxes(bboxes, image_shape, bbox_format='xywh', clip=True)

        # Should be clipped to start at (0, 0)
        assert result[0, 0] >= 0
        assert result[0, 1] >= 0

    def test_validate_cxcywh_format(self):
        """Test validating center-format bboxes."""
        bboxes = np.array([[341, 188, 100, 80]])  # Center format
        image_shape = (376, 682)

        result = validate_bboxes(bboxes, image_shape, bbox_format='cxcywh', clip=True)

        # Check center remains valid
        assert 0 <= result[0, 0] <= 682  # cx
        assert 0 <= result[0, 1] <= 376  # cy

    def test_validate_xyxy_format(self):
        """Test validating corner-format bboxes."""
        bboxes = np.array([[100, 200, 150, 280]])  # x1, y1, x2, y2
        image_shape = (376, 682)

        result = validate_bboxes(bboxes, image_shape, bbox_format='xyxy', clip=True)

        # Verify corners are valid
        assert result[0, 0] < result[0, 2]  # x1 < x2
        assert result[0, 1] < result[0, 3]  # y1 < y2

    def test_filter_by_minimum_area(self):
        """Test filtering bboxes below minimum area threshold."""
        bboxes = np.array([
            [100, 200, 50, 80],    # Area = 4000 (valid)
            [300, 150, 2, 2],      # Area = 4 (invalid)
            [500, 300, 30, 30],    # Area = 900 (valid)
        ])
        image_shape = (376, 682)

        result = validate_bboxes(
            bboxes, image_shape, bbox_format='xywh', min_area=100.0
        )

        # Only first and third bbox should remain
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], bboxes[0])
        np.testing.assert_array_equal(result[1], bboxes[2])

    def test_validate_torch_tensor(self):
        """Test validating torch tensor bboxes."""
        bboxes = torch.tensor([[100.0, 200.0, 50.0, 80.0]])
        image_shape = (376, 682)

        result = validate_bboxes(bboxes, image_shape, bbox_format='xywh')

        assert isinstance(result, torch.Tensor)
        assert result.shape == bboxes.shape

    def test_validate_empty_bboxes(self):
        """Test handling empty bbox arrays."""
        bboxes = np.array([]).reshape(0, 4)
        image_shape = (376, 682)

        result = validate_bboxes(bboxes, image_shape, bbox_format='xywh')

        assert len(result) == 0

    def test_validate_none_bboxes(self):
        """Test handling None bboxes."""
        result = validate_bboxes(None, (376, 682), bbox_format='xywh')
        assert result is None

    def test_invalid_shape_raises_error(self):
        """Test that invalid bbox shape raises ValueError."""
        bboxes = np.array([[100, 200]])  # Only 2 values, not 4
        image_shape = (376, 682)

        with pytest.raises(ValueError, match="shape"):
            validate_bboxes(bboxes, image_shape)

    def test_invalid_format_raises_error(self):
        """Test that invalid bbox format raises ValueError."""
        bboxes = np.array([[100, 200, 50, 80]])
        image_shape = (376, 682)

        with pytest.raises(ValueError, match="Unknown bbox format"):
            validate_bboxes(bboxes, image_shape, bbox_format='invalid')


class TestTransformBboxesAffine:
    """Test affine transformation of bounding boxes."""

    def test_identity_transform(self):
        """Test that identity transform preserves bboxes."""
        bboxes = np.array([[100, 200, 50, 80]])
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]], dtype=np.float32)
        image_shape = (376, 682)

        result = transform_bboxes_affine(bboxes, M, image_shape, bbox_format='xywh')

        np.testing.assert_allclose(result, bboxes, rtol=1e-5)

    def test_translation_transform(self):
        """Test bbox translation."""
        bboxes = np.array([[100, 200, 50, 80]])
        M = np.array([[1.0, 0.0, 10.0],   # Translate +10 in x
                      [0.0, 1.0, -20.0]], dtype=np.float32)  # Translate -20 in y
        image_shape = (376, 682)

        result = transform_bboxes_affine(bboxes, M, image_shape, bbox_format='xywh')

        expected = np.array([[110, 180, 50, 80]])  # Same size, translated
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_uniform_scaling_transform(self):
        """Test bbox scaling (radial jitter scenario)."""
        bboxes = np.array([[100, 200, 50, 80]])
        # Scale 1.1x around center (341, 188)
        cx, cy = 341, 188
        scale = 1.1
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        image_shape = (376, 682)

        result = transform_bboxes_affine(bboxes, M, image_shape, bbox_format='xywh')

        # Bbox should be scaled proportionally
        # Check width and height increased by ~10%
        assert result[0, 2] > bboxes[0, 2]  # Width increased
        assert result[0, 3] > bboxes[0, 3]  # Height increased
        np.testing.assert_allclose(result[0, 2], bboxes[0, 2] * 1.1, rtol=0.05)

    def test_rotation_transform(self):
        """Test bbox rotation updates."""
        bboxes = np.array([[300, 150, 40, 60]])
        # Rotate 15 degrees around center
        cx, cy = 341, 188
        M = cv2.getRotationMatrix2D((cx, cy), 15, 1.0)
        image_shape = (376, 682)

        result = transform_bboxes_affine(bboxes, M, image_shape, bbox_format='xywh')

        # After rotation, axis-aligned bbox should be larger
        # (encompasses rotated rectangle)
        assert result[0, 2] >= bboxes[0, 2]  # Width >= original
        assert result[0, 3] >= bboxes[0, 3]  # Height >= original

    def test_cxcywh_format_transform(self):
        """Test affine transform with center-format bboxes."""
        bboxes = np.array([[341, 188, 50, 80]])  # Center format
        M = np.array([[1.0, 0.0, 20.0],
                      [0.0, 1.0, -10.0]], dtype=np.float32)
        image_shape = (376, 682)

        result = transform_bboxes_affine(bboxes, M, image_shape, bbox_format='cxcywh')

        # Center should be translated
        expected_cx = 341 + 20
        expected_cy = 188 - 10
        np.testing.assert_allclose(result[0, 0], expected_cx, rtol=1e-3)
        np.testing.assert_allclose(result[0, 1], expected_cy, rtol=1e-3)

    def test_xyxy_format_transform(self):
        """Test affine transform with corner-format bboxes."""
        bboxes = np.array([[100, 200, 150, 280]])  # x1, y1, x2, y2
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]], dtype=np.float32)
        image_shape = (376, 682)

        result = transform_bboxes_affine(bboxes, M, image_shape, bbox_format='xyxy')

        np.testing.assert_allclose(result, bboxes, rtol=1e-5)

    def test_batch_transform(self):
        """Test transforming batch of bboxes."""
        bboxes = np.array([
            [100, 200, 50, 80],
            [300, 150, 60, 90],
            [500, 250, 40, 70]
        ])
        M = np.array([[1.0, 0.0, 10.0],
                      [0.0, 1.0, -5.0]], dtype=np.float32)
        image_shape = (376, 682)

        result = transform_bboxes_affine(bboxes, M, image_shape, bbox_format='xywh')

        # All bboxes should be translated
        assert len(result) == 3
        for i in range(3):
            np.testing.assert_allclose(result[i, 0], bboxes[i, 0] + 10, rtol=1e-5)

    def test_torch_tensor_transform(self):
        """Test transforming torch tensor bboxes."""
        bboxes = torch.tensor([[100.0, 200.0, 50.0, 80.0]])
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]], dtype=np.float32)
        image_shape = (376, 682)

        result = transform_bboxes_affine(bboxes, M, image_shape, bbox_format='xywh')

        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, bboxes)

    def test_transform_with_clipping(self):
        """Test that transformed bboxes are clipped to image boundaries."""
        bboxes = np.array([[600, 300, 100, 100]])
        # Translate further out of bounds
        M = np.array([[1.0, 0.0, 50.0],
                      [0.0, 1.0, 50.0]], dtype=np.float32)
        image_shape = (376, 682)

        result = transform_bboxes_affine(bboxes, M, image_shape, bbox_format='xywh')

        # Result should be clipped to image boundaries
        assert result[0, 0] + result[0, 2] <= 682
        assert result[0, 1] + result[0, 3] <= 376

    def test_empty_bboxes(self):
        """Test handling empty bbox arrays."""
        bboxes = np.array([]).reshape(0, 4)
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]], dtype=np.float32)
        image_shape = (376, 682)

        result = transform_bboxes_affine(bboxes, M, image_shape, bbox_format='xywh')

        assert len(result) == 0

    def test_none_bboxes(self):
        """Test handling None bboxes."""
        M = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]], dtype=np.float32)
        image_shape = (376, 682)

        result = transform_bboxes_affine(None, M, image_shape, bbox_format='xywh')

        assert result is None


class DummyGeometricTransform(BboxAwareTransform):
    """Dummy geometric transform for testing."""

    def apply_to_image(self, image):
        # Simple translation
        M = np.array([[1.0, 0.0, 10.0],
                      [0.0, 1.0, -5.0]], dtype=np.float32)
        H, W = image.shape[:2]
        return cv2.warpAffine(image, M, (W, H))

    def apply_to_bboxes(self, bboxes, image_shape):
        M = np.array([[1.0, 0.0, 10.0],
                      [0.0, 1.0, -5.0]], dtype=np.float32)
        return transform_bboxes_affine(bboxes, M, image_shape, self.bbox_format)


class DummyPhotometricTransform(PhotometricTransform):
    """Dummy photometric transform for testing."""

    def apply_to_image(self, image):
        return (image * 1.2).clip(0, 255).astype(image.dtype)


class TestBboxAwareTransform:
    """Test BboxAwareTransform base class."""

    def test_geometric_transform_synchronization(self):
        """Test that geometric transform updates both image and bboxes."""
        transform = DummyGeometricTransform(bbox_format='xywh', p=1.0)

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        # Bboxes should be transformed (translated +10 in x, -5 in y)
        expected_bbox = np.array([[110, 195, 50, 80]])
        np.testing.assert_allclose(result['bboxes'], expected_bbox, rtol=1e-3)

    def test_photometric_transform_preserves_bboxes(self):
        """Test that photometric transform preserves bboxes."""
        transform = DummyPhotometricTransform(bbox_format='xywh', p=1.0)

        image = np.random.randint(0, 200, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        # Bboxes should be unchanged
        np.testing.assert_array_equal(result['bboxes'], bboxes)

    def test_probabilistic_execution(self):
        """Test that transforms respect probability parameter."""
        transform = DummyGeometricTransform(bbox_format='xywh', p=0.0)

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = transform(image=image, bboxes=bboxes)

        # With p=0, transform should not be applied
        np.testing.assert_array_equal(result['image'], image)
        np.testing.assert_array_equal(result['bboxes'], bboxes)

    def test_metadata_passthrough(self):
        """Test that kwargs metadata is passed through."""
        transform = DummyPhotometricTransform(bbox_format='xywh', p=1.0)

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])
        metadata = {'frame_id': 42, 'sequence_name': 'uuv1'}

        result = transform(image=image, bboxes=bboxes, **metadata)

        # Metadata should be preserved
        assert result['frame_id'] == 42
        assert result['sequence_name'] == 'uuv1'

    def test_none_bboxes_handling(self):
        """Test handling None bboxes (e.g., for evaluation)."""
        transform = DummyGeometricTransform(bbox_format='xywh', p=1.0)

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)

        result = transform(image=image, bboxes=None)

        assert result['bboxes'] is None

    def test_invalid_bbox_format_raises_error(self):
        """Test that invalid bbox format raises ValueError."""
        with pytest.raises(ValueError, match="bbox_format"):
            DummyGeometricTransform(bbox_format='invalid_format')


class DummyLegacyTransform:
    """Dummy legacy transform with old interface."""

    def __call__(self, image):
        return (image * 0.8).clip(0, 255).astype(image.dtype)


class TestLegacyTransformWrapper:
    """Test LegacyTransformWrapper for backward compatibility."""

    def test_wrap_photometric_transform(self):
        """Test wrapping legacy photometric transform."""
        legacy = DummyLegacyTransform()
        wrapped = LegacyTransformWrapper(legacy, is_photometric=True, bbox_format='xywh')

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = wrapped(image=image, bboxes=bboxes)

        # Image should be transformed
        assert not np.array_equal(result['image'], image)
        # Bboxes should be unchanged
        np.testing.assert_array_equal(result['bboxes'], bboxes)

    def test_wrap_geometric_transform_warns(self):
        """Test that wrapping geometric transform issues warning."""
        legacy = DummyLegacyTransform()

        with pytest.warns(UserWarning, match="geometric transform"):
            LegacyTransformWrapper(legacy, is_photometric=False, bbox_format='xywh')

    def test_legacy_wrapper_in_pipeline(self):
        """Test legacy wrapper works in new pipeline."""
        legacy = DummyLegacyTransform()
        wrapped = LegacyTransformWrapper(legacy, is_photometric=True, bbox_format='xywh')

        # Compose with new bbox-aware transform
        pipeline = ComposeTransforms([
            wrapped,
            DummyPhotometricTransform(bbox_format='xywh')
        ])

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = pipeline(image=image, bboxes=bboxes)

        # Should work without errors
        assert 'image' in result
        assert 'bboxes' in result


class TestComposeTransforms:
    """Test transform composition."""

    def test_compose_multiple_transforms(self):
        """Test composing multiple transforms."""
        transforms = [
            DummyPhotometricTransform(bbox_format='xywh', p=1.0),
            DummyGeometricTransform(bbox_format='xywh', p=1.0),
            DummyPhotometricTransform(bbox_format='xywh', p=1.0),
        ]
        pipeline = ComposeTransforms(transforms)

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = pipeline(image=image, bboxes=bboxes)

        # Bboxes should be transformed by geometric transform
        assert not np.array_equal(result['bboxes'], bboxes)

    def test_compose_factory_function(self):
        """Test compose_transforms factory function."""
        transforms = [
            DummyPhotometricTransform(bbox_format='xywh'),
            DummyGeometricTransform(bbox_format='xywh'),
        ]
        pipeline = compose_transforms(transforms)

        assert isinstance(pipeline, ComposeTransforms)

    def test_compose_preserves_metadata(self):
        """Test that composition preserves metadata."""
        pipeline = ComposeTransforms([
            DummyPhotometricTransform(bbox_format='xywh'),
        ])

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])
        metadata = {'frame_id': 123, 'custom_key': 'value'}

        result = pipeline(image=image, bboxes=bboxes, **metadata)

        assert result['frame_id'] == 123
        assert result['custom_key'] == 'value'

    def test_compose_empty_list(self):
        """Test composing empty transform list."""
        pipeline = ComposeTransforms([])

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = pipeline(image=image, bboxes=bboxes)

        # Should return unchanged
        np.testing.assert_array_equal(result['image'], image)
        np.testing.assert_array_equal(result['bboxes'], bboxes)


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def test_radial_scaling_jitter_scenario(self):
        """
        Test scenario from Audit 2025111801 Finding 2:
        RadialScalingJitter causes bbox desynchronization.
        """
        # Simulate RadialScalingJitter with affine transform
        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[250, 120, 80, 60]])  # Target bbox

        # Scale 1.05x around sonar sensor position (341, 375)
        cx, cy = 341, 375
        scale = 1.05
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)

        # Transform image
        image_transformed = cv2.warpAffine(image, M, (682, 376))

        # Transform bbox using our utility
        bboxes_transformed = transform_bboxes_affine(
            bboxes, M, (376, 682), bbox_format='xywh'
        )

        # Verify bbox is synchronized (width/height scaled by ~1.05)
        np.testing.assert_allclose(
            bboxes_transformed[0, 2:], bboxes[0, 2:] * 1.05, rtol=0.1
        )

    def test_training_pipeline_flow(self):
        """
        Test bbox flow through typical training augmentation pipeline.
        """
        # Build augmentation pipeline
        pipeline = ComposeTransforms([
            DummyPhotometricTransform(bbox_format='xywh', p=1.0),
            DummyGeometricTransform(bbox_format='xywh', p=1.0),
            DummyPhotometricTransform(bbox_format='xywh', p=1.0),
        ])

        # Simulate dataset sample
        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])
        metadata = {'frame_id': 42, 'sequence_name': 'uuv1'}

        # Apply pipeline
        result = pipeline(image=image, bboxes=bboxes, **metadata)

        # Verify outputs
        assert result['image'].shape == image.shape
        assert result['bboxes'].shape == bboxes.shape
        assert result['frame_id'] == 42
        assert result['sequence_name'] == 'uuv1'

    def test_backward_compatibility_migration(self):
        """
        Test incremental migration from legacy to bbox-aware transforms.
        """
        # Mix legacy and new transforms
        legacy_transform = DummyLegacyTransform()
        wrapped_legacy = LegacyTransformWrapper(
            legacy_transform, is_photometric=True, bbox_format='xywh'
        )

        pipeline = ComposeTransforms([
            wrapped_legacy,  # Legacy (wrapped)
            DummyPhotometricTransform(bbox_format='xywh'),  # New bbox-aware
        ])

        image = np.random.randint(0, 255, (376, 682, 3), dtype=np.uint8)
        bboxes = np.array([[100, 200, 50, 80]])

        result = pipeline(image=image, bboxes=bboxes)

        # Should work without errors
        assert 'image' in result
        assert 'bboxes' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
