"""
Unit Tests for Bounding Box Utility Functions

Tests the bbox format conversion and manipulation utilities introduced
in Audit 2025111405 to formalize coordinate format contracts.

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import pytest
import numpy as np
import torch

from aquasot3d.utils.bbox_utils import (
    convert_bbox_format,
    normalize_bbox,
    denormalize_bbox,
    scale_bbox,
    convert_boxes,
)


class TestConvertBboxFormat:
    """Test bbox format conversion between xywh, cxcywh, and xyxy."""

    def test_xywh_to_cxcywh(self):
        """Test converting from top-left format to center format."""
        boxes_xywh = np.array([[100, 200, 50, 80]])
        boxes_cxcywh = convert_bbox_format(boxes_xywh, 'xywh', 'cxcywh')

        expected = np.array([[125, 240, 50, 80]])  # cx=x+w/2, cy=y+h/2
        np.testing.assert_allclose(boxes_cxcywh, expected, rtol=1e-5)

    def test_cxcywh_to_xywh(self):
        """Test converting from center format to top-left format."""
        boxes_cxcywh = np.array([[125, 240, 50, 80]])
        boxes_xywh = convert_bbox_format(boxes_cxcywh, 'cxcywh', 'xywh')

        expected = np.array([[100, 200, 50, 80]])  # x=cx-w/2, y=cy-h/2
        np.testing.assert_allclose(boxes_xywh, expected, rtol=1e-5)

    def test_xywh_to_xyxy(self):
        """Test converting from xywh to corner format."""
        boxes_xywh = np.array([[10, 20, 40, 60]])
        boxes_xyxy = convert_bbox_format(boxes_xywh, 'xywh', 'xyxy')

        expected = np.array([[10, 20, 50, 80]])  # x2=x+w, y2=y+h
        np.testing.assert_allclose(boxes_xyxy, expected, rtol=1e-5)

    def test_xyxy_to_xywh(self):
        """Test converting from corner format to xywh."""
        boxes_xyxy = np.array([[10, 20, 50, 80]])
        boxes_xywh = convert_bbox_format(boxes_xyxy, 'xyxy', 'xywh')

        expected = np.array([[10, 20, 40, 60]])  # w=x2-x1, h=y2-y1
        np.testing.assert_allclose(boxes_xywh, expected, rtol=1e-5)

    def test_same_format_returns_unchanged(self):
        """Test that converting to same format returns unchanged."""
        boxes = np.array([[100, 200, 50, 80]])
        result = convert_bbox_format(boxes, 'xywh', 'xywh')
        np.testing.assert_array_equal(result, boxes)

    def test_batch_conversion(self):
        """Test converting batch of boxes."""
        boxes_xywh = np.array([
            [100, 200, 50, 80],
            [300, 400, 60, 90],
            [500, 600, 70, 100]
        ])
        boxes_cxcywh = convert_bbox_format(boxes_xywh, 'xywh', 'cxcywh')

        expected = np.array([
            [125, 240, 50, 80],
            [330, 445, 60, 90],
            [535, 650, 70, 100]
        ])
        np.testing.assert_allclose(boxes_cxcywh, expected, rtol=1e-5)

    def test_torch_tensor_input(self):
        """Test that torch tensors are handled correctly."""
        boxes_xywh = torch.tensor([[100, 200, 50, 80]], dtype=torch.float32)
        boxes_cxcywh = convert_bbox_format(boxes_xywh, 'xywh', 'cxcywh')

        assert isinstance(boxes_cxcywh, torch.Tensor)
        expected = torch.tensor([[125, 240, 50, 80]], dtype=torch.float32)
        torch.testing.assert_close(boxes_cxcywh, expected)

    def test_roundtrip_conversion(self):
        """Test that xywh -> cxcywh -> xywh is identity."""
        original = np.array([[100, 200, 50, 80]])
        intermediate = convert_bbox_format(original, 'xywh', 'cxcywh')
        roundtrip = convert_bbox_format(intermediate, 'cxcywh', 'xywh')
        np.testing.assert_allclose(roundtrip, original, rtol=1e-5)


class TestNormalizeBbox:
    """Test bbox normalization from pixel to [0, 1] coordinates."""

    def test_normalize_xywh(self):
        """Test normalizing xywh format boxes."""
        boxes = np.array([[100, 200, 50, 80]])
        image_size = (1080, 1920)  # (height, width)

        normalized = normalize_bbox(boxes, image_size, format='xywh')

        expected = np.array([[
            100/1920,  # x / width
            200/1080,  # y / height
            50/1920,   # w / width
            80/1080    # h / height
        ]])
        np.testing.assert_allclose(normalized, expected, rtol=1e-5)

    def test_normalize_cxcywh(self):
        """Test normalizing cxcywh format boxes."""
        boxes = np.array([[960, 540, 192, 324]])  # Center of 1920x1080
        image_size = (1080, 1920)

        normalized = normalize_bbox(boxes, image_size, format='cxcywh')

        expected = np.array([[0.5, 0.5, 0.1, 0.3]])
        np.testing.assert_allclose(normalized, expected, rtol=1e-5)

    def test_normalize_xyxy(self):
        """Test normalizing xyxy format boxes."""
        boxes = np.array([[100, 200, 150, 280]])
        image_size = (1080, 1920)

        normalized = normalize_bbox(boxes, image_size, format='xyxy')

        expected = np.array([[
            100/1920, 200/1080,
            150/1920, 280/1080
        ]])
        np.testing.assert_allclose(normalized, expected, rtol=1e-5)

    def test_normalize_torch_tensor(self):
        """Test normalizing torch tensor input."""
        boxes = torch.tensor([[100.0, 200.0, 50.0, 80.0]])
        image_size = (1080, 1920)

        normalized = normalize_bbox(boxes, image_size, format='xywh')

        assert isinstance(normalized, torch.Tensor)
        assert normalized[0, 0] < 1.0  # Verify it's normalized


class TestDenormalizeBbox:
    """Test bbox denormalization from [0, 1] to pixel coordinates."""

    def test_denormalize_cxcywh(self):
        """Test denormalizing cxcywh format boxes."""
        boxes_norm = np.array([[0.5, 0.5, 0.2, 0.3]])
        image_size = (1080, 1920)

        pixel_boxes = denormalize_bbox(boxes_norm, image_size, format='cxcywh')

        expected = np.array([[960, 540, 384, 324]])
        np.testing.assert_allclose(pixel_boxes, expected, rtol=1e-5)

    def test_denormalize_xywh(self):
        """Test denormalizing xywh format boxes."""
        boxes_norm = np.array([[0.0521, 0.1852, 0.026, 0.074]])
        image_size = (1080, 1920)

        pixel_boxes = denormalize_bbox(boxes_norm, image_size, format='xywh')

        expected = np.array([[100.032, 199.992, 49.92, 79.92]])
        np.testing.assert_allclose(pixel_boxes, expected, rtol=1e-3)

    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize -> denormalize is identity."""
        original = np.array([[100, 200, 50, 80]])
        image_size = (1080, 1920)

        normalized = normalize_bbox(original, image_size, format='xywh')
        roundtrip = denormalize_bbox(normalized, image_size, format='xywh')

        np.testing.assert_allclose(roundtrip, original, rtol=1e-4)


class TestScaleBbox:
    """Test bbox scaling for image resize operations."""

    def test_scale_uniform(self):
        """Test uniform scaling."""
        boxes = np.array([[1000, 500, 200, 100]])
        scale_factor = 0.133  # 255/1920 ≈ 0.133

        scaled = scale_bbox(boxes, scale_factor, format='xywh')

        expected = np.array([[133, 66.5, 26.6, 13.3]])
        np.testing.assert_allclose(scaled, expected, rtol=1e-5)

    def test_scale_non_uniform(self):
        """Test non-uniform scaling with separate x/y factors."""
        boxes = np.array([[100, 200, 50, 80]])
        scale_factor = (2.0, 0.5)  # Double width, half height

        scaled = scale_bbox(boxes, scale_factor, format='xywh')

        expected = np.array([[200, 100, 100, 40]])
        np.testing.assert_allclose(scaled, expected, rtol=1e-5)

    def test_scale_cxcywh(self):
        """Test scaling center format boxes."""
        boxes = np.array([[125, 240, 50, 80]])
        scale_factor = 0.5

        scaled = scale_bbox(boxes, scale_factor, format='cxcywh')

        expected = np.array([[62.5, 120, 25, 40]])
        np.testing.assert_allclose(scaled, expected, rtol=1e-5)

    def test_scale_xyxy(self):
        """Test scaling corner format boxes."""
        boxes = np.array([[10, 20, 50, 80]])
        scale_factor = 2.0

        scaled = scale_bbox(boxes, scale_factor, format='xyxy')

        expected = np.array([[20, 40, 100, 160]])
        np.testing.assert_allclose(scaled, expected, rtol=1e-5)

    def test_scale_torch_tensor(self):
        """Test scaling torch tensor input."""
        boxes = torch.tensor([[100.0, 200.0, 50.0, 80.0]])
        scale_factor = 0.5

        scaled = scale_bbox(boxes, scale_factor, format='xywh')

        assert isinstance(scaled, torch.Tensor)
        expected = torch.tensor([[50.0, 100.0, 25.0, 40.0]])
        torch.testing.assert_close(scaled, expected)


class TestConvertBoxes:
    """Test high-level unified bbox converter API."""

    def test_dataset_to_model_conversion(self):
        """Test typical dataset (pixel xywh) to model (normalized cxcywh) conversion."""
        boxes = np.array([[1000, 500, 200, 100]])
        image_size = (1080, 1920)

        model_boxes = convert_boxes(boxes, 'xywh_px', 'cxcywh_norm', image_size)

        # Expected: xywh -> cxcywh -> normalize
        # xywh [1000, 500, 200, 100] -> cxcywh [1100, 550, 200, 100]
        # Then normalize by (1080, 1920)
        expected = np.array([[1100/1920, 550/1080, 200/1920, 100/1080]])
        np.testing.assert_allclose(model_boxes, expected, rtol=1e-5)

    def test_model_to_visualizer_conversion(self):
        """Test model output (normalized cxcywh) to visualizer (pixel xywh) conversion."""
        pred = np.array([[0.5, 0.5, 0.1, 0.1]])
        image_size = (1080, 1920)

        viz_boxes = convert_boxes(pred, 'cxcywh_norm', 'xywh_px', image_size)

        # Expected: denormalize -> cxcywh to xywh
        # [0.5, 0.5, 0.1, 0.1] * (1920, 1080) = [960, 540, 192, 108]
        # cxcywh [960, 540, 192, 108] -> xywh [864, 486, 192, 108]
        expected = np.array([[864, 486, 192, 108]])
        np.testing.assert_allclose(viz_boxes, expected, rtol=1e-5)

    def test_same_format_and_space(self):
        """Test conversion with same source and destination."""
        boxes = np.array([[100, 200, 50, 80]])
        result = convert_boxes(boxes, 'xywh_px', 'xywh_px')
        np.testing.assert_array_equal(result, boxes)

    def test_format_only_conversion(self):
        """Test converting format without changing coordinate space."""
        boxes = np.array([[0.5, 0.5, 0.2, 0.3]])
        result = convert_boxes(boxes, 'cxcywh_norm', 'xywh_norm')

        # cxcywh [0.5, 0.5, 0.2, 0.3] -> xywh [0.4, 0.35, 0.2, 0.3]
        expected = np.array([[0.4, 0.35, 0.2, 0.3]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_space_only_conversion(self):
        """Test converting coordinate space without changing format."""
        boxes = np.array([[100, 200, 50, 80]])
        image_size = (1080, 1920)

        result = convert_boxes(boxes, 'xywh_px', 'xywh_norm', image_size)

        expected = np.array([[100/1920, 200/1080, 50/1920, 80/1080]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_missing_image_size_raises_error(self):
        """Test that missing image_size for px<->norm conversion raises error."""
        boxes = np.array([[100, 200, 50, 80]])

        with pytest.raises(ValueError, match="image_size required"):
            convert_boxes(boxes, 'xywh_px', 'cxcywh_norm', image_size=None)

    def test_roundtrip_conversion(self):
        """Test full roundtrip: pixel xywh -> norm cxcywh -> pixel xywh."""
        original = np.array([[1000, 500, 200, 100]])
        image_size = (1080, 1920)

        # Convert to model format
        model_boxes = convert_boxes(original, 'xywh_px', 'cxcywh_norm', image_size)
        # Convert back to dataset format
        roundtrip = convert_boxes(model_boxes, 'cxcywh_norm', 'xywh_px', image_size)

        np.testing.assert_allclose(roundtrip, original, rtol=1e-4)

    def test_xyxy_format_support(self):
        """Test that xyxy format is supported."""
        boxes = np.array([[10, 20, 50, 80]])
        image_size = (1080, 1920)

        result = convert_boxes(boxes, 'xyxy_px', 'xywh_norm', image_size)

        # xyxy [10, 20, 50, 80] -> xywh [10, 20, 40, 60] -> normalize
        expected = np.array([[10/1920, 20/1080, 40/1920, 60/1080]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestBboxUtilsIntegration:
    """Integration tests for bbox utilities with real-world scenarios."""

    def test_training_pipeline_bbox_flow(self):
        """
        Test bbox coordinate flow through typical training pipeline.

        Scenario: Dataset (pixel xywh) -> Model (norm cxcywh) -> Loss computation
        """
        # Dataset provides ground truth in pixel xywh
        gt_xywh_px = np.array([[1000, 500, 200, 100]])
        image_size = (1080, 1920)

        # Convert to model format for training
        gt_cxcywh_norm = convert_boxes(gt_xywh_px, 'xywh_px', 'cxcywh_norm', image_size)

        # Verify conversion is correct for loss computation
        assert 0 <= gt_cxcywh_norm[0, 0] <= 1  # Normalized cx in [0, 1]
        assert 0 <= gt_cxcywh_norm[0, 1] <= 1  # Normalized cy in [0, 1]
        assert gt_cxcywh_norm.shape == (1, 4)

    def test_evaluation_pipeline_bbox_flow(self):
        """
        Test bbox coordinate flow through typical evaluation pipeline.

        Scenario: Model output (norm cxcywh) -> Visualizer (pixel xywh)
        """
        # Model predicts in normalized cxcywh
        pred_cxcywh_norm = np.array([[0.5, 0.5, 0.1, 0.1]])
        image_size = (1080, 1920)

        # Convert to pixel xywh for visualization
        pred_xywh_px = convert_boxes(pred_cxcywh_norm, 'cxcywh_norm', 'xywh_px', image_size)

        # Verify result is in pixel coordinates
        assert pred_xywh_px[0, 0] > 1  # x in pixels, not normalized
        assert pred_xywh_px[0, 2] > 1  # w in pixels, not normalized

    def test_image_resize_bbox_scaling(self):
        """
        Test bbox scaling after image resize (Audit 2025111405 Fix #2).

        Scenario: Image resized from 1920x1080 to 255x255, bbox must scale accordingly
        """
        # Original bbox in 1920x1080 image
        bbox_original = np.array([[1000, 500, 200, 100]])

        # Image resized to 255x255, calculate scale factor
        original_size = (1080, 1920)
        target_size = 255
        scale_factor = target_size / ((original_size[0] + original_size[1]) / 2.0)

        # Scale bbox to match resized image
        bbox_scaled = scale_bbox(bbox_original, scale_factor, format='xywh')

        # Verify bbox is scaled proportionally
        assert bbox_scaled[0, 0] < bbox_original[0, 0]  # x scaled down
        assert bbox_scaled[0, 2] < bbox_original[0, 2]  # w scaled down

    def test_batch_processing(self):
        """Test processing batches of boxes efficiently."""
        # Batch of 8 boxes from a training batch
        boxes_batch = np.random.rand(8, 4) * np.array([1920, 1080, 200, 100])
        image_size = (1080, 1920)

        # Convert entire batch
        normalized_batch = convert_boxes(boxes_batch, 'xywh_px', 'cxcywh_norm', image_size)

        # Verify batch dimensions preserved
        assert normalized_batch.shape == (8, 4)
        # Verify all normalized
        assert np.all((normalized_batch >= 0) & (normalized_batch <= 1.5))  # Allow some margin


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
