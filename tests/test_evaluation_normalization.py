"""
Unit tests for evaluation coordinate normalization

This test suite verifies that evaluation metrics correctly handle
coordinate denormalization from [0,1] to pixel space.

Critical tests:
1. Bbox denormalization: verify correct conversion from normalized to pixel coords
2. Metrics consistency: precision@20 should work correctly with pixel coords
3. Center error: should report pixel distances, not normalized distances
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock

from src.aquasot3d.evaluation.evaluator import TrackingEvaluator
from src.aquasot3d.evaluation.metrics import (
    compute_center_error,
    compute_precision_at_threshold,
    compute_sequence_metrics
)


class TestBboxDenormalization:
    """Test bounding box denormalization"""

    def test_denormalize_bbox_basic(self):
        """Test basic bbox denormalization"""
        # Create a mock evaluator just to access the method
        mock_model = Mock()
        mock_loader = Mock()
        evaluator = TrackingEvaluator(
            model=mock_model,
            dataloader=mock_loader,
            search_size=256
        )

        # Normalized bbox (center at 0.5, 0.5, size 0.2, 0.2)
        bbox_norm = np.array([0.5, 0.5, 0.2, 0.2])
        bbox_pixel = evaluator._denormalize_bbox(bbox_norm, 256)

        # Expected pixel coordinates
        expected = np.array([128.0, 128.0, 51.2, 51.2])

        np.testing.assert_allclose(bbox_pixel, expected, rtol=1e-5)

    def test_denormalize_bbox_already_pixel(self):
        """Test that already denormalized boxes are not modified"""
        mock_model = Mock()
        mock_loader = Mock()
        evaluator = TrackingEvaluator(
            model=mock_model,
            dataloader=mock_loader,
            search_size=256
        )

        # Already in pixel coordinates
        bbox_pixel = np.array([128.0, 128.0, 50.0, 50.0])
        bbox_result = evaluator._denormalize_bbox(bbox_pixel, 256)

        np.testing.assert_array_equal(bbox_result, bbox_pixel)

    def test_denormalize_none(self):
        """Test handling of None bbox"""
        mock_model = Mock()
        mock_loader = Mock()
        evaluator = TrackingEvaluator(
            model=mock_model,
            dataloader=mock_loader,
            search_size=256
        )

        result = evaluator._denormalize_bbox(None, 256)
        assert result is None


class TestCenterErrorCalculation:
    """Test center error calculation with different coordinate systems"""

    def test_center_error_normalized_vs_pixel(self):
        """
        Test that center error scales correctly between normalized and pixel coords.
        """
        # Two boxes with 0.1 normalized distance (25.6 pixels at 256x256)
        bbox1_norm = np.array([0.5, 0.5, 0.2, 0.2])
        bbox2_norm = np.array([0.6, 0.5, 0.2, 0.2])  # 0.1 distance in x

        # Convert to pixel coordinates (256x256)
        bbox1_pixel = bbox1_norm * 256
        bbox2_pixel = bbox2_norm * 256

        # Compute center errors
        error_norm = compute_center_error(bbox1_norm, bbox2_norm)
        error_pixel = compute_center_error(bbox1_pixel, bbox2_pixel)

        # Expected relationship: error_pixel = error_norm * 256
        expected_pixel_error = error_norm * 256

        assert abs(error_pixel - expected_pixel_error) < 1e-3, \
            f"Pixel error {error_pixel:.2f} != normalized error {error_norm:.4f} * 256 = {expected_pixel_error:.2f}"

        # Verify pixel error is reasonable
        assert error_pixel == pytest.approx(25.6, rel=0.01), \
            f"Expected ~25.6 pixels, got {error_pixel:.2f}"

    def test_center_error_pixel_threshold(self):
        """
        Test that precision@20 works correctly with pixel coordinates.
        """
        # Create predictions and GTs in pixel coordinates
        # All predictions within 20 pixels
        pred_boxes = [
            np.array([100, 100, 40, 40]),
            np.array([120, 120, 40, 40]),  # 20*sqrt(2) ≈ 28.3 pixels away
            np.array([110, 110, 40, 40]),  # 10*sqrt(2) ≈ 14.1 pixels away
        ]

        gt_boxes = [
            np.array([100, 100, 40, 40]),
            np.array([100, 100, 40, 40]),
            np.array([100, 100, 40, 40]),
        ]

        # First two: 0 and ~28.3 pixels
        # Within 20px threshold: 2 out of 3 (first and third)
        precision = compute_precision_at_threshold(pred_boxes, gt_boxes, threshold=20.0)

        # Expected: 2/3 ≈ 0.667
        assert precision == pytest.approx(2.0/3.0, abs=0.01), \
            f"Expected precision ~0.667, got {precision:.4f}"


class TestPrecisionConsistency:
    """Test precision computation consistency"""

    def test_precision_with_normalized_coords_gives_misleading_results(self):
        """
        Verify that normalized coordinates give misleading results for precision@20.

        This test demonstrates why we need to denormalize before computing metrics.
        """
        # Same boxes as above, but in normalized coordinates [0, 1]
        pred_boxes_norm = [
            np.array([0.390625, 0.390625, 0.15625, 0.15625]),  # 100/256
            np.array([0.46875, 0.46875, 0.15625, 0.15625]),    # 120/256
            np.array([0.4296875, 0.4296875, 0.15625, 0.15625]), # 110/256
        ]

        gt_boxes_norm = [
            np.array([0.390625, 0.390625, 0.15625, 0.15625]),
            np.array([0.390625, 0.390625, 0.15625, 0.15625]),
            np.array([0.390625, 0.390625, 0.15625, 0.15625]),
        ]

        # Computing precision@20 with normalized coords (MISLEADING)
        precision_misleading = compute_precision_at_threshold(
            pred_boxes_norm, gt_boxes_norm, threshold=20.0
        )

        # With normalized coords, all errors are < 0.2, which is << 20.0
        # So precision@20 = 1.0 (all "within 20 pixels" but scale is wrong!)
        assert precision_misleading == pytest.approx(1.0, abs=0.01), \
            "Normalized coords give precision@20 = 1.0 (misleading due to scale mismatch)"

    def test_precision_with_pixel_coords_correct(self):
        """
        Verify that pixel coordinates give correct results for precision@20.

        This test verifies the fix works.
        """
        # Same boxes in pixel coordinates
        pred_boxes_pixel = [
            np.array([100, 100, 40, 40]),
            np.array([120, 120, 40, 40]),
            np.array([110, 110, 40, 40]),
        ]

        gt_boxes_pixel = [
            np.array([100, 100, 40, 40]),
            np.array([100, 100, 40, 40]),
            np.array([100, 100, 40, 40]),
        ]

        # Computing precision@20 with pixel coords (FIXED)
        precision_correct = compute_precision_at_threshold(
            pred_boxes_pixel, gt_boxes_pixel, threshold=20.0
        )

        # Expected: 2 out of 3 within 20 pixels
        assert precision_correct == pytest.approx(2.0/3.0, abs=0.01), \
            f"Pixel coords should give correct precision@20 (~0.667), got {precision_correct:.4f}"


class TestSequenceMetricsIntegration:
    """Test complete sequence metrics computation"""

    def test_sequence_metrics_with_pixel_coords(self):
        """
        Test that compute_sequence_metrics works correctly with pixel coordinates.
        """
        # Create a small sequence with pixel coordinates
        pred_boxes = [
            np.array([128, 128, 50, 50]),
            np.array([130, 130, 50, 50]),  # 2.83 pixels error
            np.array([150, 150, 50, 50]),  # 31.1 pixels error
            np.array([135, 135, 50, 50]),  # 9.9 pixels error
        ]

        gt_boxes = [
            np.array([128, 128, 50, 50]),
            np.array([128, 128, 50, 50]),
            np.array([128, 128, 50, 50]),
            np.array([128, 128, 50, 50]),
        ]

        metrics = compute_sequence_metrics(pred_boxes, gt_boxes, modality='test')

        # Check precision@20: 3 out of 4 within 20 pixels
        assert metrics['precision_20'] == pytest.approx(0.75, abs=0.01), \
            f"Expected precision@20 = 0.75, got {metrics['precision_20']:.4f}"

        # Check average center error: (0 + 2.83 + 31.1 + 9.9) / 4 ≈ 10.96 pixels
        assert 5 < metrics['avg_center_error'] < 15, \
            f"Expected avg error ~11 pixels, got {metrics['avg_center_error']:.2f}"

        # Check mean IoU (boxes are same size, so IoU depends on overlap)
        assert 0.5 < metrics['mean_iou'] < 1.0, \
            f"Expected mean IoU in [0.5, 1.0], got {metrics['mean_iou']:.4f}"

    def test_sequence_metrics_detects_normalized_coords(self):
        """
        Verify that normalized coords would produce misleading metrics.

        This documents the expected behavior difference.
        """
        # Same boxes in normalized coordinates
        pred_boxes_norm = [
            np.array([0.5, 0.5, 0.195, 0.195]),
            np.array([0.508, 0.508, 0.195, 0.195]),
            np.array([0.586, 0.586, 0.195, 0.195]),
            np.array([0.527, 0.527, 0.195, 0.195]),
        ]

        gt_boxes_norm = [
            np.array([0.5, 0.5, 0.195, 0.195]),
            np.array([0.5, 0.5, 0.195, 0.195]),
            np.array([0.5, 0.5, 0.195, 0.195]),
            np.array([0.5, 0.5, 0.195, 0.195]),
        ]

        metrics_norm = compute_sequence_metrics(pred_boxes_norm, gt_boxes_norm, modality='test')

        # With normalized coords, all errors < 0.2, which is << 20.0
        # So precision@20 = 1.0 (all "within 20 pixels" but scale is completely wrong!)
        assert metrics_norm['precision_20'] == pytest.approx(1.0, abs=0.01), \
            "Normalized coords give precision@20 = 1.0 (misleading - scale mismatch)"

        # Average error would be tiny (~0.04) instead of ~11 pixels
        assert metrics_norm['avg_center_error'] < 0.2, \
            f"Normalized coords give tiny error (<0.2), got {metrics_norm['avg_center_error']:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])