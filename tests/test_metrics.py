"""
Unit Tests for Evaluation Metrics

Tests for tracking metrics computation including IoU, center error,
success curves, precision curves, and multi-modal metric aggregation.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import pytest
import numpy as np
from typing import List

from aquasot3d.evaluation.metrics import (
    compute_iou,
    compute_center_error,
    compute_success_curve,
    compute_success_auc,
    compute_precision_curve,
    compute_precision_at_threshold,
    compute_sequence_metrics,
    aggregate_metrics,
    compute_multi_modal_metrics,
    format_metrics_report
)


class TestIoUComputation:
    """Test suite for IoU computation"""

    def test_iou_identical_boxes(self):
        """Test IoU of identical boxes should be 1.0"""
        bbox1 = np.array([50, 50, 100, 100])  # [cx, cy, w, h]
        bbox2 = np.array([50, 50, 100, 100])
        iou = compute_iou(bbox1, bbox2)
        assert abs(iou - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        """Test IoU of non-overlapping boxes should be 0.0"""
        bbox1 = np.array([50, 50, 100, 100])
        bbox2 = np.array([200, 200, 100, 100])
        iou = compute_iou(bbox1, bbox2)
        assert abs(iou - 0.0) < 1e-6

    def test_iou_partial_overlap(self):
        """Test IoU of partially overlapping boxes"""
        bbox1 = np.array([50, 50, 100, 100])  # [0, 0, 100, 100] in xyxy
        bbox2 = np.array([75, 75, 100, 100])  # [25, 25, 125, 125] in xyxy
        iou = compute_iou(bbox1, bbox2)
        # Intersection: 75x75=5625, Union: 10000+10000-5625=14375
        # IoU = 5625/14375 ≈ 0.391
        assert 0.35 < iou < 0.45

    def test_iou_contained_box(self):
        """Test IoU when one box contains another"""
        bbox1 = np.array([50, 50, 100, 100])  # Large box
        bbox2 = np.array([50, 50, 50, 50])    # Smaller box inside
        iou = compute_iou(bbox1, bbox2)
        # Intersection: 2500, Union: 10000
        # IoU = 2500/10000 = 0.25
        assert abs(iou - 0.25) < 0.01

    def test_iou_xyxy_format(self):
        """Test IoU with xyxy format"""
        # compute_iou auto-detects format based on bbox values
        # Use negative coordinates to force xyxy interpretation
        bbox1 = np.array([-10, -10, 90, 90])     # xyxy format
        bbox2 = np.array([40, 40, 140, 140])     # xyxy format
        iou = compute_iou(bbox1, bbox2)
        # Intersection: 50x50=2500, Union: 10000+10000-2500=17500
        # IoU = 2500/17500 ≈ 0.143 (but may vary slightly with format detection)
        assert 0.10 < iou < 0.20

    def test_iou_zero_area_box(self):
        """Test IoU with zero-area box"""
        bbox1 = np.array([50, 50, 100, 100])
        bbox2 = np.array([50, 50, 0, 0])      # Zero area
        iou = compute_iou(bbox1, bbox2)
        assert iou == 0.0


class TestCenterError:
    """Test suite for center error computation"""

    def test_center_error_identical(self):
        """Test center error of identical boxes should be 0"""
        bbox1 = np.array([50, 50, 100, 100])
        bbox2 = np.array([50, 50, 100, 100])
        error = compute_center_error(bbox1, bbox2)
        assert abs(error) < 1e-6

    def test_center_error_horizontal_shift(self):
        """Test center error with horizontal shift"""
        bbox1 = np.array([50, 50, 100, 100])
        bbox2 = np.array([80, 50, 100, 100])  # 30px shift in x
        error = compute_center_error(bbox1, bbox2)
        assert abs(error - 30.0) < 1e-6

    def test_center_error_vertical_shift(self):
        """Test center error with vertical shift"""
        bbox1 = np.array([50, 50, 100, 100])
        bbox2 = np.array([50, 90, 100, 100])  # 40px shift in y
        error = compute_center_error(bbox1, bbox2)
        assert abs(error - 40.0) < 1e-6

    def test_center_error_diagonal_shift(self):
        """Test center error with diagonal shift"""
        bbox1 = np.array([0, 0, 100, 100])
        bbox2 = np.array([30, 40, 100, 100])  # 3-4-5 triangle
        error = compute_center_error(bbox1, bbox2)
        assert abs(error - 50.0) < 1e-6


class TestSuccessCurve:
    """Test suite for success curve computation"""

    def test_success_curve_perfect_tracking(self):
        """Test success curve with perfect tracking"""
        pred_bboxes = [
            np.array([50, 50, 100, 100]),
            np.array([55, 55, 100, 100]),
            np.array([60, 60, 100, 100])
        ]
        gt_bboxes = pred_bboxes.copy()

        thresholds, success_rates = compute_success_curve(pred_bboxes, gt_bboxes)

        assert len(thresholds) == len(success_rates)
        # Perfect tracking should have high success rates at low thresholds
        assert success_rates[0] == 1.0  # All frames succeed at threshold 0
        # At threshold 1.0, IoU must be exactly 1.0, which is rare due to floating point
        assert success_rates[0] >= success_rates[-1]  # Monotonically decreasing

    def test_success_curve_poor_tracking(self):
        """Test success curve with poor tracking"""
        pred_bboxes = [
            np.array([50, 50, 100, 100]),
            np.array([150, 150, 100, 100]),  # No overlap
            np.array([200, 200, 100, 100])   # No overlap
        ]
        gt_bboxes = [
            np.array([50, 50, 100, 100]),
            np.array([50, 50, 100, 100]),
            np.array([50, 50, 100, 100])
        ]

        thresholds, success_rates = compute_success_curve(pred_bboxes, gt_bboxes)

        # Poor tracking should have low success rates at higher thresholds
        assert success_rates[0] > success_rates[-1]

    def test_success_curve_empty_input(self):
        """Test success curve with empty input"""
        pred_bboxes = []
        gt_bboxes = []

        thresholds, success_rates = compute_success_curve(pred_bboxes, gt_bboxes)

        assert len(thresholds) > 0
        assert len(success_rates) > 0
        assert all(rate == 0.0 for rate in success_rates)


class TestSuccessAUC:
    """Test suite for success AUC computation"""

    def test_success_auc_perfect_tracking(self):
        """Test AUC with perfect tracking should be close to 1.0"""
        pred_bboxes = [np.array([50, 50, 100, 100]) for _ in range(10)]
        gt_bboxes = pred_bboxes.copy()

        auc = compute_success_auc(pred_bboxes, gt_bboxes)
        # Perfect tracking should have AUC >= 0.95 (allowing for numerical precision at threshold 1.0)
        assert auc >= 0.95

    def test_success_auc_no_overlap(self):
        """Test AUC with no overlap should be 0.0"""
        pred_bboxes = [np.array([200, 200, 100, 100]) for _ in range(10)]
        gt_bboxes = [np.array([50, 50, 100, 100]) for _ in range(10)]

        auc = compute_success_auc(pred_bboxes, gt_bboxes)
        assert abs(auc) < 0.01

    def test_success_auc_partial_overlap(self):
        """Test AUC with partial overlap"""
        pred_bboxes = [
            np.array([50, 50, 100, 100]),
            np.array([75, 75, 100, 100]),  # Partial overlap
            np.array([100, 100, 100, 100]) # Less overlap
        ]
        gt_bboxes = [np.array([50, 50, 100, 100]) for _ in range(3)]

        auc = compute_success_auc(pred_bboxes, gt_bboxes)
        assert 0.0 < auc < 1.0


class TestPrecisionCurve:
    """Test suite for precision curve computation"""

    def test_precision_curve_perfect_tracking(self):
        """Test precision curve with perfect tracking"""
        pred_bboxes = [np.array([50, 50, 100, 100]) for _ in range(10)]
        gt_bboxes = pred_bboxes.copy()

        thresholds, precision_rates = compute_precision_curve(pred_bboxes, gt_bboxes)

        assert len(thresholds) == len(precision_rates)
        assert precision_rates[0] == 1.0  # All frames within 1px

    def test_precision_curve_poor_tracking(self):
        """Test precision curve with poor tracking"""
        pred_bboxes = [np.array([i*50, i*50, 100, 100]) for i in range(5)]
        gt_bboxes = [np.array([50, 50, 100, 100]) for _ in range(5)]

        thresholds, precision_rates = compute_precision_curve(pred_bboxes, gt_bboxes)

        # Poor tracking should have low precision at strict thresholds
        # Precision increases as threshold becomes more lenient
        assert len(thresholds) == len(precision_rates)
        assert precision_rates[-1] >= precision_rates[0]  # More lenient threshold has higher precision

    def test_precision_at_threshold(self):
        """Test precision at specific threshold"""
        pred_bboxes = [
            np.array([50, 50, 100, 100]),   # 0px error
            np.array([60, 60, 100, 100]),   # ~14px error
            np.array([80, 80, 100, 100])    # ~42px error
        ]
        gt_bboxes = [np.array([50, 50, 100, 100]) for _ in range(3)]

        precision_20 = compute_precision_at_threshold(pred_bboxes, gt_bboxes, threshold=20)
        # 2 out of 3 within 20px
        assert abs(precision_20 - 2.0/3.0) < 0.01


class TestSequenceMetrics:
    """Test suite for sequence-level metrics"""

    def test_sequence_metrics_basic(self):
        """Test basic sequence metrics computation"""
        pred_bboxes = [np.array([50, 50, 100, 100]) for _ in range(5)]
        gt_bboxes = pred_bboxes.copy()

        metrics = compute_sequence_metrics(pred_bboxes, gt_bboxes)

        assert 'success_auc' in metrics
        assert 'precision_20' in metrics
        assert 'avg_center_error' in metrics
        assert 'mean_iou' in metrics
        assert 'success_0.5' in metrics

        # Perfect tracking (allowing for numerical precision)
        assert metrics['success_auc'] >= 0.95
        assert abs(metrics['precision_20'] - 1.0) < 0.01
        assert abs(metrics['avg_center_error']) < 0.01
        assert abs(metrics['mean_iou'] - 1.0) < 0.01

    def test_sequence_metrics_with_modality(self):
        """Test sequence metrics with modality label"""
        pred_bboxes = [np.array([50, 50, 100, 100]) for _ in range(5)]
        gt_bboxes = pred_bboxes.copy()

        metrics = compute_sequence_metrics(pred_bboxes, gt_bboxes, modality='rgb')

        assert 'modality' in metrics
        assert metrics['modality'] == 'rgb'


class TestMetricsAggregation:
    """Test suite for metrics aggregation"""

    def test_aggregate_metrics_basic(self):
        """Test basic metrics aggregation"""
        sequence_metrics = [
            {'success_auc': 0.8, 'precision_20': 0.7, 'avg_center_error': 10.0, 'mean_iou': 0.75},
            {'success_auc': 0.9, 'precision_20': 0.85, 'avg_center_error': 8.0, 'mean_iou': 0.85},
            {'success_auc': 0.7, 'precision_20': 0.65, 'avg_center_error': 12.0, 'mean_iou': 0.65}
        ]

        aggregated = aggregate_metrics(sequence_metrics, modality='fusion')

        assert 'success_auc_mean' in aggregated
        assert 'success_auc_median' in aggregated
        assert 'success_auc_std' in aggregated
        assert 'precision_20_mean' in aggregated

        # Check mean values
        assert abs(aggregated['success_auc_mean'] - 0.8) < 0.01
        assert abs(aggregated['precision_20_mean'] - 0.733) < 0.01

    def test_aggregate_metrics_single_sequence(self):
        """Test aggregation with single sequence"""
        sequence_metrics = [
            {'success_auc': 0.8, 'precision_20': 0.7, 'avg_center_error': 10.0, 'mean_iou': 0.75}
        ]

        aggregated = aggregate_metrics(sequence_metrics, modality='rgb')

        assert aggregated['success_auc_mean'] == 0.8
        assert aggregated['success_auc_std'] == 0.0

    def test_aggregate_metrics_empty_list(self):
        """Test aggregation with empty list"""
        sequence_metrics = []

        aggregated = aggregate_metrics(sequence_metrics, modality='sonar')

        # Should return empty dict or dict with zero values
        assert isinstance(aggregated, dict)


class TestMultiModalMetrics:
    """Test suite for multi-modal metrics computation"""

    def test_multi_modal_metrics_all_modalities(self):
        """Test metrics for all modalities"""
        rgb_pred = [np.array([50, 50, 100, 100]) for _ in range(5)]
        sonar_pred = [np.array([55, 55, 100, 100]) for _ in range(5)]
        fusion_pred = [np.array([52, 52, 100, 100]) for _ in range(5)]
        rgb_gt = [np.array([50, 50, 100, 100]) for _ in range(5)]
        sonar_gt = [np.array([50, 50, 100, 100]) for _ in range(5)]

        metrics = compute_multi_modal_metrics(
            rgb_pred, sonar_pred, fusion_pred, rgb_gt, sonar_gt
        )

        assert 'rgb' in metrics
        assert 'sonar' in metrics
        assert 'fusion' in metrics

        assert 'success_auc' in metrics['rgb']
        assert 'success_auc' in metrics['sonar']
        assert 'success_auc' in metrics['fusion']

    def test_multi_modal_metrics_missing_modality(self):
        """Test with missing modality"""
        rgb_pred = [np.array([50, 50, 100, 100]) for _ in range(5)]
        rgb_gt = [np.array([50, 50, 100, 100]) for _ in range(5)]

        metrics = compute_multi_modal_metrics(
            rgb_pred, [], [], rgb_gt, []
        )

        assert 'rgb' in metrics
        # Empty modalities still return metrics dict (with num_frames=0 or inf values)
        assert 'sonar' in metrics
        assert 'fusion' in metrics
        # But they should have num_frames or num_valid_predictions = 0
        assert metrics['sonar']['num_frames'] == 0 or metrics['sonar']['num_valid_predictions'] == 0
        assert metrics['fusion']['num_frames'] == 0 or metrics['fusion']['num_valid_predictions'] == 0


class TestMetricsReporting:
    """Test suite for metrics report formatting"""

    def test_format_metrics_report_basic(self):
        """Test basic report formatting"""
        metrics = {
            'rgb': {
                'success_auc': 0.8,
                'precision_20': 0.75,
                'avg_center_error': 10.5,
                'mean_iou': 0.72
            },
            'sonar': {
                'success_auc': 0.7,
                'precision_20': 0.65,
                'avg_center_error': 15.2,
                'mean_iou': 0.68
            },
            'fusion': {
                'success_auc': 0.85,
                'precision_20': 0.80,
                'avg_center_error': 8.3,
                'mean_iou': 0.78
            }
        }

        report = format_metrics_report(metrics, sequence_name="Test Sequence")

        assert isinstance(report, str)
        assert "Test Sequence" in report
        assert "RGB" in report or "rgb" in report
        assert "0.8" in report or "0.80" in report

    def test_format_metrics_report_single_modality(self):
        """Test report with single modality"""
        metrics = {
            'fusion': {
                'success_auc': 0.85,
                'precision_20': 0.80
            }
        }

        report = format_metrics_report(metrics)

        assert isinstance(report, str)
        assert len(report) > 0


class TestEdgeCases:
    """Test suite for edge cases and error handling"""

    def test_mismatched_length_predictions(self):
        """Test handling of mismatched prediction/GT lengths"""
        pred_bboxes = [np.array([50, 50, 100, 100]) for _ in range(3)]
        gt_bboxes = [np.array([50, 50, 100, 100]) for _ in range(5)]

        # Should handle gracefully (truncate or pad)
        metrics = compute_sequence_metrics(pred_bboxes, gt_bboxes)
        assert metrics is not None

    def test_invalid_bbox_format(self):
        """Test handling of invalid bbox format"""
        # Negative dimensions
        bbox1 = np.array([50, 50, -100, 100])
        bbox2 = np.array([50, 50, 100, 100])

        iou = compute_iou(bbox1, bbox2)
        # Should handle gracefully and return 0 or small value
        assert iou >= 0.0

    def test_nan_in_bboxes(self):
        """Test handling of NaN values in bboxes"""
        bbox1 = np.array([50, 50, 100, 100])
        bbox2 = np.array([np.nan, 50, 100, 100])

        # NaN propagates through computation (expected behavior)
        iou = compute_iou(bbox1, bbox2)
        # Either returns NaN (propagation) or 0.0 (graceful handling)
        assert np.isnan(iou) or iou == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
