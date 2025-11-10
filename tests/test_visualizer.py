"""
Unit Tests for Tracking Visualizer

Tests for visualization utilities including frame drawing, trajectory plotting,
success/precision curves, and video generation.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Test with matplotlib in Agg backend to avoid display issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from aquasot3d.evaluation.visualizer import TrackingVisualizer


class TestVisualizerInitialization:
    """Test suite for TrackingVisualizer initialization"""

    def test_basic_initialization(self):
        """Test basic visualizer initialization"""
        visualizer = TrackingVisualizer()

        assert visualizer.save_dir is None
        assert visualizer.dpi == 100
        assert visualizer.font_size == 12
        assert 'pred' in visualizer.colors
        assert 'gt' in visualizer.colors

    def test_initialization_with_save_dir(self):
        """Test initialization with save directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = TrackingVisualizer(save_dir=tmpdir)

            assert visualizer.save_dir is not None
            assert visualizer.save_dir.exists()

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters"""
        visualizer = TrackingVisualizer(dpi=150, font_size=14)

        assert visualizer.dpi == 150
        assert visualizer.font_size == 14

    def test_color_scheme(self):
        """Test that color scheme is properly defined"""
        visualizer = TrackingVisualizer()

        assert len(visualizer.colors['pred']) == 3
        assert len(visualizer.colors['gt']) == 3
        assert len(visualizer.colors['fusion']) == 3
        assert all(0 <= c <= 255 for c in visualizer.colors['pred'])


class TestFrameDrawing:
    """Test suite for frame drawing functions"""

    def test_draw_tracking_result_basic(self):
        """Test basic tracking result drawing"""
        visualizer = TrackingVisualizer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        pred_box = np.array([320, 240, 100, 80])
        gt_box = np.array([320, 240, 100, 80])

        result = visualizer.draw_tracking_result(frame, pred_box, gt_box)

        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_draw_tracking_result_with_frame_id(self):
        """Test drawing with frame ID annotation"""
        visualizer = TrackingVisualizer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        pred_box = np.array([100, 100, 50, 50])
        gt_box = np.array([105, 105, 50, 50])

        result = visualizer.draw_tracking_result(
            frame, pred_box, gt_box, frame_id=42
        )

        assert result.shape == frame.shape
        # Frame should be modified (not identical to input)
        assert not np.array_equal(result, frame)

    def test_draw_tracking_result_none_boxes(self):
        """Test drawing with None boxes"""
        visualizer = TrackingVisualizer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # Only pred box
        result1 = visualizer.draw_tracking_result(frame, pred_box=np.array([100, 100, 50, 50]))
        assert result1.shape == frame.shape

        # Only GT box
        result2 = visualizer.draw_tracking_result(frame, gt_box=np.array([100, 100, 50, 50]))
        assert result2.shape == frame.shape

        # Neither box
        result3 = visualizer.draw_tracking_result(frame)
        assert result3.shape == frame.shape

    def test_draw_tracking_result_different_formats(self):
        """Test drawing with different bbox formats"""
        visualizer = TrackingVisualizer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # cxcywh format
        pred_box = np.array([100, 100, 50, 50])
        result = visualizer.draw_tracking_result(frame, pred_box=pred_box)
        assert result.shape == frame.shape


class TestMultiModalDrawing:
    """Test suite for multi-modal frame drawing"""

    def test_draw_multi_modal_basic(self):
        """Test basic multi-modal drawing"""
        visualizer = TrackingVisualizer()
        rgb_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        sonar_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        rgb_pred = np.array([320, 240, 100, 80])
        sonar_pred = np.array([320, 240, 100, 80])
        fusion_pred = np.array([320, 240, 100, 80])

        rgb_gt = np.array([320, 240, 100, 80])
        sonar_gt = np.array([320, 240, 100, 80])

        result = visualizer.draw_multi_modal_result(
            rgb_frame, sonar_frame,
            rgb_pred, sonar_pred, fusion_pred,
            rgb_gt, sonar_gt
        )

        # Should be concatenated horizontally
        assert result.shape[0] == 480  # Same height
        assert result.shape[1] == 640 * 2  # Double width
        assert result.shape[2] == 3  # RGB

    def test_draw_multi_modal_with_frame_id(self):
        """Test multi-modal drawing with frame ID"""
        visualizer = TrackingVisualizer()
        rgb_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        sonar_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        result = visualizer.draw_multi_modal_result(
            rgb_frame, sonar_frame,
            rgb_pred=np.array([100, 100, 50, 50]),
            frame_id=10
        )

        assert result.shape[1] == 640 * 2

    def test_draw_multi_modal_missing_predictions(self):
        """Test multi-modal drawing with missing predictions"""
        visualizer = TrackingVisualizer()
        rgb_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        sonar_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Should handle missing predictions
        result = visualizer.draw_multi_modal_result(
            rgb_frame, sonar_frame,
            rgb_pred=np.array([100, 100, 50, 50])
        )

        assert result.shape[1] == 640 * 2


class TestTrajectoryPlotting:
    """Test suite for trajectory plotting"""

    def test_plot_trajectory_3d_basic(self):
        """Test basic 3D trajectory plotting"""
        visualizer = TrackingVisualizer()

        centers_3d = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])

        fig = visualizer.plot_trajectory_3d(centers_3d)

        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_trajectory_3d_with_gt(self):
        """Test 3D trajectory plotting with ground truth"""
        visualizer = TrackingVisualizer()

        centers_3d = np.array([[i, i, i] for i in range(10)])
        gt_centers_3d = np.array([[i+0.1, i+0.1, i+0.1] for i in range(10)])

        fig = visualizer.plot_trajectory_3d(centers_3d, gt_centers_3d)

        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_trajectory_3d_save(self):
        """Test saving 3D trajectory plot"""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = TrackingVisualizer(save_dir=tmpdir)

            centers_3d = np.array([[i, i, i] for i in range(5)])
            fig = visualizer.plot_trajectory_3d(centers_3d, filename='trajectory.png')

            save_path = Path(tmpdir) / 'trajectory.png'
            assert save_path.exists()
            plt.close(fig)

    def test_plot_trajectory_3d_empty_input(self):
        """Test 3D trajectory plotting with empty input"""
        visualizer = TrackingVisualizer()

        centers_3d = np.array([]).reshape(0, 3)

        # Should handle gracefully
        fig = visualizer.plot_trajectory_3d(centers_3d)
        assert fig is not None
        plt.close(fig)


class TestSuccessCurvePlotting:
    """Test suite for success curve plotting"""

    def test_plot_success_curves_basic(self):
        """Test basic success curves plotting"""
        visualizer = TrackingVisualizer()

        thresholds = np.linspace(0, 1, 21)
        rgb_curve = (thresholds, np.linspace(1.0, 0.6, 21))
        sonar_curve = (thresholds, np.linspace(1.0, 0.55, 21))
        fusion_curve = (thresholds, np.linspace(1.0, 0.7, 21))

        fig = visualizer.plot_success_curves(rgb_curve, sonar_curve, fusion_curve)

        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_success_curves_with_missing_modality(self):
        """Test success curves with missing modality"""
        visualizer = TrackingVisualizer()

        thresholds = np.linspace(0, 1, 21)
        fusion_curve = (thresholds, np.linspace(1.0, 0.7, 21))

        # Only fusion curve
        fig = visualizer.plot_success_curves(fusion_curve=fusion_curve)

        assert fig is not None
        plt.close(fig)

    def test_plot_success_curves_save(self):
        """Test saving success curves"""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = TrackingVisualizer(save_dir=tmpdir)

            thresholds = np.linspace(0, 1, 21)
            fusion_curve = (thresholds, np.linspace(1.0, 0.7, 21))

            fig = visualizer.plot_success_curves(
                fusion_curve=fusion_curve,
                filename='success_curves.png'
            )

            save_path = Path(tmpdir) / 'success_curves.png'
            assert save_path.exists()
            plt.close(fig)


class TestPrecisionCurvePlotting:
    """Test suite for precision curve plotting"""

    def test_plot_precision_curves_basic(self):
        """Test basic precision curves plotting"""
        visualizer = TrackingVisualizer()

        thresholds = np.linspace(0, 50, 51)
        rgb_curve = (thresholds, np.linspace(0.8, 1.0, 51))
        sonar_curve = (thresholds, np.linspace(0.75, 1.0, 51))
        fusion_curve = (thresholds, np.linspace(0.85, 1.0, 51))

        fig = visualizer.plot_precision_curves(rgb_curve, sonar_curve, fusion_curve)

        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_precision_curves_with_missing_modality(self):
        """Test precision curves with missing modality"""
        visualizer = TrackingVisualizer()

        thresholds = np.linspace(0, 50, 51)
        fusion_curve = (thresholds, np.linspace(0.85, 1.0, 51))

        fig = visualizer.plot_precision_curves(fusion_curve=fusion_curve)

        assert fig is not None
        plt.close(fig)

    def test_plot_precision_curves_save(self):
        """Test saving precision curves"""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = TrackingVisualizer(save_dir=tmpdir)

            thresholds = np.linspace(0, 50, 51)
            fusion_curve = (thresholds, np.linspace(0.85, 1.0, 51))

            fig = visualizer.plot_precision_curves(
                fusion_curve=fusion_curve,
                filename='precision_curves.png'
            )

            save_path = Path(tmpdir) / 'precision_curves.png'
            assert save_path.exists()
            plt.close(fig)


class TestAttentionVisualization:
    """Test suite for attention map visualization"""

    def test_visualize_attention_map_basic(self):
        """Test basic attention map visualization"""
        visualizer = TrackingVisualizer()

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        attention_map = np.random.rand(60, 80)  # Smaller than frame

        result = visualizer.visualize_attention_map(frame, attention_map)

        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_visualize_attention_map_different_sizes(self):
        """Test attention map with different sizes"""
        visualizer = TrackingVisualizer()

        frame = np.ones((256, 256, 3), dtype=np.uint8) * 255
        attention_map = np.random.rand(16, 16)

        result = visualizer.visualize_attention_map(frame, attention_map)

        assert result.shape == frame.shape

    def test_visualize_attention_map_custom_alpha(self):
        """Test attention visualization with custom alpha"""
        visualizer = TrackingVisualizer()

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        attention_map = np.random.rand(60, 80)

        result = visualizer.visualize_attention_map(
            frame, attention_map, alpha=0.3
        )

        assert result.shape == frame.shape

    def test_visualize_attention_map_with_bbox(self):
        """Test attention visualization with bounding box"""
        visualizer = TrackingVisualizer()

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        attention_map = np.random.rand(60, 80)
        pred_box = np.array([320, 240, 100, 80])

        result = visualizer.visualize_attention_map(
            frame, attention_map, pred_box=pred_box
        )

        assert result.shape == frame.shape


class TestFailureCaseReport:
    """Test suite for failure case report generation"""

    def test_create_failure_case_report_basic(self):
        """Test basic failure case report"""
        visualizer = TrackingVisualizer()

        frames = [np.ones((480, 640, 3), dtype=np.uint8) * 255 for _ in range(10)]
        pred_boxes = [np.array([100+i*5, 100+i*5, 50, 50]) for i in range(10)]
        gt_boxes = [np.array([100, 100, 50, 50]) for _ in range(10)]
        metrics = {
            'success_auc': 0.45,
            'avg_center_error': 25.3,
            'mean_iou': 0.38
        }

        fig = visualizer.create_failure_case_report(
            'test_sequence', frames, pred_boxes, gt_boxes, metrics
        )

        assert fig is not None
        assert len(fig.axes) >= 4  # Start, middle, end, worst
        plt.close(fig)

    def test_create_failure_case_report_save(self):
        """Test saving failure case report"""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = TrackingVisualizer(save_dir=tmpdir)

            frames = [np.ones((480, 640, 3), dtype=np.uint8) * 255 for _ in range(10)]
            pred_boxes = [np.array([100, 100, 50, 50]) for _ in range(10)]
            gt_boxes = [np.array([100, 100, 50, 50]) for _ in range(10)]
            metrics = {'success_auc': 0.45}

            fig = visualizer.create_failure_case_report(
                'test_seq', frames, pred_boxes, gt_boxes, metrics,
                filename='failure_report.png'
            )

            save_path = Path(tmpdir) / 'failure_report.png'
            assert save_path.exists()
            plt.close(fig)

    def test_create_failure_case_report_short_sequence(self):
        """Test failure report with short sequence"""
        visualizer = TrackingVisualizer()

        # Only 3 frames
        frames = [np.ones((480, 640, 3), dtype=np.uint8) * 255 for _ in range(3)]
        pred_boxes = [np.array([100, 100, 50, 50]) for _ in range(3)]
        gt_boxes = [np.array([100, 100, 50, 50]) for _ in range(3)]
        metrics = {'success_auc': 0.45}

        # Should handle gracefully
        fig = visualizer.create_failure_case_report(
            'short_seq', frames, pred_boxes, gt_boxes, metrics
        )

        assert fig is not None
        plt.close(fig)


class TestVideoCreation:
    """Test suite for video creation"""

    @pytest.mark.skipif(True, reason="Video creation requires cv2.VideoWriter which may not be available")
    def test_create_video_from_frames_basic(self):
        """Test basic video creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = TrackingVisualizer(save_dir=tmpdir)

            frames = [
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                for _ in range(10)
            ]

            output_path = visualizer.create_video_from_frames(
                frames, 'test_video.mp4', fps=10
            )

            # Note: Actual video creation may fail if cv2 not properly configured
            # This test mainly checks the interface
            assert output_path is not None

    def test_create_video_empty_frames(self):
        """Test video creation with empty frames list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = TrackingVisualizer(save_dir=tmpdir)

            frames = []

            # Should handle gracefully (likely return None or raise informative error)
            try:
                result = visualizer.create_video_from_frames(frames, 'empty.mp4')
                # If it doesn't raise, result should be None or handle gracefully
                assert result is None or isinstance(result, (str, Path))
            except ValueError:
                # Acceptable to raise ValueError for empty frames
                pass


class TestVisualizeBatchResults:
    """Test suite for batch result visualization"""

    def test_visualize_batch_results_basic(self):
        """Test visualizing batch results"""
        visualizer = TrackingVisualizer()

        rgb_frames = [np.ones((480, 640, 3), dtype=np.uint8) * 255 for _ in range(5)]
        sonar_frames = [np.ones((480, 640, 3), dtype=np.uint8) * 128 for _ in range(5)]
        predictions = {
            'rgb_pred': [np.array([100, 100, 50, 50]) for _ in range(5)],
            'sonar_pred': [np.array([100, 100, 50, 50]) for _ in range(5)],
            'fusion_pred': [np.array([100, 100, 50, 50]) for _ in range(5)],
            'rgb_gt': [np.array([100, 100, 50, 50]) for _ in range(5)],
            'sonar_gt': [np.array([100, 100, 50, 50]) for _ in range(5)]
        }

        results = visualizer.visualize_batch_results(
            rgb_frames, sonar_frames, predictions
        )

        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(frame, np.ndarray) for frame in results)

    def test_visualize_batch_results_save(self):
        """Test saving batch results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = TrackingVisualizer(save_dir=tmpdir)

            rgb_frames = [np.ones((480, 640, 3), dtype=np.uint8) * 255 for _ in range(3)]
            sonar_frames = [np.ones((480, 640, 3), dtype=np.uint8) * 128 for _ in range(3)]
            predictions = {
                'fusion_pred': [np.array([100, 100, 50, 50]) for _ in range(3)]
            }

            results = visualizer.visualize_batch_results(
                rgb_frames, sonar_frames, predictions,
                save_frames=True, prefix='batch'
            )

            # Check that frames were saved
            save_dir = Path(tmpdir)
            saved_frames = list(save_dir.glob('batch_*.png'))
            assert len(saved_frames) > 0


class TestEdgeCases:
    """Test suite for edge cases and error handling"""

    def test_invalid_frame_shape(self):
        """Test handling of invalid frame shapes"""
        visualizer = TrackingVisualizer()

        # 2D grayscale frame
        frame_2d = np.ones((480, 640), dtype=np.uint8) * 255
        pred_box = np.array([100, 100, 50, 50])

        # Should handle or convert to RGB
        result = visualizer.draw_tracking_result(frame_2d, pred_box=pred_box)
        assert result is not None
        assert len(result.shape) == 3  # Should be RGB

    def test_out_of_bounds_bbox(self):
        """Test handling of out-of-bounds bounding boxes"""
        visualizer = TrackingVisualizer()

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        # Box extends beyond frame boundaries
        pred_box = np.array([600, 450, 200, 200])

        # Should handle gracefully (clipping or warning)
        result = visualizer.draw_tracking_result(frame, pred_box=pred_box)
        assert result.shape == frame.shape

    def test_zero_size_bbox(self):
        """Test handling of zero-size bounding box"""
        visualizer = TrackingVisualizer()

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        pred_box = np.array([100, 100, 0, 0])

        # Should handle gracefully
        result = visualizer.draw_tracking_result(frame, pred_box=pred_box)
        assert result.shape == frame.shape

    def test_negative_bbox_dimensions(self):
        """Test handling of negative bbox dimensions"""
        visualizer = TrackingVisualizer()

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        pred_box = np.array([100, 100, -50, -50])

        # Should handle gracefully
        result = visualizer.draw_tracking_result(frame, pred_box=pred_box)
        assert result.shape == frame.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
