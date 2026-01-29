"""
Test evaluation pipeline following got10k-toolkit structure.

This module tests:
1. Tracker base interface
2. AquaSOTTracker wrapper
3. Metrics (IoU, center error, curves)
4. ExperimentAquaSOT for evaluation
"""

import os
import sys
import pytest
import torch
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# ==============================================================================
# Metrics Tests
# ==============================================================================

class TestMetrics:
    """Test evaluation metrics."""

    def test_metrics_import(self):
        """Test that metrics can be imported."""
        from lib.test.evaluation import rect_iou, center_error
        assert rect_iou is not None
        assert center_error is not None

    def test_rect_iou_identical(self):
        """Test IoU of identical boxes."""
        from lib.test.evaluation import rect_iou

        box = np.array([[10, 20, 30, 40]])
        iou = rect_iou(box, box)

        assert len(iou) == 1
        assert np.isclose(iou[0], 1.0)

    def test_rect_iou_no_overlap(self):
        """Test IoU of non-overlapping boxes."""
        from lib.test.evaluation import rect_iou

        box1 = np.array([[0, 0, 10, 10]])
        box2 = np.array([[100, 100, 10, 10]])
        iou = rect_iou(box1, box2)

        assert len(iou) == 1
        assert np.isclose(iou[0], 0.0)

    def test_rect_iou_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        from lib.test.evaluation import rect_iou

        box1 = np.array([[0, 0, 10, 10]])
        box2 = np.array([[5, 5, 10, 10]])
        iou = rect_iou(box1, box2)

        # Intersection: 5x5 = 25
        # Union: 100 + 100 - 25 = 175
        # IoU: 25/175 = 0.1428...
        assert len(iou) == 1
        assert np.isclose(iou[0], 25 / 175, atol=1e-4)

    def test_rect_iou_batch(self):
        """Test IoU for batch of boxes."""
        from lib.test.evaluation import rect_iou

        boxes1 = np.array([
            [0, 0, 10, 10],
            [0, 0, 10, 10],
            [0, 0, 10, 10],
        ])
        boxes2 = np.array([
            [0, 0, 10, 10],   # Identical
            [5, 5, 10, 10],   # Partial overlap
            [100, 100, 10, 10],  # No overlap
        ])
        ious = rect_iou(boxes1, boxes2)

        assert len(ious) == 3
        assert np.isclose(ious[0], 1.0)
        assert np.isclose(ious[1], 25 / 175, atol=1e-4)
        assert np.isclose(ious[2], 0.0)

    def test_center_error_identical(self):
        """Test center error of identical boxes."""
        from lib.test.evaluation import center_error

        box = np.array([[10, 20, 30, 40]])
        error = center_error(box, box)

        assert len(error) == 1
        assert np.isclose(error[0], 0.0)

    def test_center_error_offset(self):
        """Test center error with known offset."""
        from lib.test.evaluation import center_error

        # Center of box1: (15, 25)
        box1 = np.array([[10, 20, 10, 10]])
        # Center of box2: (18, 29) -> offset (3, 4)
        box2 = np.array([[13, 24, 10, 10]])
        error = center_error(box1, box2)

        # Euclidean distance: sqrt(3^2 + 4^2) = 5
        assert len(error) == 1
        assert np.isclose(error[0], 5.0)

    def test_success_curve(self):
        """Test success curve computation."""
        from lib.test.evaluation import success_curve

        ious = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        thresholds = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        curve = success_curve(ious, thresholds)

        # At threshold 0.0: all pass -> 1.0
        # At threshold 0.25: 4/5 pass -> 0.8
        # At threshold 0.5: 3/5 pass -> 0.6
        # At threshold 0.75: 1/5 pass -> 0.2
        # At threshold 1.0: 1/5 pass -> 0.2
        assert len(curve) == 5
        assert np.isclose(curve[0], 1.0)
        assert np.isclose(curve[1], 0.8)
        assert np.isclose(curve[2], 0.6)

    def test_precision_curve(self):
        """Test precision curve computation."""
        from lib.test.evaluation import precision_curve

        errors = np.array([0, 10, 20, 30, 40])
        thresholds = np.array([0, 10, 20, 30, 40])
        curve = precision_curve(errors, thresholds)

        # At threshold 0: 1/5 pass -> 0.2
        # At threshold 10: 2/5 pass -> 0.4
        # At threshold 20: 3/5 pass -> 0.6
        # At threshold 30: 4/5 pass -> 0.8
        # At threshold 40: 5/5 pass -> 1.0
        assert len(curve) == 5
        assert np.isclose(curve[0], 0.2)
        assert np.isclose(curve[1], 0.4)
        assert np.isclose(curve[2], 0.6)

    def test_calculate_auc(self):
        """Test AUC calculation."""
        from lib.test.evaluation.metrics import calculate_auc

        # Constant curve = 0.5 should give AUC = 0.5
        curve = np.ones(101) * 0.5
        auc = calculate_auc(curve)
        assert np.isclose(auc, 0.5, atol=1e-4)

        # Perfect curve (all 1s) should give AUC = 1.0
        curve = np.ones(101)
        auc = calculate_auc(curve)
        assert np.isclose(auc, 1.0, atol=1e-4)


# ==============================================================================
# Tracker Interface Tests
# ==============================================================================

class TestTrackerInterface:
    """Test tracker base interface."""

    def test_base_tracker_import(self):
        """Test that base tracker can be imported."""
        from lib.test.tracker import Tracker
        assert Tracker is not None

    def test_base_tracker_abstract(self):
        """Test that base tracker is abstract."""
        from lib.test.tracker import Tracker

        with pytest.raises(TypeError):
            Tracker('test')  # Should fail - abstract class

    def test_dummy_tracker(self):
        """Test implementing a dummy tracker."""
        from lib.test.tracker import Tracker

        class DummyTracker(Tracker):
            def __init__(self):
                super().__init__('dummy', is_deterministic=True)
                self.init_box = None

            def init(self, image, box):
                self.init_box = box

            def update(self, image):
                return self.init_box  # Return same box

        tracker = DummyTracker()
        assert tracker.name == 'dummy'
        assert tracker.is_deterministic

        # Test init and update
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        init_box = [10, 20, 30, 40]

        tracker.init(dummy_image, init_box)
        result = tracker.update(dummy_image)

        assert np.array_equal(result, init_box)


# ==============================================================================
# AquaSOTTracker Tests
# ==============================================================================

class TestAquaSOTTracker:
    """Test AquaSOT tracker wrapper."""

    @pytest.fixture(autouse=True)
    def clear_pretrained(self):
        """Clear pretrained weights config for tests."""
        from lib.config.aquasot.config import cfg
        # Save original value
        original = cfg.MODEL.PRETRAIN_FILE
        # Clear for test
        cfg.MODEL.PRETRAIN_FILE = ""
        yield
        # Restore
        cfg.MODEL.PRETRAIN_FILE = original

    def test_tracker_import(self):
        """Test that AquaSOTTracker can be imported."""
        from lib.test.tracker import AquaSOTTracker
        assert AquaSOTTracker is not None

    def test_tracker_creation(self):
        """Test creating AquaSOTTracker instance."""
        from lib.test.tracker import AquaSOTTracker

        params = {
            'search_area_factor': 4.0,
            'template_area_factor': 2.0,
            'search_size': 256,
            'template_size': 128,
        }

        # Should create without checkpoint (random weights)
        tracker = AquaSOTTracker(params)
        assert tracker.name == 'AquaSOT'

    def test_tracker_init_update(self):
        """Test tracker init and update methods."""
        from lib.test.tracker import AquaSOTTracker

        params = {
            'search_size': 256,
            'template_size': 128,
        }

        tracker = AquaSOTTracker(params)

        # Create dummy images
        rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        sonar_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

        image = {'rgb': rgb_image, 'sonar': sonar_image}
        init_box = [200, 200, 50, 50]  # [x, y, w, h]

        # Initialize
        tracker.init(image, init_box)
        assert tracker.state is not None
        assert tracker.frame_id == 0

        # Update
        result = tracker.update(image)

        # Result is now a dict with dual-modal predictions
        assert isinstance(result, dict)
        assert 'rgb_bbox' in result
        assert 'sonar_bbox' in result
        assert len(result['rgb_bbox']) == 4  # [x, y, w, h]
        assert tracker.frame_id == 1

    def test_tracker_rgb_only(self):
        """Test tracker with RGB only (no sonar)."""
        from lib.test.tracker import AquaSOTTracker

        params = {
            'search_size': 256,
            'template_size': 128,
        }

        tracker = AquaSOTTracker(params)

        # RGB only image
        rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        init_box = [200, 200, 50, 50]

        # Should work with RGB only
        tracker.init(rgb_image, init_box)
        result = tracker.update(rgb_image)

        # Result is a dict with dual-modal predictions
        assert isinstance(result, dict)
        assert len(result['rgb_bbox']) == 4

    def test_coordinate_conversion(self):
        """Test coordinate conversion utilities."""
        from lib.test.tracker import AquaSOTTracker

        params = {}
        tracker = AquaSOTTracker(params)

        # Test xywh to cxcywh
        xywh = np.array([10, 20, 30, 40])
        cxcywh = tracker._xywh_to_cxcywh(xywh)
        expected = np.array([25, 40, 30, 40])  # cx=10+15, cy=20+20
        assert np.allclose(cxcywh, expected)

        # Test cxcywh to xywh (inverse)
        back = tracker._cxcywh_to_xywh(cxcywh)
        assert np.allclose(back, xywh)


# ==============================================================================
# Experiment Tests
# ==============================================================================

class TestExperimentAquaSOT:
    """Test experiment class for evaluation."""

    def test_experiment_import(self):
        """Test that ExperimentAquaSOT can be imported."""
        from lib.test.evaluation import ExperimentAquaSOT
        assert ExperimentAquaSOT is not None

    def test_experiment_creation(self, tmp_path):
        """Test creating experiment instance."""
        from lib.test.evaluation import ExperimentAquaSOT

        # Create mock dataset directory
        dataset_dir = tmp_path / 'dataset'
        dataset_dir.mkdir()

        exp = ExperimentAquaSOT(
            root_dir=str(dataset_dir),
            result_dir=str(tmp_path / 'results'),
            report_dir=str(tmp_path / 'reports'),
        )

        assert exp.root_dir == str(dataset_dir)
        assert os.path.exists(str(tmp_path / 'results'))
        assert os.path.exists(str(tmp_path / 'reports'))

    def test_experiment_with_mock_data(self, tmp_path):
        """Test experiment with mock dataset."""
        import cv2
        from lib.test.evaluation import ExperimentAquaSOT
        from lib.test.tracker import Tracker

        # Create mock sequence (actual AquaSOT structure)
        seq_path = tmp_path / 'dataset' / 'test_seq'
        rgb_path = seq_path / 'light' / 'img'
        sonar_path = seq_path / 'sonar' / 'img_rect'
        rgb_path.mkdir(parents=True)
        sonar_path.mkdir(parents=True)

        # Create mock images (10 frames)
        for i in range(10):
            rgb_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(rgb_path / f'{i:06d}.jpg'), rgb_img)
            sonar_img = np.random.randint(0, 255, (376, 682), dtype=np.uint8)
            cv2.imwrite(str(sonar_path / f'{i:06d}.jpg'), sonar_img)

        # Create ground truth (actual structure)
        with open(seq_path / 'light' / 'groundtruth.txt', 'w') as f:
            for i in range(10):
                f.write(f'{200},{200},{50},{50}\n')

        with open(seq_path / 'sonar' / 'groundtruth_rect.txt', 'w') as f:
            for i in range(10):
                f.write(f'{200},{200},{50},{50}\n')

        # Create dummy tracker
        class DummyTracker(Tracker):
            def __init__(self):
                super().__init__('DummyTracker', is_deterministic=True)
                self.init_box = None

            def init(self, image, box):
                # Handle dict format (dual-modal) or array format (legacy)
                if isinstance(box, dict):
                    self.init_box = np.array(box.get('rgb', box.get('sonar')))
                else:
                    self.init_box = np.array(box)

            def update(self, image):
                # Add small noise to box
                return self.init_box + np.random.randn(4) * 2

        # Create experiment
        exp = ExperimentAquaSOT(
            root_dir=str(tmp_path / 'dataset'),
            result_dir=str(tmp_path / 'results'),
            report_dir=str(tmp_path / 'reports'),
        )

        assert exp.sequence_names == ['test_seq']

        # Run tracker
        tracker = DummyTracker()
        exp.run(tracker, visualize=False)

        # Check results saved
        result_file = tmp_path / 'results' / 'DummyTracker' / 'test_seq_rgb.txt'
        assert result_file.exists()

        # Generate report
        performance = exp.report(['DummyTracker'])

        assert 'DummyTracker' in performance
        assert 'overall' in performance['DummyTracker']
        assert 'success_auc' in performance['DummyTracker']['overall']

    def test_load_groundtruth(self, tmp_path):
        """Test ground truth loading."""
        from lib.test.evaluation import ExperimentAquaSOT

        exp = ExperimentAquaSOT(
            root_dir=str(tmp_path),
            result_dir=str(tmp_path / 'results'),
            report_dir=str(tmp_path / 'reports'),
        )

        # Create test groundtruth file
        gt_file = tmp_path / 'test_gt.txt'
        with open(gt_file, 'w') as f:
            f.write('10,20,30,40\n')
            f.write('15,25,35,45\n')
            f.write('20,30,40,50\n')

        gt = exp._load_groundtruth(str(gt_file))

        assert gt is not None
        assert gt.shape == (3, 4)
        assert np.allclose(gt[0], [10, 20, 30, 40])
        assert np.allclose(gt[2], [20, 30, 40, 50])


# ==============================================================================
# Visibility Filtering Tests
# ==============================================================================

class TestVisibilityFiltering:
    """Test visibility filtering in evaluation."""

    def test_compute_visibility_mask_zero_boxes(self):
        """Test visibility mask detection for (0,0,0,0) boxes."""
        from lib.test.evaluation import ExperimentAquaSOT

        exp = ExperimentAquaSOT(root_dir='/tmp', result_dir='/tmp/r', report_dir='/tmp/p')

        gt = np.array([[0, 0, 0, 0], [10, 20, 30, 40], [0, 0, 0, 0], [50, 60, 70, 80]])
        vis = exp._compute_visibility_mask(gt)

        assert vis.tolist() == [False, True, False, True]

    def test_compute_visibility_mask_invalid_size(self):
        """Test visibility mask detection for invalid size boxes."""
        from lib.test.evaluation import ExperimentAquaSOT

        exp = ExperimentAquaSOT(root_dir='/tmp', result_dir='/tmp/r', report_dir='/tmp/p')

        # Invalid: w <= 0 or h <= 0
        gt = np.array([
            [10, 20, -5, 40],    # Invalid: negative width
            [10, 20, 30, 0],     # Invalid: zero height
            [10, 20, 0, 40],     # Invalid: zero width
            [10, 20, 30, 40],    # Valid
        ])
        vis = exp._compute_visibility_mask(gt)

        assert vis.tolist() == [False, False, False, True]

    def test_compute_visibility_mask_empty(self):
        """Test visibility mask with empty array."""
        from lib.test.evaluation import ExperimentAquaSOT

        exp = ExperimentAquaSOT(root_dir='/tmp', result_dir='/tmp/r', report_dir='/tmp/p')

        gt = np.array([]).reshape(0, 4)
        vis = exp._compute_visibility_mask(gt)

        assert len(vis) == 0

    def test_compute_visibility_mask_none(self):
        """Test visibility mask with None input."""
        from lib.test.evaluation import ExperimentAquaSOT

        exp = ExperimentAquaSOT(root_dir='/tmp', result_dir='/tmp/r', report_dir='/tmp/p')

        vis = exp._compute_visibility_mask(None)

        assert len(vis) == 0

    def test_filter_visibility_enabled(self, tmp_path):
        """Test that visibility filtering excludes absent frames."""
        import cv2
        from lib.test.evaluation import ExperimentAquaSOT
        from lib.test.tracker import Tracker

        # Create mock sequence with some absent frames
        seq_path = tmp_path / 'dataset' / 'test_seq'
        rgb_path = seq_path / 'light' / 'img'
        sonar_path = seq_path / 'sonar' / 'img_rect'
        rgb_path.mkdir(parents=True)
        sonar_path.mkdir(parents=True)

        # Create 5 frames
        for i in range(5):
            rgb_img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            cv2.imwrite(str(rgb_path / f'{i:06d}.jpg'), rgb_img)
            sonar_img = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
            cv2.imwrite(str(sonar_path / f'{i:06d}.jpg'), sonar_img)

        # Create GT with 2 absent frames (0,0,0,0)
        with open(seq_path / 'light' / 'groundtruth.txt', 'w') as f:
            f.write('100,100,40,40\n')  # Frame 0: visible (but skipped as init)
            f.write('0,0,0,0\n')        # Frame 1: absent
            f.write('100,100,40,40\n')  # Frame 2: visible
            f.write('0,0,0,0\n')        # Frame 3: absent
            f.write('100,100,40,40\n')  # Frame 4: visible

        with open(seq_path / 'sonar' / 'groundtruth_rect.txt', 'w') as f:
            for i in range(5):
                f.write('100,100,40,40\n')  # All visible in sonar

        # Simple tracker
        class SimpleTracker(Tracker):
            def __init__(self):
                super().__init__('SimpleTracker', is_deterministic=True)
                self.box = None

            def init(self, image, box):
                if isinstance(box, dict):
                    self.box = np.array(box.get('rgb', box.get('sonar')))
                else:
                    self.box = np.array(box)

            def update(self, image):
                return self.box.copy()

        # Run with filter_visibility=True (default)
        exp = ExperimentAquaSOT(
            root_dir=str(tmp_path / 'dataset'),
            result_dir=str(tmp_path / 'results'),
            report_dir=str(tmp_path / 'reports'),
            filter_visibility=True,
            skip_init_frame=True,
        )

        tracker = SimpleTracker()
        exp.run(tracker)
        performance = exp.report(['SimpleTracker'])

        # With visibility filtering:
        # - Frame 0: skipped (init)
        # - Frame 1: absent in RGB, excluded from RGB metrics
        # - Frame 2: visible
        # - Frame 3: absent in RGB, excluded from RGB metrics
        # - Frame 4: visible
        # RGB visible frames = 2 (frames 2, 4)
        # Sonar visible frames = 4 (frames 1, 2, 3, 4)

        rgb_vis = performance['SimpleTracker']['rgb']['visible_frames']
        sonar_vis = performance['SimpleTracker']['sonar']['visible_frames']

        assert rgb_vis == 2, f'Expected 2 RGB visible frames, got {rgb_vis}'
        assert sonar_vis == 4, f'Expected 4 Sonar visible frames, got {sonar_vis}'

    def test_filter_visibility_disabled(self, tmp_path):
        """Test that visibility filtering can be disabled."""
        import cv2
        from lib.test.evaluation import ExperimentAquaSOT
        from lib.test.tracker import Tracker

        # Create mock sequence
        seq_path = tmp_path / 'dataset' / 'test_seq'
        rgb_path = seq_path / 'light' / 'img'
        sonar_path = seq_path / 'sonar' / 'img_rect'
        rgb_path.mkdir(parents=True)
        sonar_path.mkdir(parents=True)

        for i in range(5):
            rgb_img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            cv2.imwrite(str(rgb_path / f'{i:06d}.jpg'), rgb_img)
            sonar_img = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
            cv2.imwrite(str(sonar_path / f'{i:06d}.jpg'), sonar_img)

        # GT with 2 absent frames
        with open(seq_path / 'light' / 'groundtruth.txt', 'w') as f:
            f.write('100,100,40,40\n')
            f.write('0,0,0,0\n')
            f.write('100,100,40,40\n')
            f.write('0,0,0,0\n')
            f.write('100,100,40,40\n')

        with open(seq_path / 'sonar' / 'groundtruth_rect.txt', 'w') as f:
            for i in range(5):
                f.write('100,100,40,40\n')

        class SimpleTracker(Tracker):
            def __init__(self):
                super().__init__('SimpleTracker', is_deterministic=True)
                self.box = None

            def init(self, image, box):
                if isinstance(box, dict):
                    self.box = np.array(box.get('rgb', box.get('sonar')))
                else:
                    self.box = np.array(box)

            def update(self, image):
                return self.box.copy()

        # Run with filter_visibility=False
        exp = ExperimentAquaSOT(
            root_dir=str(tmp_path / 'dataset'),
            result_dir=str(tmp_path / 'results2'),
            report_dir=str(tmp_path / 'reports2'),
            filter_visibility=False,
            skip_init_frame=True,
        )

        tracker = SimpleTracker()
        exp.run(tracker)
        performance = exp.report(['SimpleTracker'])

        # Without visibility filtering:
        # All 4 frames after init should be counted
        rgb_vis = performance['SimpleTracker']['rgb']['visible_frames']

        assert rgb_vis == 4, f'Expected 4 RGB visible frames (filter disabled), got {rgb_vis}'


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""

    def test_full_evaluation_pipeline(self, tmp_path):
        """Full integration test of evaluation pipeline."""
        import cv2
        from lib.test.tracker import Tracker
        from lib.test.evaluation import ExperimentAquaSOT, rect_iou

        # Create mock dataset with 2 sequences (actual AquaSOT structure)
        for seq_idx in range(2):
            seq_name = f'seq_{seq_idx}'
            seq_path = tmp_path / 'dataset' / seq_name
            rgb_path = seq_path / 'light' / 'img'
            sonar_path = seq_path / 'sonar' / 'img_rect'
            rgb_path.mkdir(parents=True)
            sonar_path.mkdir(parents=True)

            for i in range(5):
                img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                cv2.imwrite(str(rgb_path / f'{i:06d}.jpg'), img)
                sonar_img = np.random.randint(0, 255, (376, 682), dtype=np.uint8)
                cv2.imwrite(str(sonar_path / f'{i:06d}.jpg'), sonar_img)

            with open(seq_path / 'light' / 'groundtruth.txt', 'w') as f:
                for i in range(5):
                    f.write(f'{100+i},{100+i},{40},{40}\n')

            with open(seq_path / 'sonar' / 'groundtruth_rect.txt', 'w') as f:
                for i in range(5):
                    f.write(f'{100+i},{100+i},{40},{40}\n')

        # Create a tracker that follows GT with noise
        class NoisyFollower(Tracker):
            def __init__(self):
                super().__init__('NoisyFollower', is_deterministic=False)
                self.box = None
                self.frame = 0

            def init(self, image, box):
                # Handle dict format (dual-modal) or array format (legacy)
                if isinstance(box, dict):
                    init_box = box.get('rgb', box.get('sonar'))
                else:
                    init_box = box
                self.box = np.array(init_box, dtype=np.float32)
                self.frame = 0

            def update(self, image):
                self.frame += 1
                # Move box slightly (simulating tracking)
                self.box[:2] += 1 + np.random.randn(2) * 2
                return self.box.copy()

        # Run evaluation
        exp = ExperimentAquaSOT(
            root_dir=str(tmp_path / 'dataset'),
            result_dir=str(tmp_path / 'results'),
            report_dir=str(tmp_path / 'reports'),
        )

        tracker = NoisyFollower()
        exp.run(tracker)

        # Generate report
        performance = exp.report(['NoisyFollower'])

        # Verify metrics are computed
        assert 'NoisyFollower' in performance
        overall = performance['NoisyFollower']['overall']
        assert 0 <= overall['success_auc'] <= 1
        assert 0 <= overall['precision_score'] <= 1

        # Check curves are computed
        assert 'success_curve' in performance['NoisyFollower']
        assert 'precision_curve' in performance['NoisyFollower']
        assert len(performance['NoisyFollower']['success_curve']) == 101
        assert len(performance['NoisyFollower']['precision_curve']) == 51

    def test_metrics_consistency(self):
        """Test that metrics are consistent with manual calculation."""
        from lib.test.evaluation import rect_iou, center_error, success_curve

        # Create predictable data
        gt_boxes = np.array([
            [100, 100, 50, 50],
            [110, 110, 50, 50],
            [120, 120, 50, 50],
        ])

        # Perfect predictions
        pred_perfect = gt_boxes.copy()
        ious_perfect = rect_iou(pred_perfect, gt_boxes)
        errors_perfect = center_error(pred_perfect, gt_boxes)

        assert np.allclose(ious_perfect, 1.0)
        assert np.allclose(errors_perfect, 0.0)

        # Predictions with known offset
        pred_offset = gt_boxes.copy()
        pred_offset[:, 0] += 10  # Shift x by 10
        errors_offset = center_error(pred_offset, gt_boxes)

        # Center error should be 10 for all
        assert np.allclose(errors_offset, 10.0)


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
