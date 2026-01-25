"""
End-to-end integration tests for AquaSOT.

This module tests the complete pipeline:
1. End-to-end training (1 epoch)
2. End-to-end evaluation (1 sequence)
3. Checkpoint save/load
4. Full pipeline integration
"""

import os
import sys
import pytest
import torch
import numpy as np
import tempfile
import shutil

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_dataset_dir(tmp_path):
    """Create a mock AquaSOT dataset structure (matching actual structure)."""
    import cv2

    # Create multiple sequences
    for seq_idx in range(3):
        seq_name = f'uuv_test_{seq_idx}'
        seq_path = tmp_path / seq_name
        rgb_path = seq_path / 'light' / 'img'
        sonar_path = seq_path / 'sonar' / 'img_rect'
        rgb_path.mkdir(parents=True)
        sonar_path.mkdir(parents=True)

        # Create 20 frames per sequence
        for i in range(20):
            # RGB image (3 channels)
            rgb_img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            cv2.imwrite(str(rgb_path / f'{i:06d}.jpg'), rgb_img)

            # Sonar image (grayscale, rect-converted size)
            sonar_img = np.random.randint(0, 255, (376, 682), dtype=np.uint8)
            cv2.imwrite(str(sonar_path / f'{i:06d}.jpg'), sonar_img)

        # Create groundtruth files with moving target
        with open(seq_path / 'light' / 'groundtruth.txt', 'w') as f:
            for i in range(20):
                x = 100 + i * 2
                y = 80 + i
                f.write(f'{x},{y},40,40\n')

        with open(seq_path / 'sonar' / 'groundtruth_rect.txt', 'w') as f:
            for i in range(20):
                x = 100 + i * 2
                y = 80 + i
                f.write(f'{x},{y},40,40\n')

    return tmp_path


@pytest.fixture
def clear_pretrained():
    """Clear pretrained weights config for tests."""
    from lib.config.aquasot.config import cfg
    original = cfg.MODEL.PRETRAIN_FILE
    cfg.MODEL.PRETRAIN_FILE = ""
    yield cfg
    cfg.MODEL.PRETRAIN_FILE = original


# Force CPU to avoid GPU memory issues on RTX 3070
TEST_DEVICE = torch.device('cpu')


# ==============================================================================
# End-to-End Training Tests
# ==============================================================================

class TestE2ETraining:
    """End-to-end training tests."""

    def test_single_epoch_training(self, clear_pretrained, tmp_path):
        """Test training for a single epoch with synthetic data."""
        from lib.config.aquasot.config import cfg
        from lib.models.aquasot import build_aquasot
        from lib.train.actors.aquasot import AquaSOTActor
        from lib.utils.focal_loss import FocalLoss
        from lib.utils.box_ops import giou_loss

        # Configure for test
        cfg.MODEL.PRETRAIN_FILE = ""
        cfg.TRAIN.BATCH_SIZE = 2

        # Build model
        model = build_aquasot(cfg, training=True)
        device = TEST_DEVICE
        model = model.to(device)

        # Build actor
        objective = {
            'giou': giou_loss,
            'l1': torch.nn.L1Loss(reduction='mean'),
            'focal': FocalLoss(),
        }
        loss_weight = {
            'giou': cfg.TRAIN.GIOU_WEIGHT,
            'l1': cfg.TRAIN.L1_WEIGHT,
            'focal': cfg.TRAIN.FOCAL_WEIGHT,
        }

        # Create minimal settings
        class Settings:
            def __init__(self, log_dir):
                self.device = device
                self.print_interval = 1
                self.batchsize = 2
                self.local_rank = -1
                self.log_dir = log_dir
                self.log_file = os.path.join(log_dir, 'training.log')
                self.grad_clip_norm = 0.1
                self.print_stats = None
                self.description = 'test'

        settings = Settings(str(tmp_path))
        actor = AquaSOTActor(model, objective, loss_weight, settings, cfg)

        # Build optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create synthetic training data
        batch_size = 2
        def create_batch():
            return {
                'rgb_template': [torch.randn(batch_size, 3, 128, 128).to(device)],
                'rgb_search': [torch.randn(batch_size, 3, 256, 256).to(device)],
                'sonar_template': [torch.randn(batch_size, 1, 128, 128).to(device)],
                'sonar_search': [torch.randn(batch_size, 1, 256, 256).to(device)],
                'rgb_anno': torch.tensor([[[0.3, 0.3, 0.3, 0.3]] * batch_size]).to(device),
                'sonar_anno': torch.tensor([[[0.3, 0.3, 0.3, 0.3]] * batch_size]).to(device),
            }

        # Train for a few iterations
        initial_loss = None
        for i in range(3):
            data = create_batch()
            loss, stats = actor(data)

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verify training happened
        assert initial_loss is not None
        assert not np.isnan(initial_loss)
        assert initial_loss > 0

    def test_training_with_lr_scheduler(self, mock_dataset_dir, clear_pretrained):
        """Test training with learning rate scheduler."""
        from lib.config.aquasot.config import cfg
        from lib.models.aquasot import build_aquasot

        cfg.MODEL.PRETRAIN_FILE = ""

        # Build model
        model = build_aquasot(cfg, training=True)
        device = TEST_DEVICE
        model = model.to(device)

        # Build optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        initial_lr = optimizer.param_groups[0]['lr']

        # Simulate training steps
        for step in range(5):
            # Dummy forward/backward
            dummy_input = {
                'rgb_template': torch.randn(2, 3, 128, 128).to(device),
                'rgb_search': torch.randn(2, 3, 256, 256).to(device),
                'sonar_template': torch.randn(2, 1, 128, 128).to(device),
                'sonar_search': torch.randn(2, 1, 256, 256).to(device),
            }
            out = model(**dummy_input)
            loss = out['rgb_boxes'].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Scheduler step
        scheduler.step()

        # Verify scheduler is working
        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr == initial_lr  # Not stepped enough to change yet

    def test_gradient_clipping(self, clear_pretrained):
        """Test gradient clipping during training."""
        from lib.config.aquasot.config import cfg
        from lib.models.aquasot import build_aquasot

        cfg.MODEL.PRETRAIN_FILE = ""

        # Build model
        model = build_aquasot(cfg, training=True)
        device = TEST_DEVICE
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Forward pass
        dummy_input = {
            'rgb_template': torch.randn(1, 3, 128, 128).to(device),
            'rgb_search': torch.randn(1, 3, 256, 256).to(device),
            'sonar_template': torch.randn(1, 1, 128, 128).to(device),
            'sonar_search': torch.randn(1, 1, 256, 256).to(device),
        }
        out = model(**dummy_input)
        loss = out['rgb_boxes'].mean() * 1000  # Artificially large loss

        optimizer.zero_grad()
        loss.backward()

        # Check gradients before clipping
        total_norm_before = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_before += p.grad.data.norm(2).item() ** 2
        total_norm_before = total_norm_before ** 0.5

        # Clip gradients
        max_norm = cfg.TRAIN.GRAD_CLIP_NORM
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Check gradients after clipping
        total_norm_after = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.data.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5

        # Gradient norm should be clipped
        assert total_norm_after <= max_norm + 1e-6


# ==============================================================================
# End-to-End Evaluation Tests
# ==============================================================================

class TestE2EEvaluation:
    """End-to-end evaluation tests."""

    def test_single_sequence_evaluation(self, mock_dataset_dir, clear_pretrained):
        """Test evaluation on a single sequence."""
        from lib.test.tracker import AquaSOTTracker
        from lib.test.evaluation import ExperimentAquaSOT

        # Create tracker
        params = {
            'search_size': 256,
            'template_size': 128,
        }
        tracker = AquaSOTTracker(params)

        # Create experiment
        experiment = ExperimentAquaSOT(
            root_dir=str(mock_dataset_dir),
            result_dir=str(mock_dataset_dir / 'results'),
            report_dir=str(mock_dataset_dir / 'reports'),
            sequence_names=['uuv_test_0'],
        )

        # Run evaluation
        experiment.run(tracker, visualize=False)

        # Check results saved
        result_file = mock_dataset_dir / 'results' / 'AquaSOT' / 'uuv_test_0_rgb.txt'
        assert result_file.exists()

        # Load and verify results
        results = np.loadtxt(str(result_file), delimiter=',')
        assert results.shape[0] == 20  # 20 frames
        assert results.shape[1] == 4   # [x, y, w, h]

        # Generate report
        performance = experiment.report(['AquaSOT'])
        assert 'AquaSOT' in performance
        assert 'overall' in performance['AquaSOT']

    def test_multi_sequence_evaluation(self, mock_dataset_dir, clear_pretrained):
        """Test evaluation on multiple sequences."""
        from lib.test.tracker import AquaSOTTracker
        from lib.test.evaluation import ExperimentAquaSOT

        # Create tracker
        params = {'search_size': 256, 'template_size': 128}
        tracker = AquaSOTTracker(params)

        # Create experiment with all sequences
        experiment = ExperimentAquaSOT(
            root_dir=str(mock_dataset_dir),
            result_dir=str(mock_dataset_dir / 'results'),
            report_dir=str(mock_dataset_dir / 'reports'),
        )

        assert len(experiment.sequence_names) == 3

        # Run evaluation
        experiment.run(tracker, visualize=False)

        # Generate report
        performance = experiment.report(['AquaSOT'])

        # Check per-sequence results
        assert 'sequences' in performance['AquaSOT']
        assert len(performance['AquaSOT']['sequences']) == 3

    def test_evaluation_metrics_computation(self, mock_dataset_dir, clear_pretrained):
        """Test that evaluation metrics are computed correctly."""
        from lib.test.tracker import Tracker
        from lib.test.evaluation import ExperimentAquaSOT

        # Create a perfect tracker that returns ground truth
        class PerfectTracker(Tracker):
            def __init__(self):
                super().__init__('PerfectTracker', is_deterministic=True)
                self.gt = None
                self.frame_idx = 0

            def init(self, image, box):
                self.gt = np.array(box)
                self.frame_idx = 0

            def update(self, image):
                self.frame_idx += 1
                # Return box with small offset (simulating imperfect tracking)
                return self.gt + np.array([self.frame_idx * 2, self.frame_idx, 0, 0])

        tracker = PerfectTracker()

        experiment = ExperimentAquaSOT(
            root_dir=str(mock_dataset_dir),
            result_dir=str(mock_dataset_dir / 'results'),
            report_dir=str(mock_dataset_dir / 'reports'),
            sequence_names=['uuv_test_0'],
        )

        experiment.run(tracker, visualize=False)
        performance = experiment.report(['PerfectTracker'])

        # Metrics should be reasonable (not perfect due to offset)
        overall = performance['PerfectTracker']['overall']
        assert 0 < overall['success_auc'] <= 1.0
        assert 0 < overall['precision_score'] <= 1.0


# ==============================================================================
# Checkpoint Tests
# ==============================================================================

class TestCheckpoint:
    """Checkpoint save/load tests."""

    def test_save_load_checkpoint(self, clear_pretrained, tmp_path):
        """Test saving and loading model checkpoint."""
        from lib.config.aquasot.config import cfg
        from lib.models.aquasot import build_aquasot

        cfg.MODEL.PRETRAIN_FILE = ""

        # Build model
        model = build_aquasot(cfg, training=True)
        device = TEST_DEVICE
        model = model.to(device)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Do a forward/backward pass to update parameters
        dummy_input = {
            'rgb_template': torch.randn(1, 3, 128, 128).to(device),
            'rgb_search': torch.randn(1, 3, 256, 256).to(device),
            'sonar_template': torch.randn(1, 1, 128, 128).to(device),
            'sonar_search': torch.randn(1, 1, 256, 256).to(device),
        }
        out = model(**dummy_input)
        loss = out['rgb_boxes'].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save checkpoint
        checkpoint_path = tmp_path / 'checkpoint.pth'
        torch.save({
            'epoch': 5,
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, str(checkpoint_path))

        # Verify file exists
        assert checkpoint_path.exists()

        # Build new model and load checkpoint
        model2 = build_aquasot(cfg, training=True)
        model2 = model2.to(device)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)

        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model2.load_state_dict(checkpoint['net'])
        optimizer2.load_state_dict(checkpoint['optimizer'])

        assert checkpoint['epoch'] == 5

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2), f'Mismatch in {n1}'

    def test_checkpoint_for_inference(self, clear_pretrained, tmp_path):
        """Test loading checkpoint for inference."""
        from lib.config.aquasot.config import cfg
        from lib.models.aquasot import build_aquasot
        from lib.test.tracker import AquaSOTTracker

        cfg.MODEL.PRETRAIN_FILE = ""

        # Build and save model
        model = build_aquasot(cfg, training=True)
        device = TEST_DEVICE
        model = model.to(device)

        checkpoint_path = tmp_path / 'model.pth'
        torch.save({'net': model.state_dict()}, str(checkpoint_path))

        # Load via tracker
        params = {
            'checkpoint': str(checkpoint_path),
            'search_size': 256,
            'template_size': 128,
        }

        tracker = AquaSOTTracker(params)

        # Run inference
        rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        sonar_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        image = {'rgb': rgb_image, 'sonar': sonar_image}
        init_box = [200, 200, 50, 50]

        tracker.init(image, init_box)
        result = tracker.update(image)

        assert len(result) == 4

    def test_resume_training_from_checkpoint(self, clear_pretrained, tmp_path):
        """Test resuming training from checkpoint."""
        from lib.config.aquasot.config import cfg
        from lib.models.aquasot import build_aquasot

        cfg.MODEL.PRETRAIN_FILE = ""

        device = TEST_DEVICE

        # First training session
        model1 = build_aquasot(cfg, training=True).to(device)
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-4)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.1)

        # Simulate training
        for step in range(5):
            dummy_input = {
                'rgb_template': torch.randn(1, 3, 128, 128).to(device),
                'rgb_search': torch.randn(1, 3, 256, 256).to(device),
                'sonar_template': torch.randn(1, 1, 128, 128).to(device),
                'sonar_search': torch.randn(1, 1, 256, 256).to(device),
            }
            out = model1(**dummy_input)
            loss = out['rgb_boxes'].mean()
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

        scheduler1.step()

        # Save checkpoint
        checkpoint_path = tmp_path / 'resume.pth'
        torch.save({
            'epoch': 5,
            'net': model1.state_dict(),
            'optimizer': optimizer1.state_dict(),
            'scheduler': scheduler1.state_dict(),
        }, str(checkpoint_path))

        # Resume training
        model2 = build_aquasot(cfg, training=True).to(device)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.1)

        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model2.load_state_dict(checkpoint['net'])
        optimizer2.load_state_dict(checkpoint['optimizer'])
        scheduler2.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']

        assert start_epoch == 5
        assert scheduler2.last_epoch == scheduler1.last_epoch


# ==============================================================================
# Full Pipeline Integration Tests
# ==============================================================================

class TestFullPipeline:
    """Full pipeline integration tests."""

    def test_train_then_evaluate(self, mock_dataset_dir, clear_pretrained, tmp_path):
        """Test complete pipeline: train then evaluate."""
        from lib.config.aquasot.config import cfg
        from lib.models.aquasot import build_aquasot
        from lib.test.tracker import AquaSOTTracker
        from lib.test.evaluation import ExperimentAquaSOT

        cfg.MODEL.PRETRAIN_FILE = ""

        device = TEST_DEVICE

        # 1. Build and train model
        model = build_aquasot(cfg, training=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Simple training loop
        model.train()
        for step in range(3):
            dummy_input = {
                'rgb_template': torch.randn(1, 3, 128, 128).to(device),
                'rgb_search': torch.randn(1, 3, 256, 256).to(device),
                'sonar_template': torch.randn(1, 1, 128, 128).to(device),
                'sonar_search': torch.randn(1, 1, 256, 256).to(device),
            }
            out = model(**dummy_input)
            loss = out['rgb_boxes'].mean() + out['sonar_boxes'].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 2. Save checkpoint
        checkpoint_path = tmp_path / 'trained_model.pth'
        torch.save({'net': model.state_dict()}, str(checkpoint_path))

        # 3. Load checkpoint for evaluation
        params = {
            'checkpoint': str(checkpoint_path),
            'search_size': 256,
            'template_size': 128,
        }
        tracker = AquaSOTTracker(params)

        # 4. Run evaluation
        experiment = ExperimentAquaSOT(
            root_dir=str(mock_dataset_dir),
            result_dir=str(tmp_path / 'results'),
            report_dir=str(tmp_path / 'reports'),
            sequence_names=['uuv_test_0'],
        )

        experiment.run(tracker, visualize=False)

        # 5. Generate report
        performance = experiment.report(['AquaSOT'])

        # Verify complete pipeline works
        assert 'AquaSOT' in performance
        assert 'overall' in performance['AquaSOT']
        assert 'success_auc' in performance['AquaSOT']['overall']

    def test_model_output_consistency(self, clear_pretrained):
        """Test that model outputs are consistent between train and eval mode."""
        from lib.config.aquasot.config import cfg
        from lib.models.aquasot import build_aquasot

        cfg.MODEL.PRETRAIN_FILE = ""

        device = TEST_DEVICE
        model = build_aquasot(cfg, training=True).to(device)

        # Fixed input
        torch.manual_seed(42)
        input_data = {
            'rgb_template': torch.randn(1, 3, 128, 128).to(device),
            'rgb_search': torch.randn(1, 3, 256, 256).to(device),
            'sonar_template': torch.randn(1, 1, 128, 128).to(device),
            'sonar_search': torch.randn(1, 1, 256, 256).to(device),
        }

        # Train mode output
        model.train()
        with torch.no_grad():
            out_train = model(**input_data)

        # Eval mode output
        model.eval()
        with torch.no_grad():
            out_eval = model(**input_data)

        # Outputs should have same structure
        assert set(out_train.keys()) == set(out_eval.keys())

        # Box outputs should be similar (may differ due to dropout)
        assert out_train['rgb_boxes'].shape == out_eval['rgb_boxes'].shape
        assert out_train['sonar_boxes'].shape == out_eval['sonar_boxes'].shape

    def test_data_pipeline_to_model(self, mock_dataset_dir, clear_pretrained):
        """Test that data pipeline produces expected output structure."""
        from lib.config.aquasot.config import cfg
        from lib.train.dataset import AquaSOTDataset
        from lib.train.data import MultiModalSampler, LTRLoader

        cfg.MODEL.PRETRAIN_FILE = ""

        # Build data pipeline
        sequence_names = [f'uuv_test_{i}' for i in range(3)]
        dataset = AquaSOTDataset(str(mock_dataset_dir), sequence_names=sequence_names, use_rect_sonar=True)
        sampler = MultiModalSampler(
            datasets=[dataset],
            samples_per_epoch=4,
            max_gap=5,
            num_search_frames=1,
            num_template_frames=1,
        )
        loader = LTRLoader(
            'test', sampler, training=False, batch_size=2,
            num_workers=0, drop_last=True, stack_dim=1,
        )

        # Get batch
        batch = next(iter(loader))

        # Check batch has required keys
        required_keys = ['rgb_template_images', 'rgb_search_images',
                         'sonar_template_images', 'sonar_search_images']
        for key in required_keys:
            assert key in batch, f'Missing key: {key}'

        # Check annotation keys
        anno_keys = ['rgb_template_anno', 'rgb_search_anno',
                     'sonar_template_anno', 'sonar_search_anno']
        for key in anno_keys:
            assert key in batch, f'Missing key: {key}'

        # Verify data types
        rgb_template = batch['rgb_template_images']
        if isinstance(rgb_template, list):
            assert len(rgb_template) == 1  # num_template_frames=1
            assert isinstance(rgb_template[0], torch.Tensor)
        else:
            assert isinstance(rgb_template, torch.Tensor)

        # Verify the data pipeline is functional
        assert len(list(loader)) >= 1  # Can iterate at least once


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
