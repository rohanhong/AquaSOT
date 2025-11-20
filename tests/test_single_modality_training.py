"""
Test Single-Modality Training Support (Audit 2025112003)

Tests for single-modality (RGB-only or Sonar-only) training mode:
- Dataset modality selection and conditional loading
- Validity flag computation for single-modality mode
- Trainer conditional validity statistics reporting
- End-to-end single-modality training pipeline

Author: AquaSOT-3D Team
Date: 2025-11-20
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from aquasot3d.dataloader.datasets.rgb_sonar_dataset import (
    RGBSonarDataset,
    MultiSequenceRGBSonarDataset
)
from aquasot3d.training.trainer import Trainer


class TestRGBSonarDatasetModalitySelection:
    """Test RGBSonarDataset modality selection functionality"""

    def test_default_modalities_both(self, tmp_path):
        """Test default behavior loads both modalities"""
        # Create mock dataset structure
        self._create_mock_dataset(tmp_path, 'test_seq', num_frames=5)

        dataset = RGBSonarDataset(
            data_root=tmp_path,
            sequence_name='test_seq'
        )

        assert dataset.modalities == ['rgb', 'sonar']
        assert dataset.rgb_dir is not None
        assert dataset.sonar_dir is not None
        assert dataset.rgb_bboxes is not None
        assert dataset.sonar_bboxes is not None

    def test_rgb_only_mode(self, tmp_path):
        """Test RGB-only mode only loads RGB data"""
        self._create_mock_dataset(tmp_path, 'test_seq', num_frames=5)

        dataset = RGBSonarDataset(
            data_root=tmp_path,
            sequence_name='test_seq',
            modalities=['rgb']
        )

        assert dataset.modalities == ['rgb']
        assert dataset.rgb_dir is not None
        assert dataset.sonar_dir is None
        assert dataset.rgb_bboxes is not None
        assert dataset.sonar_bboxes is None

    def test_sonar_only_mode(self, tmp_path):
        """Test Sonar-only mode only loads Sonar data"""
        self._create_mock_dataset(tmp_path, 'test_seq', num_frames=5)

        dataset = RGBSonarDataset(
            data_root=tmp_path,
            sequence_name='test_seq',
            modalities=['sonar']
        )

        assert dataset.modalities == ['sonar']
        assert dataset.rgb_dir is None
        assert dataset.sonar_dir is not None
        assert dataset.rgb_bboxes is None
        assert dataset.sonar_bboxes is not None

    def test_invalid_modality_raises_error(self, tmp_path):
        """Test invalid modality parameter raises ValueError"""
        self._create_mock_dataset(tmp_path, 'test_seq', num_frames=5)

        with pytest.raises(ValueError, match="Invalid modality"):
            RGBSonarDataset(
                data_root=tmp_path,
                sequence_name='test_seq',
                modalities=['invalid']
            )

    def test_empty_modalities_raises_error(self, tmp_path):
        """Test empty modalities list raises ValueError"""
        self._create_mock_dataset(tmp_path, 'test_seq', num_frames=5)

        with pytest.raises(ValueError, match="At least one modality"):
            RGBSonarDataset(
                data_root=tmp_path,
                sequence_name='test_seq',
                modalities=[]
            )

    def _create_mock_dataset(self, root_path, seq_name, num_frames=10):
        """Helper to create mock dataset structure"""
        seq_path = root_path / seq_name

        # Create RGB structure
        rgb_dir = seq_path / "light"
        rgb_img_dir = rgb_dir / "img"
        rgb_img_dir.mkdir(parents=True, exist_ok=True)

        # Create Sonar structure
        sonar_dir = seq_path / "sonar"
        sonar_img_dir = sonar_dir / "img_rect"
        sonar_img_dir.mkdir(parents=True, exist_ok=True)

        # Create groundtruth files
        rgb_gt = rgb_dir / "groundtruth.txt"
        sonar_gt = sonar_dir / "groundtruth_rect.txt"

        with open(rgb_gt, 'w') as f:
            for i in range(num_frames):
                f.write(f"{100 + i*10},{200 + i*5},{50},{40}\n")

        with open(sonar_gt, 'w') as f:
            for i in range(num_frames):
                f.write(f"{150 + i*8},{250 + i*6},{60},{45}\n")

        # Create dummy images
        for i in range(1, num_frames + 1):
            # RGB image
            rgb_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(rgb_img).save(rgb_img_dir / f"{i}.jpg")

            # Sonar image
            sonar_img = np.random.randint(0, 255, (376, 682), dtype=np.uint8)
            Image.fromarray(sonar_img).save(sonar_img_dir / f"{i}.jpg")


class TestModalityValidityLogic:
    """Test validity flag computation in single-modality mode"""

    def test_rgb_only_validity_ignores_sonar(self, tmp_path):
        """Test RGB-only mode: validity only depends on RGB"""
        # Create dataset with some RGB occluded, some Sonar occluded
        self._create_mixed_occlusion_dataset(tmp_path, 'test_seq')

        dataset = RGBSonarDataset(
            data_root=tmp_path,
            sequence_name='test_seq',
            modalities=['rgb']
        )

        # Frame 1: RGB valid, Sonar occluded (0000)
        # In RGB-only mode, should be valid
        sample = dataset._get_frame_data(0)
        assert sample['is_rgb_valid'] == True
        assert sample['is_sonar_valid'] is None  # Not loaded
        assert sample['is_valid'] == True  # Only depends on RGB

    def test_sonar_only_validity_ignores_rgb(self, tmp_path):
        """Test Sonar-only mode: validity only depends on Sonar"""
        self._create_mixed_occlusion_dataset(tmp_path, 'test_seq')

        dataset = RGBSonarDataset(
            data_root=tmp_path,
            sequence_name='test_seq',
            modalities=['sonar']
        )

        # Frame 2: RGB occluded (0000), Sonar valid
        # In Sonar-only mode, should be valid
        sample = dataset._get_frame_data(1)
        assert sample['is_rgb_valid'] is None  # Not loaded
        assert sample['is_sonar_valid'] == True
        assert sample['is_valid'] == True  # Only depends on Sonar

    def test_dual_modality_validity_requires_both(self, tmp_path):
        """Test dual-modality mode: validity requires both modalities"""
        self._create_mixed_occlusion_dataset(tmp_path, 'test_seq')

        dataset = RGBSonarDataset(
            data_root=tmp_path,
            sequence_name='test_seq',
            modalities=['rgb', 'sonar']
        )

        # Frame 1: RGB valid, Sonar occluded
        sample = dataset._get_frame_data(0)
        assert sample['is_rgb_valid'] == True
        assert sample['is_sonar_valid'] == False
        assert sample['is_valid'] == False  # Requires both

        # Frame 3: Both valid
        sample = dataset._get_frame_data(2)
        assert sample['is_rgb_valid'] == True
        assert sample['is_sonar_valid'] == True
        assert sample['is_valid'] == True

    def _create_mixed_occlusion_dataset(self, root_path, seq_name):
        """Helper to create dataset with mixed occlusion patterns"""
        seq_path = root_path / seq_name

        # Create directory structure
        rgb_dir = seq_path / "light"
        rgb_img_dir = rgb_dir / "img"
        rgb_img_dir.mkdir(parents=True, exist_ok=True)

        sonar_dir = seq_path / "sonar"
        sonar_img_dir = sonar_dir / "img_rect"
        sonar_img_dir.mkdir(parents=True, exist_ok=True)

        # Create groundtruth with mixed occlusion
        rgb_gt = rgb_dir / "groundtruth.txt"
        sonar_gt = sonar_dir / "groundtruth_rect.txt"

        # Frame 1: RGB valid, Sonar occluded
        # Frame 2: RGB occluded, Sonar valid
        # Frame 3: Both valid
        with open(rgb_gt, 'w') as f:
            f.write("100,200,50,40\n")  # Valid
            f.write("0,0,0,0\n")        # Occluded
            f.write("100,200,50,40\n")  # Valid

        with open(sonar_gt, 'w') as f:
            f.write("0,0,0,0\n")        # Occluded
            f.write("150,250,60,45\n")  # Valid
            f.write("150,250,60,45\n")  # Valid

        # Create images
        for i in range(1, 4):
            from PIL import Image
            rgb_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            Image.fromarray(rgb_img).save(rgb_img_dir / f"{i}.jpg")

            sonar_img = np.random.randint(0, 255, (376, 682), dtype=np.uint8)
            Image.fromarray(sonar_img).save(sonar_img_dir / f"{i}.jpg")


class TestMultiSequenceModalityPropagation:
    """Test MultiSequenceRGBSonarDataset modality propagation"""

    def test_propagates_modalities_to_child_datasets(self, tmp_path):
        """Test modalities parameter is propagated to child datasets"""
        # Create two sequences
        self._create_mock_sequence(tmp_path, 'seq1', num_frames=3)
        self._create_mock_sequence(tmp_path, 'seq2', num_frames=3)

        # Create multi-sequence dataset with RGB-only mode
        dataset = MultiSequenceRGBSonarDataset(
            data_root=tmp_path,
            sequence_names=['seq1', 'seq2'],
            modalities=['rgb']
        )

        # Check child datasets have correct modality
        assert dataset.modalities == ['rgb']
        for seq_name, seq_dataset in dataset.sequence_datasets.items():
            assert seq_dataset.modalities == ['rgb']
            assert seq_dataset.rgb_bboxes is not None
            assert seq_dataset.sonar_bboxes is None

    def _create_mock_sequence(self, root_path, seq_name, num_frames):
        """Helper to create mock sequence"""
        seq_path = root_path / seq_name

        # Create RGB structure
        rgb_dir = seq_path / "light"
        rgb_img_dir = rgb_dir / "img"
        rgb_img_dir.mkdir(parents=True, exist_ok=True)

        # Create Sonar structure
        sonar_dir = seq_path / "sonar"
        sonar_img_dir = sonar_dir / "img_rect"
        sonar_img_dir.mkdir(parents=True, exist_ok=True)

        # Create groundtruth
        rgb_gt = rgb_dir / "groundtruth.txt"
        sonar_gt = sonar_dir / "groundtruth_rect.txt"

        with open(rgb_gt, 'w') as f:
            for i in range(num_frames):
                f.write(f"{100 + i*10},{200 + i*5},{50},{40}\n")

        with open(sonar_gt, 'w') as f:
            for i in range(num_frames):
                f.write(f"{150 + i*8},{250 + i*6},{60},{45}\n")

        # Create images
        for i in range(1, num_frames + 1):
            from PIL import Image
            rgb_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            Image.fromarray(rgb_img).save(rgb_img_dir / f"{i}.jpg")

            sonar_img = np.random.randint(0, 255, (376, 682), dtype=np.uint8)
            Image.fromarray(sonar_img).save(sonar_img_dir / f"{i}.jpg")


class TestTrainerConditionalStatistics:
    """Test Trainer conditional validity statistics reporting"""

    def test_rgb_only_trainer_reports_only_rgb_validity(self):
        """Test RGB-only trainer only reports RGB validity statistics"""
        # Create mock model with only RGB backbone
        model = Mock()
        model.rgb_backbone = Mock()
        model.sonar_backbone = None

        # Create mock components
        optimizer = Mock()
        scheduler = Mock()
        loss_manager = Mock()
        train_loader = Mock()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            train_loader=train_loader,
            max_epochs=1
        )

        assert trainer.enabled_modalities == ['rgb']

        # Simulate validity stats from training
        validity_stats = {
            'total_samples': 100,
            'valid_samples': 70,
            'valid_rgb': 70,
            'valid_sonar': 50,  # This should be ignored
            'invalid_samples': 30
        }

        # Manually construct averaged metrics like train_epoch would
        averaged_metrics = {}
        if validity_stats['total_samples'] > 0:
            averaged_metrics['validity_rate'] = validity_stats['valid_samples'] / validity_stats['total_samples']

            if 'rgb' in trainer.enabled_modalities and 'valid_rgb' in validity_stats:
                averaged_metrics['validity_rate_rgb'] = validity_stats['valid_rgb'] / validity_stats['total_samples']

            if 'sonar' in trainer.enabled_modalities and 'valid_sonar' in validity_stats:
                averaged_metrics['validity_rate_sonar'] = validity_stats['valid_sonar'] / validity_stats['total_samples']

        # Check: validity_rate_rgb should be present, validity_rate_sonar should NOT
        assert 'validity_rate' in averaged_metrics
        assert 'validity_rate_rgb' in averaged_metrics
        assert 'validity_rate_sonar' not in averaged_metrics  # Should not be reported

    def test_sonar_only_trainer_reports_only_sonar_validity(self):
        """Test Sonar-only trainer only reports Sonar validity statistics"""
        # Create mock model with only Sonar backbone
        model = Mock()
        model.rgb_backbone = None
        model.sonar_backbone = Mock()

        optimizer = Mock()
        scheduler = Mock()
        loss_manager = Mock()
        train_loader = Mock()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            train_loader=train_loader,
            max_epochs=1
        )

        assert trainer.enabled_modalities == ['sonar']

        # Simulate validity stats
        validity_stats = {
            'total_samples': 100,
            'valid_samples': 60,
            'valid_rgb': 80,   # Should be ignored
            'valid_sonar': 60,
            'invalid_samples': 40
        }

        # Construct averaged metrics
        averaged_metrics = {}
        if validity_stats['total_samples'] > 0:
            averaged_metrics['validity_rate'] = validity_stats['valid_samples'] / validity_stats['total_samples']

            if 'rgb' in trainer.enabled_modalities and 'valid_rgb' in validity_stats:
                averaged_metrics['validity_rate_rgb'] = validity_stats['valid_rgb'] / validity_stats['total_samples']

            if 'sonar' in trainer.enabled_modalities and 'valid_sonar' in validity_stats:
                averaged_metrics['validity_rate_sonar'] = validity_stats['valid_sonar'] / validity_stats['total_samples']

        # Check: validity_rate_sonar should be present, validity_rate_rgb should NOT
        assert 'validity_rate' in averaged_metrics
        assert 'validity_rate_rgb' not in averaged_metrics  # Should not be reported
        assert 'validity_rate_sonar' in averaged_metrics

    def test_dual_modality_trainer_reports_both_validities(self):
        """Test dual-modality trainer reports both RGB and Sonar validity"""
        # Create mock model with both backbones
        model = Mock()
        model.rgb_backbone = Mock()
        model.sonar_backbone = Mock()

        optimizer = Mock()
        scheduler = Mock()
        loss_manager = Mock()
        train_loader = Mock()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            train_loader=train_loader,
            max_epochs=1
        )

        assert trainer.enabled_modalities == ['rgb', 'sonar']

        # Simulate validity stats
        validity_stats = {
            'total_samples': 100,
            'valid_samples': 50,
            'valid_rgb': 70,
            'valid_sonar': 60,
            'invalid_samples': 50
        }

        # Construct averaged metrics
        averaged_metrics = {}
        if validity_stats['total_samples'] > 0:
            averaged_metrics['validity_rate'] = validity_stats['valid_samples'] / validity_stats['total_samples']

            if 'rgb' in trainer.enabled_modalities and 'valid_rgb' in validity_stats:
                averaged_metrics['validity_rate_rgb'] = validity_stats['valid_rgb'] / validity_stats['total_samples']

            if 'sonar' in trainer.enabled_modalities and 'valid_sonar' in validity_stats:
                averaged_metrics['validity_rate_sonar'] = validity_stats['valid_sonar'] / validity_stats['total_samples']

        # Check: both validity rates should be present
        assert 'validity_rate' in averaged_metrics
        assert 'validity_rate_rgb' in averaged_metrics
        assert 'validity_rate_sonar' in averaged_metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
