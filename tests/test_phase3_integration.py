"""
Integration Tests for Phase 3: Training Script Adaptation

Tests the complete data flow from FrameSequenceDataset through dataloader
to verify Phase 1-3 integration.

Author: AquaSOT-3D Team
Date: 2025-11-20
Audit: 2025112006
"""

import pytest
import torch
from torch.utils.data import DataLoader
import tempfile
import shutil
from pathlib import Path
from PIL import Image

from aquasot3d.dataloader.datasets import RGBSonarDataset, FrameSequenceDataset
from aquasot3d.dataloader.loader import collate_fn_sequence


@pytest.fixture
def temp_sequence_dataset():
    """Create temporary sequence dataset for testing."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create directory structure
        seq_dir = temp_dir / "test_seq"
        rgb_dir = seq_dir / "light"
        sonar_dir = seq_dir / "sonar"
        rgb_img_dir = rgb_dir / "img"
        sonar_img_dir = sonar_dir / "img_rect"

        rgb_img_dir.mkdir(parents=True, exist_ok=True)
        sonar_img_dir.mkdir(parents=True, exist_ok=True)

        # Create 20 frames
        num_frames = 20
        for i in range(1, num_frames + 1):
            # RGB image
            rgb_img = Image.new('RGB', (1920, 1080), color=(i * 10, i * 10, i * 10))
            rgb_img.save(rgb_img_dir / f"{i}.jpg")

            # Sonar image
            sonar_img = Image.new('L', (682, 376), color=(i * 10))
            sonar_img.save(sonar_img_dir / f"{i}.jpg")

        # Create groundtruth
        rgb_gt = [[100 + i*5, 200 + i*5, 300, 400] for i in range(1, num_frames + 1)]
        sonar_gt = [[50 + i*3, 100 + i*3, 150, 200] for i in range(1, num_frames + 1)]

        with open(rgb_dir / "groundtruth.txt", 'w') as f:
            for bbox in rgb_gt:
                f.write(','.join(map(str, bbox)) + '\n')

        with open(sonar_dir / "groundtruth_rect.txt", 'w') as f:
            for bbox in sonar_gt:
                f.write(','.join(map(str, bbox)) + '\n')

        yield temp_dir

    finally:
        shutil.rmtree(temp_dir)


class TestDataLoaderIntegration:
    """Test DataLoader integration with FrameSequenceDataset."""

    def test_dataloader_basic(self, temp_sequence_dataset):
        """Test basic dataloader functionality."""
        # Create base dataset
        base_dataset = RGBSonarDataset(
            data_root=temp_sequence_dataset,
            sequence_name="test_seq",
            return_occluded=True
        )

        # Create frame sequence dataset
        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512,
            frame_stride=1
        )

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn_sequence
        )

        # Get first batch
        batch = next(iter(dataloader))

        # Verify batch structure
        assert 'rgb_frames' in batch
        assert 'sonar_frames' in batch
        assert 'rgb_bboxes' in batch
        assert 'sonar_bboxes' in batch
        assert 'is_valid_rgb' in batch
        assert 'is_valid_sonar' in batch

        # Verify batch shapes
        B, T = 2, 5
        assert batch['rgb_frames'].shape == (B, T, 3, 512, 512)
        assert batch['sonar_frames'].shape == (B, T, 1, 512, 512)
        assert batch['rgb_bboxes'].shape == (B, T, 4)
        assert batch['sonar_bboxes'].shape == (B, T, 4)
        assert batch['is_valid_rgb'].shape == (B, T)
        assert batch['is_valid_sonar'].shape == (B, T)

    def test_dataloader_metadata(self, temp_sequence_dataset):
        """Test that metadata is properly collated."""
        base_dataset = RGBSonarDataset(
            data_root=temp_sequence_dataset,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn_sequence
        )

        batch = next(iter(dataloader))

        # Check metadata presence
        assert 'rgb_metadata' in batch
        assert 'sonar_metadata' in batch

        # Check metadata structure (List of List of Dict)
        assert isinstance(batch['rgb_metadata'], list)
        assert len(batch['rgb_metadata']) == 2  # Batch size

        # Each batch item should have T metadata dicts
        assert isinstance(batch['rgb_metadata'][0], list)
        assert len(batch['rgb_metadata'][0]) == 5  # Sequence length

        # Each metadata dict should have transformation info
        meta = batch['rgb_metadata'][0][0]
        assert 'original_height' in meta
        assert 'original_width' in meta
        assert 'scale_x' in meta
        assert 'scale_y' in meta

    def test_dataloader_single_modality(self, temp_sequence_dataset):
        """Test dataloader with single-modality dataset."""
        base_dataset = RGBSonarDataset(
            data_root=temp_sequence_dataset,
            sequence_name="test_seq",
            return_occluded=True,
            modalities=['rgb']  # RGB only
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn_sequence
        )

        batch = next(iter(dataloader))

        # RGB should be present, Sonar should be None
        assert batch['rgb_frames'] is not None
        assert batch['sonar_frames'] is None
        assert batch['rgb_frames'].shape == (2, 5, 3, 512, 512)

    def test_dataloader_validity_flags(self, temp_sequence_dataset):
        """Test that validity flags are properly batched."""
        base_dataset = RGBSonarDataset(
            data_root=temp_sequence_dataset,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn_sequence
        )

        batch = next(iter(dataloader))

        # Check validity flags
        assert batch['is_valid_rgb'].dtype == torch.float32
        assert batch['is_valid_sonar'].dtype == torch.float32
        assert batch['is_valid_rgb'].shape == (2, 5)
        assert batch['is_valid_sonar'].shape == (2, 5)

        # All flags should be 1.0 (valid) for this test dataset
        assert torch.all(batch['is_valid_rgb'] == 1.0)
        assert torch.all(batch['is_valid_sonar'] == 1.0)

    def test_dataloader_target_idx(self, temp_sequence_dataset):
        """Test that target_idx is properly batched."""
        base_dataset = RGBSonarDataset(
            data_root=temp_sequence_dataset,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn_sequence
        )

        batch = next(iter(dataloader))

        # Check target_idx
        assert 'target_idx' in batch
        assert batch['target_idx'].shape == (2,)
        # For sequence_length=5, target should be center frame (index 2)
        assert torch.all(batch['target_idx'] == 2)


class TestDataFlowConsistency:
    """Test data flow consistency through entire pipeline."""

    def test_bbox_coordinate_consistency(self, temp_sequence_dataset):
        """Test that bboxes are properly scaled through pipeline."""
        base_dataset = RGBSonarDataset(
            data_root=temp_sequence_dataset,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=3,
            frame_size=512
        )

        # Get single sample
        sample = dataset[10]

        # Check bbox scales are in expected range
        # Original RGB: 1920x1080, letterbox to: 512x512
        # Uniform scale: min(512/1920, 512/1080) = 0.333
        # Result: 512x360, pad_top=140, pad_bottom=140
        assert sample['rgb_bboxes'].min() >= 0
        assert sample['rgb_bboxes'].max() <= 512

        # Check metadata matches
        meta = sample['rgb_metadata'][0]
        assert meta['scale_x'] == pytest.approx(512 / 1920, rel=1e-5)  # Letterbox uniform scale
        assert meta['scale_y'] == pytest.approx(512 / 1920, rel=1e-5)  # Same as scale_x (uniform)

    def test_frame_ids_sequential(self, temp_sequence_dataset):
        """Test that frame IDs are sequential for consecutive frames."""
        base_dataset = RGBSonarDataset(
            data_root=temp_sequence_dataset,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512,
            frame_stride=1  # Consecutive frames
        )

        sample = dataset[10]

        # Check frame IDs are sequential
        frame_ids = sample['frame_ids']
        assert len(frame_ids) == 5

        # With stride=1, difference should be 1
        for i in range(len(frame_ids) - 1):
            assert frame_ids[i+1] - frame_ids[i] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
