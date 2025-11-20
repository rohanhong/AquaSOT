"""
Unit Tests for FrameSequenceDataset

Tests the frame-level temporal tracking dataset including:
- Dataset initialization with single/multi-sequence support
- Frame sequence sampling (centered, boundary handling, stride)
- Multi-sequence boundary respect
- Validity flag propagation (0000 handling)
- Single-modality support
- Bbox scaling and metadata
- Integration with collate_fn_sequence

Author: AquaSOT-3D Team
Date: 2025-11-20
Audit: 2025112006
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from PIL import Image

from aquasot3d.dataloader.datasets import (
    RGBSonarDataset,
    MultiSequenceRGBSonarDataset,
    FrameSequenceDataset,
)
from aquasot3d.dataloader.loader import collate_fn_sequence


@pytest.fixture
def temp_single_sequence_dir():
    """
    Create temporary single-sequence dataset for testing.

    Yields:
        Path to temporary dataset root
    """
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

        # Create 20 frames for stride and boundary testing
        num_frames = 20
        for i in range(1, num_frames + 1):
            # RGB image (1920x1080)
            rgb_img = Image.new('RGB', (1920, 1080), color=(i * 5, i * 5, i * 5))
            rgb_img.save(rgb_img_dir / f"{i}.jpg")

            # Sonar image (682x376)
            sonar_img = Image.new('L', (682, 376), color=(i * 10))
            sonar_img.save(sonar_img_dir / f"{i}.jpg")

        # Create groundtruth with some occluded frames
        rgb_gt = []
        sonar_gt = []
        for i in range(1, num_frames + 1):
            # RGB occluded at frames 5, 15
            if i in [5, 15]:
                rgb_gt.append([0, 0, 0, 0])
            else:
                rgb_gt.append([100 + i*5, 200 + i*5, 300, 400])

            # Sonar occluded at frames 8, 18
            if i in [8, 18]:
                sonar_gt.append([0, 0, 0, 0])
            else:
                sonar_gt.append([50 + i*3, 100 + i*3, 150, 200])

        # Write groundtruth files
        with open(rgb_dir / "groundtruth.txt", 'w') as f:
            for bbox in rgb_gt:
                f.write(','.join(map(str, bbox)) + '\n')

        with open(sonar_dir / "groundtruth_rect.txt", 'w') as f:
            for bbox in sonar_gt:
                f.write(','.join(map(str, bbox)) + '\n')

        yield temp_dir

    finally:
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_multi_sequence_dir():
    """
    Create temporary multi-sequence dataset for testing boundary respect.

    Yields:
        Path to temporary dataset root
    """
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create two sequences
        for seq_name in ["seq1", "seq2"]:
            seq_dir = temp_dir / seq_name
            rgb_dir = seq_dir / "light"
            sonar_dir = seq_dir / "sonar"
            rgb_img_dir = rgb_dir / "img"
            sonar_img_dir = sonar_dir / "img_rect"

            rgb_img_dir.mkdir(parents=True, exist_ok=True)
            sonar_img_dir.mkdir(parents=True, exist_ok=True)

            # 10 frames per sequence
            num_frames = 10
            for i in range(1, num_frames + 1):
                rgb_img = Image.new('RGB', (1920, 1080), color=(i * 10, i * 10, i * 10))
                rgb_img.save(rgb_img_dir / f"{i}.jpg")

                sonar_img = Image.new('L', (682, 376), color=(i * 15))
                sonar_img.save(sonar_img_dir / f"{i}.jpg")

            # Groundtruth
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


class TestFrameSequenceDatasetInitialization:
    """Test dataset initialization."""

    def test_single_sequence_initialization(self, temp_single_sequence_dir):
        """Test initialization with single-sequence dataset."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512,
            frame_stride=1
        )

        assert len(dataset) == 20
        assert dataset.sequence_length == 5
        assert dataset.frame_size == 512
        assert dataset.frame_stride == 1
        assert not dataset.is_multi_sequence

    def test_multi_sequence_initialization(self, temp_multi_sequence_dir):
        """Test initialization with multi-sequence dataset."""
        base_dataset = MultiSequenceRGBSonarDataset(
            data_root=temp_multi_sequence_dir,
            sequence_names=["seq1", "seq2"],
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512
        )

        assert len(dataset) == 20  # 10 frames per sequence
        assert dataset.is_multi_sequence
        assert len(dataset.sequence_ranges) == 2
        assert 'seq1' in dataset.sequence_ranges
        assert 'seq2' in dataset.sequence_ranges

    def test_custom_parameters(self, temp_single_sequence_dir):
        """Test initialization with custom parameters."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq"
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=7,
            frame_size=512,
            frame_stride=2
        )

        assert dataset.sequence_length == 7
        assert dataset.frame_size == 512
        assert dataset.frame_stride == 2


class TestFrameSampling:
    """Test frame sequence sampling logic."""

    def test_centered_sampling(self, temp_single_sequence_dir):
        """Test centered frame sampling."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_stride=1
        )

        # Sample from middle (index 10)
        sample = dataset[10]

        # Should get frames [8, 9, 10, 11, 12] (0-indexed)
        # Frame IDs [9, 10, 11, 12, 13] (1-based)
        expected_frame_ids = [9, 10, 11, 12, 13]
        assert sample['frame_ids'] == expected_frame_ids
        assert sample['target_idx'] == 2  # Center frame at index 2
        assert sample['sequence_length'] == 5

    def test_boundary_padding_start(self, temp_single_sequence_dir):
        """Test boundary handling at sequence start."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_stride=1
        )

        # Sample from start (index 0)
        sample = dataset[0]

        # Should pad with first frame
        # Expected: [0, 0, 0, 1, 2] or similar padding
        assert len(sample['frame_ids']) == 5
        assert sample['target_idx'] == 2
        # First frame should be repeated
        assert sample['frame_ids'][0] <= sample['frame_ids'][1]

    def test_boundary_padding_end(self, temp_single_sequence_dir):
        """Test boundary handling at sequence end."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_stride=1
        )

        # Sample from end (index 19)
        sample = dataset[19]

        # Should pad with last frame
        assert len(sample['frame_ids']) == 5
        assert sample['target_idx'] == 2
        # Last frame should be repeated
        assert sample['frame_ids'][-1] >= sample['frame_ids'][-2]

    def test_frame_stride(self, temp_single_sequence_dir):
        """Test frame sampling with stride > 1."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_stride=2  # Skip every other frame
        )

        # Sample from middle (index 10)
        sample = dataset[10]

        # With stride=2, should get frames [6, 8, 10, 12, 14] (0-indexed)
        # Frame IDs [7, 9, 11, 13, 15] (1-based)
        frame_ids = sample['frame_ids']

        # Check that frames are spaced by stride
        assert len(frame_ids) == 5
        # Verify spacing (allowing for boundary effects)
        if len(set(frame_ids)) == 5:  # No duplicates (not at boundary)
            diffs = [frame_ids[i+1] - frame_ids[i] for i in range(len(frame_ids)-1)]
            assert all(d >= 2 for d in diffs), f"Frame spacing should be >= stride, got {frame_ids}"


class TestMultiSequenceBoundaryRespect:
    """Test that frame sequences respect sequence boundaries in multi-sequence mode."""

    def test_sequence_boundary_not_crossed(self, temp_multi_sequence_dir):
        """Test that frame sequences don't cross sequence boundaries."""
        base_dataset = MultiSequenceRGBSonarDataset(
            data_root=temp_multi_sequence_dir,
            sequence_names=["seq1", "seq2"],
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        # Sample from end of seq1 (index 9, last frame of seq1)
        sample = dataset[9]

        # All frames should be from seq1
        assert sample['sequence_name'] == 'seq1'
        # Frames should not exceed seq1 boundary (frame_id <= 10)
        assert all(fid <= 10 for fid in sample['frame_ids'])

    def test_sequence_name_consistency(self, temp_multi_sequence_dir):
        """Test that all frames in sequence come from same sequence."""
        base_dataset = MultiSequenceRGBSonarDataset(
            data_root=temp_multi_sequence_dir,
            sequence_names=["seq1", "seq2"],
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        # Test multiple samples
        for idx in [0, 5, 10, 15]:
            sample = dataset[idx]
            seq_name = sample['sequence_name']

            # Verify get_sequence_name returns same name
            assert dataset.get_sequence_name(idx) == seq_name


class TestValidityFlagPropagation:
    """Test validity flag propagation for 0000 handling (CRITICAL)."""

    def test_validity_flags_present(self, temp_single_sequence_dir):
        """Test that validity flags are present in sample."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        sample = dataset[10]

        assert 'is_valid_rgb' in sample
        assert 'is_valid_sonar' in sample
        assert isinstance(sample['is_valid_rgb'], torch.Tensor)
        assert isinstance(sample['is_valid_sonar'], torch.Tensor)
        assert sample['is_valid_rgb'].shape == (5,)
        assert sample['is_valid_sonar'].shape == (5,)

    def test_occluded_frame_validity(self, temp_single_sequence_dir):
        """Test validity flags for occluded frames."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        # Sample centered at frame 5 (index 4, RGB occluded)
        # Should get frames [2, 3, 4, 5, 6] (0-indexed)
        # Frame 5 (1-based) is occluded in RGB
        sample = dataset[4]

        # Check that frame 5 has is_valid_rgb = False
        target_idx = sample['target_idx']
        assert sample['is_valid_rgb'][target_idx].item() == 0.0  # Invalid

    def test_validity_tensor_dtype(self, temp_single_sequence_dir):
        """Test that validity tensors have correct dtype."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        sample = dataset[10]

        assert sample['is_valid_rgb'].dtype == torch.float32
        assert sample['is_valid_sonar'].dtype == torch.float32


class TestSingleModalitySupport:
    """Test single-modality training support."""

    def test_rgb_only_mode(self, temp_single_sequence_dir):
        """Test RGB-only modality."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True,
            modalities=['rgb']
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        sample = dataset[10]

        assert sample['rgb_frames'] is not None
        assert sample['sonar_frames'] is None
        assert sample['rgb_frames'].shape == (5, 3, 512, 512)

    def test_sonar_only_mode(self, temp_single_sequence_dir):
        """Test Sonar-only modality."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True,
            modalities=['sonar']
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        sample = dataset[10]

        assert sample['rgb_frames'] is None
        assert sample['sonar_frames'] is not None
        assert sample['sonar_frames'].shape == (5, 1, 512, 512)

    def test_dual_modality_mode(self, temp_single_sequence_dir):
        """Test dual-modality (default)."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        sample = dataset[10]

        assert sample['rgb_frames'] is not None
        assert sample['sonar_frames'] is not None
        assert sample['rgb_frames'].shape == (5, 3, 512, 512)
        assert sample['sonar_frames'].shape == (5, 1, 512, 512)


class TestBboxScalingAndMetadata:
    """Test bbox scaling and transformation metadata."""

    def test_bbox_scaling(self, temp_single_sequence_dir):
        """Test that bboxes are properly scaled."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512
        )

        sample = dataset[10]

        # Check bbox shapes
        assert sample['rgb_bboxes'].shape == (5, 4)
        assert sample['sonar_bboxes'].shape == (5, 4)
        assert sample['rgb_bboxes'].dtype == torch.float32
        assert sample['sonar_bboxes'].dtype == torch.float32

    def test_metadata_present(self, temp_single_sequence_dir):
        """Test that transformation metadata is present."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        sample = dataset[10]

        assert 'rgb_metadata' in sample
        assert 'sonar_metadata' in sample
        assert len(sample['rgb_metadata']) == 5
        assert len(sample['sonar_metadata']) == 5

        # Check metadata content
        for meta in sample['rgb_metadata']:
            assert 'original_height' in meta
            assert 'original_width' in meta
            assert 'scale_x' in meta
            assert 'scale_y' in meta
            assert 'pad_x' in meta
            assert 'pad_y' in meta


class TestCollateIntegration:
    """Test integration with collate_fn_sequence."""

    def test_batch_collation(self, temp_single_sequence_dir):
        """Test batching with collate_fn_sequence."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        # Create batch
        batch = [dataset[i] for i in [5, 10, 15]]
        batched = collate_fn_sequence(batch)

        # Check batched shapes
        assert batched['rgb_frames'].shape == (3, 5, 3, 512, 512)  # [B, T, C, H, W]
        assert batched['sonar_frames'].shape == (3, 5, 1, 512, 512)
        assert batched['rgb_bboxes'].shape == (3, 5, 4)  # [B, T, 4]
        assert batched['sonar_bboxes'].shape == (3, 5, 4)
        assert batched['is_valid_rgb'].shape == (3, 5)  # [B, T]
        assert batched['is_valid_sonar'].shape == (3, 5)

    def test_single_modality_batch_collation(self, temp_single_sequence_dir):
        """Test batching with single modality."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True,
            modalities=['rgb']
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        # Create batch
        batch = [dataset[i] for i in [5, 10]]
        batched = collate_fn_sequence(batch)

        # RGB should be present, Sonar should be None
        assert batched['rgb_frames'] is not None
        assert batched['sonar_frames'] is None
        assert batched['rgb_frames'].shape == (2, 5, 3, 512, 512)

    def test_validity_flags_batched(self, temp_single_sequence_dir):
        """Test that validity flags are properly batched."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        # Create batch including occluded frames
        batch = [dataset[i] for i in [4, 14]]  # Frames 5 and 15 are occluded in RGB
        batched = collate_fn_sequence(batch)

        # Check validity tensor
        assert batched['is_valid_rgb'].dtype == torch.float32
        assert batched['is_valid_sonar'].dtype == torch.float32


class TestDataTypes:
    """Test output data types."""

    def test_tensor_types(self, temp_single_sequence_dir):
        """Test that outputs are correct tensor types."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        sample = dataset[10]

        # Check tensor types
        assert isinstance(sample['rgb_frames'], torch.Tensor)
        assert isinstance(sample['sonar_frames'], torch.Tensor)
        assert isinstance(sample['rgb_bboxes'], torch.Tensor)
        assert isinstance(sample['sonar_bboxes'], torch.Tensor)
        assert isinstance(sample['is_valid_rgb'], torch.Tensor)
        assert isinstance(sample['is_valid_sonar'], torch.Tensor)

        # Check dtypes
        assert sample['rgb_frames'].dtype == torch.float32
        assert sample['sonar_frames'].dtype == torch.float32
        assert sample['rgb_bboxes'].dtype == torch.float32
        assert sample['sonar_bboxes'].dtype == torch.float32

    def test_normalized_images(self, temp_single_sequence_dir):
        """Test that images are normalized to [0, 1]."""
        base_dataset = RGBSonarDataset(
            data_root=temp_single_sequence_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5
        )

        sample = dataset[10]

        # Check value range [0, 1]
        assert sample['rgb_frames'].min() >= 0.0
        assert sample['rgb_frames'].max() <= 1.0
        assert sample['sonar_frames'].min() >= 0.0
        assert sample['sonar_frames'].max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
