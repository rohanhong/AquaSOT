"""
Unit Tests for RGBSonarDataset

Tests the multi-modal dataset loader including:
- Dataset initialization and validation
- Frame loading and indexing
- Groundtruth parsing and occlusion handling
- Sequence metadata retrieval
- Last-valid-bbox finding

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from PIL import Image

from aquasot3d.dataloader.datasets.rgb_sonar_dataset import (
    RGBSonarDataset,
    is_occluded,
    parse_groundtruth,
)


@pytest.fixture
def temp_dataset_dir():
    """
    Create temporary dataset directory structure for testing.

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
        sonar_img_dir = sonar_dir / "img_rect"  # Use rectangular coordinates (post-preprocessing)

        rgb_img_dir.mkdir(parents=True, exist_ok=True)
        sonar_img_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy images (10 frames)
        num_frames = 10
        for i in range(1, num_frames + 1):
            # RGB image (1920x1080)
            rgb_img = Image.new('RGB', (1920, 1080), color=(i * 10, i * 10, i * 10))
            rgb_img.save(rgb_img_dir / f"{i}.jpg")

            # Sonar image (682x376, grayscale for rectangular coordinates)
            sonar_img = Image.new('L', (682, 376), color=(i * 20))
            sonar_img.save(sonar_img_dir / f"{i}.jpg")

        # Create groundtruth files
        rgb_gt = [
            [100, 200, 300, 400],   # Frame 1
            [110, 210, 310, 410],   # Frame 2
            [0, 0, 0, 0],           # Frame 3 (occluded)
            [120, 220, 320, 420],   # Frame 4
            [130, 230, 330, 430],   # Frame 5
            [0, 0, 0, 0],           # Frame 6 (occluded)
            [140, 240, 340, 440],   # Frame 7
            [150, 250, 350, 450],   # Frame 8
            [160, 260, 360, 460],   # Frame 9
            [170, 270, 370, 470],   # Frame 10
        ]

        sonar_gt = [
            [50, 100, 150, 200],    # Frame 1
            [55, 105, 155, 205],    # Frame 2
            [60, 110, 160, 210],    # Frame 3
            [0, 0, 0, 0],           # Frame 4 (occluded)
            [65, 115, 165, 215],    # Frame 5
            [70, 120, 170, 220],    # Frame 6
            [75, 125, 175, 225],    # Frame 7
            [0, 0, 0, 0],           # Frame 8 (occluded)
            [80, 130, 180, 230],    # Frame 9
            [85, 135, 185, 235],    # Frame 10
        ]

        # Write groundtruth files
        with open(rgb_dir / "groundtruth.txt", 'w') as f:
            for bbox in rgb_gt:
                f.write(','.join(map(str, bbox)) + '\n')

        # Sonar uses groundtruth_rect.txt (rectangular coordinate annotations)
        with open(sonar_dir / "groundtruth_rect.txt", 'w') as f:
            for bbox in sonar_gt:
                f.write(','.join(map(str, bbox)) + '\n')

        yield temp_dir

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


class TestOcclusionUtils:
    """Test occlusion utility functions."""

    def test_is_occluded_true(self):
        """Test is_occluded returns True for zero bbox."""
        bbox = np.array([0, 0, 0, 0])
        assert is_occluded(bbox) is True

    def test_is_occluded_false(self):
        """Test is_occluded returns False for valid bbox."""
        bbox = np.array([100, 200, 300, 400])
        assert is_occluded(bbox) is False

    def test_is_occluded_partial_zero(self):
        """Test is_occluded with partial zeros."""
        bbox = np.array([0, 0, 100, 200])
        assert is_occluded(bbox) is False


class TestGroundtruthParsing:
    """Test groundtruth file parsing."""

    def test_parse_valid_groundtruth(self, temp_dataset_dir):
        """Test parsing valid groundtruth file."""
        gt_path = temp_dataset_dir / "test_seq" / "light" / "groundtruth.txt"
        bboxes = parse_groundtruth(gt_path)

        assert bboxes.shape == (10, 4)
        assert bboxes.dtype == np.float32
        assert np.array_equal(bboxes[0], [100, 200, 300, 400])
        assert np.array_equal(bboxes[2], [0, 0, 0, 0])  # Occluded frame

    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            parse_groundtruth("/nonexistent/groundtruth.txt")


class TestRGBSonarDatasetInitialization:
    """Test dataset initialization and validation."""

    def test_successful_initialization(self, temp_dataset_dir):
        """Test successful dataset initialization."""
        dataset = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        assert dataset.num_frames == 10
        assert len(dataset) == 10
        assert dataset.sequence_name == "test_seq"

    def test_nonexistent_directory(self, temp_dataset_dir):
        """Test initialization with nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            RGBSonarDataset(
                data_root=temp_dataset_dir,
                sequence_name="nonexistent_seq"
            )

    def test_return_occluded_filtering(self, temp_dataset_dir):
        """Test return_occluded filtering."""
        # With occlusions
        dataset_with_occ = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq",
            return_occluded=True
        )
        assert len(dataset_with_occ) == 10

        # Without occlusions (both modalities must be valid)
        dataset_without_occ = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq",
            return_occluded=False
        )
        # Frames where both RGB and Sonar are valid: 1, 2, 5, 7, 9, 10
        # Frame 3: RGB occluded
        # Frame 4: Sonar occluded
        # Frame 6: RGB occluded
        # Frame 8: Sonar occluded
        expected_valid = 6
        assert len(dataset_without_occ) == expected_valid

    def test_frame_range_filtering(self, temp_dataset_dir):
        """Test start_frame and end_frame filtering."""
        dataset = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq",
            start_frame=3,
            end_frame=7,
            return_occluded=True
        )

        # Frames 3-7 = 5 frames
        assert len(dataset) == 5


class TestDatasetAccess:
    """Test dataset indexing and data access."""

    def test_getitem_valid_frame(self, temp_dataset_dir):
        """Test accessing valid frame."""
        dataset = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        sample = dataset[0]  # First frame (index 0)

        assert sample is not None
        assert 'rgb_img' in sample
        assert 'sonar_img' in sample
        assert 'rgb_bbox' in sample
        assert 'sonar_bbox' in sample
        assert 'frame_id' in sample
        assert 'is_valid' in sample

        assert sample['frame_id'] == 1  # 1-based frame ID
        assert sample['rgb_img'].shape == (1080, 1920, 3)
        # Sonar is grayscale (rectangular coordinates)
        assert sample['sonar_img'].shape == (376, 682) or sample['sonar_img'].shape == (376, 682, 1)

    def test_getitem_occluded_frame(self, temp_dataset_dir):
        """Test accessing occluded frame."""
        dataset = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq",
            return_occluded=True
        )

        # Frame 3 (index 2) is occluded in RGB
        sample = dataset[2]

        assert sample['is_rgb_valid'] is False
        # CRITICAL FIX: Occluded bboxes are replaced with centered 1x1 bbox for collate_fn consistency
        # Expected bbox: [w//2, h//2, 1, 1] = [960, 540, 1, 1] for 1920x1080 image
        assert np.array_equal(sample['rgb_bbox'], np.array([960, 540, 1, 1], dtype=np.float32))

    def test_get_frame_by_id(self, temp_dataset_dir):
        """Test get_frame_by_id method."""
        dataset = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq"
        )

        frame = dataset.get_frame_by_id(5)  # 1-based ID

        assert frame['frame_id'] == 5
        assert frame['rgb_img'] is not None
        assert frame['sonar_img'] is not None

    def test_get_frame_by_id_out_of_range(self, temp_dataset_dir):
        """Test get_frame_by_id with invalid ID."""
        dataset = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq"
        )

        with pytest.raises(IndexError):
            dataset.get_frame_by_id(0)  # Invalid (1-based)

        with pytest.raises(IndexError):
            dataset.get_frame_by_id(100)  # Out of range


class TestSequenceInfo:
    """Test sequence metadata retrieval."""

    def test_get_sequence_info(self, temp_dataset_dir):
        """Test get_sequence_info method."""
        dataset = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq",
            return_occluded=False
        )

        info = dataset.get_sequence_info()

        assert info['sequence_name'] == "test_seq"
        assert info['num_frames'] == 10
        assert info['num_valid_frames'] == 6  # Excludes occluded frames
        assert info['rgb_resolution'] == (1920, 1080)
        assert info['sonar_resolution'] == (682, 376)


class TestLastValidBbox:
    """Test find_last_valid_bbox functionality."""

    def test_find_last_valid_bbox_rgb(self, temp_dataset_dir):
        """Test finding last valid RGB bbox."""
        dataset = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq"
        )

        # Frame 3 (index 2) is occluded in RGB
        # Last valid should be frame 2 (index 1)
        last_bbox = dataset.find_last_valid_bbox(frame_idx=2, modality="rgb")

        assert last_bbox is not None
        assert np.array_equal(last_bbox, [110, 210, 310, 410])

    def test_find_last_valid_bbox_sonar(self, temp_dataset_dir):
        """Test finding last valid Sonar bbox."""
        dataset = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq"
        )

        # Frame 4 (index 3) is occluded in Sonar
        # Last valid should be frame 3 (index 2)
        last_bbox = dataset.find_last_valid_bbox(frame_idx=3, modality="sonar")

        assert last_bbox is not None
        assert np.array_equal(last_bbox, [60, 110, 160, 210])

    def test_find_last_valid_bbox_both(self, temp_dataset_dir):
        """Test finding last valid bbox for both modalities."""
        dataset = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq"
        )

        # Frame 4 (index 3): RGB valid, Sonar occluded
        # Last frame with both valid: frame 2 (index 1)
        last_bbox = dataset.find_last_valid_bbox(frame_idx=3, modality="both")

        assert last_bbox is not None
        assert 'rgb_bbox' in last_bbox
        assert 'sonar_bbox' in last_bbox
        assert np.array_equal(last_bbox['rgb_bbox'], [110, 210, 310, 410])
        assert np.array_equal(last_bbox['sonar_bbox'], [55, 105, 155, 205])

    def test_find_last_valid_bbox_none(self, temp_dataset_dir):
        """Test when no valid bbox exists."""
        dataset = RGBSonarDataset(
            data_root=temp_dataset_dir,
            sequence_name="test_seq"
        )

        # Search from frame 0 (no previous frames)
        last_bbox = dataset.find_last_valid_bbox(frame_idx=0, modality="rgb")

        assert last_bbox is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])