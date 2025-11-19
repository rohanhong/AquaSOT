"""
Unit Tests for DataLoader Utilities

Tests the data loading infrastructure including:
- Collate functions for tracking and template-search
- Multi-sequence collation
- DataLoader creation and configuration
- Distributed sampling
- Batch processing

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import pytest
import numpy as np
import torch
from torch.utils.data import Dataset

from aquasot3d.dataloader.loader import (
    collate_fn_tracking,
    collate_fn_template_search,
    create_dataloader,
    InfiniteDataLoader,
)


class DummyTrackingDataset(Dataset):
    """Dummy dataset for testing tracking collate function."""

    def __init__(self, num_samples=20):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Simulate frame data
        is_valid = idx % 3 != 0  # Every 3rd frame is invalid

        sample = {
            'rgb_img': np.random.randn(3, 256, 256).astype(np.float32) if is_valid else None,
            'sonar_img': np.random.randn(3, 256, 256).astype(np.float32) if is_valid else None,
            'rgb_bbox': np.array([idx * 10, idx * 20, 100, 200], dtype=np.float32),
            'sonar_bbox': np.array([idx * 5, idx * 10, 50, 100], dtype=np.float32),
            'frame_id': idx + 1,
            'is_valid': is_valid,
            'is_rgb_valid': is_valid,
            'is_sonar_valid': is_valid,
        }

        return sample


class DummyTemplateSearchDataset(Dataset):
    """Dummy dataset for testing template-search collate function."""

    def __init__(self, num_samples=20):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Use trainer-expected field names: rgb_template, rgb_search, sonar_template, sonar_search
        sample = {
            'rgb_template': np.random.randn(128, 128, 3).astype(np.float32),
            'sonar_template': np.random.randn(128, 128, 3).astype(np.float32),
            'rgb_search': np.random.randn(256, 256, 3).astype(np.float32),
            'sonar_search': np.random.randn(256, 256, 3).astype(np.float32),
            'template_bbox_rgb': np.array([10, 20, 100, 200], dtype=np.float32),
            'template_bbox_sonar': np.array([5, 10, 50, 100], dtype=np.float32),
            'search_bbox_rgb': np.array([15, 25, 110, 210], dtype=np.float32),
            'search_bbox_sonar': np.array([8, 12, 55, 105], dtype=np.float32),
            'template_idx': 0,
            'search_idx': idx + 1,
            'is_valid': True,
            'is_rgb_valid': True,
            'is_sonar_valid': True,
        }

        return sample


class TestCollateTracking:
    """Test collate_fn_tracking function."""

    def test_collate_valid_batch(self):
        """Test collating batch with all valid samples."""
        batch = [
            {
                'rgb_img': torch.randn(3, 256, 256),
                'sonar_img': torch.randn(3, 256, 256),
                'rgb_bbox': np.array([10, 20, 100, 200], dtype=np.float32),
                'sonar_bbox': np.array([5, 10, 50, 100], dtype=np.float32),
                'frame_id': 1,
                'is_valid': True,
            },
            {
                'rgb_img': torch.randn(3, 256, 256),
                'sonar_img': torch.randn(3, 256, 256),
                'rgb_bbox': np.array([20, 30, 110, 210], dtype=np.float32),
                'sonar_bbox': np.array([10, 15, 55, 105], dtype=np.float32),
                'frame_id': 2,
                'is_valid': True,
            },
        ]

        batched = collate_fn_tracking(batch)

        assert batched is not None
        assert batched['rgb_img'].shape == (2, 3, 256, 256)
        assert batched['sonar_img'].shape == (2, 3, 256, 256)
        assert batched['rgb_bbox'].shape == (2, 4)
        assert batched['sonar_bbox'].shape == (2, 4)
        assert batched['frame_id'].shape == (2,)

    def test_collate_with_none_images(self):
        """Test collating batch with None images."""
        batch = [
            {
                'rgb_img': torch.randn(3, 256, 256),
                'sonar_img': None,
                'rgb_bbox': np.array([10, 20, 100, 200], dtype=np.float32),
                'sonar_bbox': np.array([0, 0, 0, 0], dtype=np.float32),
                'frame_id': 1,
                'is_valid': False,
            },
            {
                'rgb_img': torch.randn(3, 256, 256),
                'sonar_img': torch.randn(3, 256, 256),
                'rgb_bbox': np.array([20, 30, 110, 210], dtype=np.float32),
                'sonar_bbox': np.array([10, 15, 55, 105], dtype=np.float32),
                'frame_id': 2,
                'is_valid': True,
            },
        ]

        batched = collate_fn_tracking(batch)

        # RGB should have 2 samples
        assert batched['rgb_img'].shape == (2, 3, 256, 256)
        # Sonar should have only 1 sample (second one)
        assert batched['sonar_img'].shape == (1, 3, 256, 256)

    def test_collate_empty_batch(self):
        """Test collating empty batch."""
        batch = []
        batched = collate_fn_tracking(batch)

        assert batched is None


class TestCollateTemplateSearch:
    """Test collate_fn_template_search function."""

    def test_collate_template_search(self):
        """Test collating template-search batch."""
        # Use trainer-expected field names: rgb_template, rgb_search, sonar_template, sonar_search
        batch = [
            {
                'rgb_template': np.random.randn(128, 128, 3).astype(np.float32),
                'sonar_template': np.random.randn(128, 128, 3).astype(np.float32),
                'rgb_search': np.random.randn(256, 256, 3).astype(np.float32),
                'sonar_search': np.random.randn(256, 256, 3).astype(np.float32),
                'template_bbox_rgb': np.array([10, 20, 100, 200], dtype=np.float32),
                'template_bbox_sonar': np.array([5, 10, 50, 100], dtype=np.float32),
                'search_bbox_rgb': np.array([15, 25, 110, 210], dtype=np.float32),
                'search_bbox_sonar': np.array([8, 12, 55, 105], dtype=np.float32),
                'template_idx': 0,
                'search_idx': 1,
                'is_valid': True,
            },
            {
                'rgb_template': np.random.randn(128, 128, 3).astype(np.float32),
                'sonar_template': np.random.randn(128, 128, 3).astype(np.float32),
                'rgb_search': np.random.randn(256, 256, 3).astype(np.float32),
                'sonar_search': np.random.randn(256, 256, 3).astype(np.float32),
                'template_bbox_rgb': np.array([20, 30, 110, 210], dtype=np.float32),
                'template_bbox_sonar': np.array([10, 15, 55, 105], dtype=np.float32),
                'search_bbox_rgb': np.array([25, 35, 120, 220], dtype=np.float32),
                'search_bbox_sonar': np.array([12, 18, 60, 110], dtype=np.float32),
                'template_idx': 0,
                'search_idx': 2,
                'is_valid': True,
            },
        ]

        batched = collate_fn_template_search(batch)

        assert batched is not None
        # Images should be (B, C, H, W)
        assert batched['rgb_template'].shape == (2, 3, 128, 128)
        assert batched['sonar_template'].shape == (2, 3, 128, 128)
        assert batched['rgb_search'].shape == (2, 3, 256, 256)
        assert batched['sonar_search'].shape == (2, 3, 256, 256)
        # Bboxes should be (B, 4)
        assert batched['template_bbox_rgb'].shape == (2, 4)
        assert batched['search_bbox_sonar'].shape == (2, 4)


class TestDataLoaderCreation:
    """Test DataLoader creation and configuration."""

    def test_create_dataloader_basic(self):
        """Test basic DataLoader creation."""
        dataset = DummyTrackingDataset(num_samples=20)

        loader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 workers for testing
            pin_memory=False,
        )

        assert loader is not None
        assert len(loader) == 5  # 20 samples / 4 batch_size

    def test_create_dataloader_iteration(self):
        """Test iterating through DataLoader."""
        dataset = DummyTrackingDataset(num_samples=20)

        loader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        batches = list(loader)
        assert len(batches) == 5

        # Check first batch
        first_batch = batches[0]
        assert first_batch is not None
        # Should have at most 4 samples (some might be None)
        if 'rgb_img' in first_batch and first_batch['rgb_img'] is not None:
            assert first_batch['rgb_img'].shape[0] <= 4

    def test_create_dataloader_template_search(self):
        """Test DataLoader with template-search collate function."""
        dataset = DummyTemplateSearchDataset(num_samples=16)

        loader = create_dataloader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_template_search,
        )

        batch = next(iter(loader))

        assert batch is not None
        assert 'rgb_template' in batch
        assert 'rgb_search' in batch
        assert batch['rgb_template'].shape[0] == 8


class TestInfiniteDataLoader:
    """Test InfiniteDataLoader wrapper."""

    def test_infinite_iteration(self):
        """Test infinite DataLoader cycles correctly."""
        dataset = DummyTrackingDataset(num_samples=10)

        loader = create_dataloader(
            dataset,
            batch_size=5,
            shuffle=False,
            num_workers=0,
        )

        infinite_loader = InfiniteDataLoader(loader)

        # Iterate more than dataset length
        batches = []
        for i, batch in enumerate(infinite_loader):
            batches.append(batch)
            if i >= 5:  # More than 2 full epochs
                break

        # Should have 6 batches (more than one epoch)
        assert len(batches) == 6

    def test_infinite_length(self):
        """Test InfiniteDataLoader __len__ returns base loader length."""
        dataset = DummyTrackingDataset(num_samples=20)

        loader = create_dataloader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
        )

        infinite_loader = InfiniteDataLoader(loader)

        assert len(infinite_loader) == len(loader)


class TestMultiSequenceCollation:
    """Test multi-sequence collation functions."""

    def test_collate_multi_sequence_tracking(self):
        """Test collating multi-sequence tracking batch."""
        batch = [
            {
                'rgb_img': torch.randn(3, 256, 256),
                'sonar_img': torch.randn(3, 256, 256),
                'rgb_bbox': np.array([10, 20, 100, 200], dtype=np.float32),
                'sonar_bbox': np.array([5, 10, 50, 100], dtype=np.float32),
                'frame_id': 1,
                'sequence_name': 'uuv2',
                'is_valid': True,
                'is_rgb_valid': True,
                'is_sonar_valid': True,
            },
            {
                'rgb_img': torch.randn(3, 256, 256),
                'sonar_img': torch.randn(3, 256, 256),
                'rgb_bbox': np.array([20, 30, 110, 210], dtype=np.float32),
                'sonar_bbox': np.array([10, 15, 55, 105], dtype=np.float32),
                'frame_id': 15,
                'sequence_name': 'uuv3',
                'is_valid': True,
                'is_rgb_valid': True,
                'is_sonar_valid': True,
            },
            {
                'rgb_img': torch.randn(3, 256, 256),
                'sonar_img': torch.randn(3, 256, 256),
                'rgb_bbox': np.array([30, 40, 120, 220], dtype=np.float32),
                'sonar_bbox': np.array([15, 20, 60, 110], dtype=np.float32),
                'frame_id': 42,
                'sequence_name': 'uuv2',
                'is_valid': True,
                'is_rgb_valid': True,
                'is_sonar_valid': True,
            },
        ]

        batched = collate_fn_tracking(batch)

        assert batched is not None
        assert batched['rgb_img'].shape == (3, 3, 256, 256)
        assert batched['sonar_img'].shape == (3, 3, 256, 256)
        assert batched['rgb_bbox'].shape == (3, 4)
        assert batched['sonar_bbox'].shape == (3, 4)
        assert batched['frame_id'].shape == (3,)

        # Verify sequence names are preserved
        assert 'sequence_name' in batched
        assert len(batched['sequence_name']) == 3
        assert batched['sequence_name'][0] == 'uuv2'
        assert batched['sequence_name'][1] == 'uuv3'
        assert batched['sequence_name'][2] == 'uuv2'

    def test_collate_multi_sequence_template_search(self):
        """Test collating multi-sequence template-search batch."""
        # Use trainer-expected field names
        batch = [
            {
                'rgb_template': np.random.randn(128, 128, 3).astype(np.float32),
                'sonar_template': np.random.randn(128, 128, 3).astype(np.float32),
                'rgb_search': np.random.randn(256, 256, 3).astype(np.float32),
                'sonar_search': np.random.randn(256, 256, 3).astype(np.float32),
                'template_bbox_rgb': np.array([10, 20, 100, 200], dtype=np.float32),
                'template_bbox_sonar': np.array([5, 10, 50, 100], dtype=np.float32),
                'search_bbox_rgb': np.array([15, 25, 110, 210], dtype=np.float32),
                'search_bbox_sonar': np.array([8, 12, 55, 105], dtype=np.float32),
                'template_idx': 0,
                'search_idx': 1,
                'template_sequence': 'uuv2',
                'search_sequence': 'uuv2',
                'template_frame_id': 1,
                'search_frame_id': 2,
                'is_valid': True,
            },
            {
                'rgb_template': np.random.randn(128, 128, 3).astype(np.float32),
                'sonar_template': np.random.randn(128, 128, 3).astype(np.float32),
                'rgb_search': np.random.randn(256, 256, 3).astype(np.float32),
                'sonar_search': np.random.randn(256, 256, 3).astype(np.float32),
                'template_bbox_rgb': np.array([20, 30, 110, 210], dtype=np.float32),
                'template_bbox_sonar': np.array([10, 15, 55, 105], dtype=np.float32),
                'search_bbox_rgb': np.array([25, 35, 120, 220], dtype=np.float32),
                'search_bbox_sonar': np.array([12, 18, 60, 110], dtype=np.float32),
                'template_idx': 0,
                'search_idx': 10,
                'template_sequence': 'uuv3',
                'search_sequence': 'uuv3',
                'template_frame_id': 1,
                'search_frame_id': 11,
                'is_valid': True,
            },
        ]

        batched = collate_fn_template_search(batch)

        assert batched is not None
        assert batched['rgb_template'].shape == (2, 3, 128, 128)
        assert batched['rgb_search'].shape == (2, 3, 256, 256)

        # Verify sequence names are preserved
        assert 'template_sequence' in batched
        assert 'search_sequence' in batched
        assert len(batched['template_sequence']) == 2
        assert len(batched['search_sequence']) == 2
        assert batched['template_sequence'][0] == 'uuv2'
        assert batched['template_sequence'][1] == 'uuv3'
        assert batched['search_sequence'][0] == 'uuv2'
        assert batched['search_sequence'][1] == 'uuv3'

        # Verify frame IDs are preserved
        assert 'template_frame_id' in batched
        assert 'search_frame_id' in batched
        assert batched['template_frame_id'].shape == (2,)
        assert batched['search_frame_id'].shape == (2,)

    def test_collate_mixed_sequences_with_occlusions(self):
        """Test collating multi-sequence batch with mixed occlusions."""
        batch = [
            {
                'rgb_img': torch.randn(3, 256, 256),
                'sonar_img': None,  # Sonar occluded
                'rgb_bbox': np.array([10, 20, 100, 200], dtype=np.float32),
                'sonar_bbox': np.array([0, 0, 0, 0], dtype=np.float32),
                'frame_id': 5,
                'sequence_name': 'uuv2',
                'is_valid': False,
                'is_rgb_valid': True,
                'is_sonar_valid': False,
            },
            {
                'rgb_img': torch.randn(3, 256, 256),
                'sonar_img': torch.randn(3, 256, 256),
                'rgb_bbox': np.array([20, 30, 110, 210], dtype=np.float32),
                'sonar_bbox': np.array([10, 15, 55, 105], dtype=np.float32),
                'frame_id': 12,
                'sequence_name': 'uuv3',
                'is_valid': True,
                'is_rgb_valid': True,
                'is_sonar_valid': True,
            },
        ]

        batched = collate_fn_tracking(batch)

        assert batched is not None
        # RGB should have 2 samples
        assert batched['rgb_img'].shape == (2, 3, 256, 256)
        # Sonar should have only 1 sample (second one)
        assert batched['sonar_img'].shape == (1, 3, 256, 256)

        # Verify sequence names present
        assert 'sequence_name' in batched
        assert len(batched['sequence_name']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])