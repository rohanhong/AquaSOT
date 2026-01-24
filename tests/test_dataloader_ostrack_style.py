"""
Test data loading pipeline following OSTrack structure.

This module tests:
1. BaseVideoDataset interface
2. AquaSOTDataset for dual-modal data
3. MultiModalSampler for frame sampling
4. Processing utilities for cropping
5. Transforms for data augmentation
6. Integration with LTRLoader
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
# BaseVideoDataset Tests
# ==============================================================================

class TestBaseVideoDataset:
    """Test base video dataset interface."""

    def test_base_dataset_import(self):
        """Test that base dataset can be imported."""
        from lib.train.dataset import BaseVideoDataset
        assert BaseVideoDataset is not None

    def test_base_dataset_creation(self):
        """Test creating base dataset instance."""
        from lib.train.dataset import BaseVideoDataset

        class DummyDataset(BaseVideoDataset):
            def __init__(self):
                super().__init__('dummy', '/tmp')
                self.sequence_list = ['seq1', 'seq2']

            def get_sequence_info(self, seq_id):
                return {'visible': torch.ones(100)}

            def get_frames(self, seq_id, frame_ids, anno=None):
                return [], {}, {}

        ds = DummyDataset()
        assert ds.get_name() == 'dummy'
        assert ds.get_num_sequences() == 2
        assert len(ds) == 2


# ==============================================================================
# AquaSOTDataset Tests
# ==============================================================================

class TestAquaSOTDataset:
    """Test AquaSOT dual-modal dataset."""

    def test_dataset_import(self):
        """Test that AquaSOTDataset can be imported."""
        from lib.train.dataset import AquaSOTDataset
        assert AquaSOTDataset is not None

    def test_dataset_with_mock_data(self, tmp_path):
        """Test dataset with mock data structure."""
        from lib.train.dataset import AquaSOTDataset

        # Create mock data structure (matching actual AquaSOT structure)
        seq_path = tmp_path / "uuv_test"
        rgb_path = seq_path / "light" / "img"
        sonar_path = seq_path / "sonar" / "img_rect"
        rgb_path.mkdir(parents=True)
        sonar_path.mkdir(parents=True)

        # Create mock images (10 frames)
        for i in range(10):
            # RGB image (3 channels)
            rgb_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(rgb_path / f"{i:06d}.jpg"), rgb_img)

            # Sonar image (grayscale)
            sonar_img = np.random.randint(0, 255, (376, 682), dtype=np.uint8)
            cv2.imwrite(str(sonar_path / f"{i:06d}.jpg"), sonar_img)

        # Create groundtruth files
        with open(seq_path / "light" / "groundtruth.txt", 'w') as f:
            for i in range(10):
                f.write(f"{100+i},{100+i},{50},{50}\n")

        with open(seq_path / "sonar" / "groundtruth_rect.txt", 'w') as f:
            for i in range(10):
                f.write(f"{100+i},{100+i},{50},{50}\n")

        # Create dataset
        dataset = AquaSOTDataset(str(tmp_path), sequence_names=['uuv_test'], use_rect_sonar=True)

        # Test basic properties
        assert dataset.get_num_sequences() == 1
        assert dataset.get_name() == 'aquasot'

        # Test sequence info
        seq_info = dataset.get_sequence_info(0)
        assert 'rgb_bbox' in seq_info
        assert 'sonar_bbox' in seq_info
        assert 'visible' in seq_info
        assert len(seq_info['rgb_frames']) == 10

        # Test get_frames
        rgb_frames, sonar_frames, rgb_anno, sonar_anno, meta = \
            dataset.get_frames(0, [0, 1, 2], seq_info)

        assert len(rgb_frames) == 3
        assert len(sonar_frames) == 3
        assert len(rgb_anno['bbox']) == 3
        assert rgb_frames[0].shape == (480, 640, 3)
        assert sonar_frames[0].shape == (376, 682)  # Rect-converted sonar size


# ==============================================================================
# MultiModalSampler Tests
# ==============================================================================

class TestMultiModalSampler:
    """Test multi-modal frame sampler."""

    def test_sampler_import(self):
        """Test that sampler can be imported."""
        from lib.train.data import MultiModalSampler
        assert MultiModalSampler is not None

    def test_sampler_with_mock_dataset(self, tmp_path):
        """Test sampler with mock dataset."""
        from lib.train.dataset import AquaSOTDataset
        from lib.train.data import MultiModalSampler

        # Create mock data (matching actual AquaSOT structure)
        seq_path = tmp_path / "uuv_test"
        rgb_path = seq_path / "light" / "img"
        sonar_path = seq_path / "sonar" / "img_rect"
        rgb_path.mkdir(parents=True)
        sonar_path.mkdir(parents=True)

        import cv2
        for i in range(30):  # Need at least 20 frames
            rgb_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(rgb_path / f"{i:06d}.jpg"), rgb_img)
            sonar_img = np.random.randint(0, 255, (376, 682), dtype=np.uint8)
            cv2.imwrite(str(sonar_path / f"{i:06d}.jpg"), sonar_img)

        with open(seq_path / "light" / "groundtruth.txt", 'w') as f:
            for i in range(30):
                f.write(f"{100},{100},{50},{50}\n")
        with open(seq_path / "sonar" / "groundtruth_rect.txt", 'w') as f:
            for i in range(30):
                f.write(f"{100},{100},{50},{50}\n")

        dataset = AquaSOTDataset(str(tmp_path), sequence_names=['uuv_test'], use_rect_sonar=True)

        # Create sampler
        sampler = MultiModalSampler(
            datasets=[dataset],
            samples_per_epoch=10,
            max_gap=5,
            num_search_frames=1,
            num_template_frames=1,
        )

        assert len(sampler) == 10

        # Sample one item
        data = sampler[0]
        assert 'rgb_template_images' in data
        assert 'rgb_search_images' in data
        assert 'sonar_template_images' in data
        assert 'sonar_search_images' in data


# ==============================================================================
# Processing Tests
# ==============================================================================

class TestProcessing:
    """Test data processing utilities."""

    def test_processing_utils_import(self):
        """Test that processing utils can be imported."""
        from lib.train.data import processing_utils
        assert processing_utils is not None

    def test_sample_target(self):
        """Test sample_target function."""
        from lib.train.data.processing_utils import sample_target

        # Create test image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = [200, 200, 100, 100]  # [x, y, w, h]

        # Sample crop
        crop, resize_factor, att_mask = sample_target(img, bbox, 4.0, 256)

        assert crop.shape == (256, 256, 3)
        assert att_mask.shape == (256, 256)
        assert resize_factor > 0

    def test_sample_target_grayscale(self):
        """Test sample_target with grayscale image."""
        from lib.train.data.processing_utils import sample_target

        # Create grayscale test image
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        bbox = [200, 200, 100, 100]

        crop, resize_factor, att_mask = sample_target(img, bbox, 4.0, 256)

        assert crop.shape == (256, 256)
        assert att_mask.shape == (256, 256)

    def test_jittered_center_crop(self):
        """Test jittered center crop."""
        from lib.train.data.processing_utils import jittered_center_crop

        # Create test images
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)]
        box_extract = [torch.tensor([200.0, 200.0, 100.0, 100.0])]
        box_gt = [torch.tensor([200.0, 200.0, 100.0, 100.0])]

        crops, boxes, att_masks, _ = jittered_center_crop(
            images, box_extract, box_gt, 4.0, 256
        )

        assert len(crops) == 1
        assert crops[0].shape == (256, 256, 3)
        assert boxes[0].shape == (4,)
        # Box should be normalized
        assert boxes[0].max() <= 1.0


# ==============================================================================
# Transforms Tests
# ==============================================================================

class TestTransforms:
    """Test data augmentation transforms."""

    def test_transforms_import(self):
        """Test that transforms can be imported."""
        from lib.train.data import transforms
        assert transforms is not None

    def test_to_tensor(self):
        """Test ToTensor transform."""
        from lib.train.data.transforms import ToTensor

        transform = ToTensor()
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        result = transform(image=img)
        tensor = result['image']

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 256, 256)
        assert tensor.max() <= 1.0

    def test_normalize(self):
        """Test Normalize transform."""
        from lib.train.data.transforms import Normalize

        transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = torch.rand(3, 256, 256)

        result = transform(image=img)
        tensor = result['image']

        assert tensor.shape == (3, 256, 256)

    def test_random_horizontal_flip(self):
        """Test RandomHorizontalFlip transform."""
        from lib.train.data.transforms import RandomHorizontalFlip

        transform = RandomHorizontalFlip(probability=1.0)  # Always flip
        img = np.arange(9).reshape(3, 3, 1).astype(np.uint8)

        result = transform(image=img)
        flipped = result['image']

        # Check that image is flipped horizontally
        assert np.array_equal(flipped[:, 0, :], img[:, 2, :])

    def test_sonar_to_tensor(self):
        """Test SonarToTensor transform."""
        from lib.train.data.transforms import SonarToTensor

        transform = SonarToTensor()
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        result = transform(image=img)
        tensor = result['image']

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 256, 256)
        assert tensor.max() <= 1.0

    def test_transform_chain(self):
        """Test chaining multiple transforms."""
        from lib.train.data.transforms import Transform, ToTensor, Normalize

        transform = Transform(
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        bbox = torch.tensor([100.0, 100.0, 50.0, 50.0])

        result = transform(image=img, bbox=bbox)

        assert result[0].shape == (3, 256, 256)  # Image
        assert result[1].shape == (4,)  # BBox unchanged


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestDataLoaderIntegration:
    """Integration tests for data loading pipeline."""

    def test_ltr_loader_import(self):
        """Test that LTRLoader can be imported."""
        from lib.train.data import LTRLoader
        assert LTRLoader is not None

    def test_full_pipeline_smoke(self, tmp_path):
        """Smoke test for full data loading pipeline."""
        from lib.train.dataset import AquaSOTDataset
        from lib.train.data import MultiModalSampler, LTRLoader

        # Create mock data (matching actual AquaSOT structure)
        seq_path = tmp_path / "uuv_test"
        rgb_path = seq_path / "light" / "img"
        sonar_path = seq_path / "sonar" / "img_rect"
        rgb_path.mkdir(parents=True)
        sonar_path.mkdir(parents=True)

        import cv2
        for i in range(30):
            rgb_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(rgb_path / f"{i:06d}.jpg"), rgb_img)
            sonar_img = np.random.randint(0, 255, (376, 682), dtype=np.uint8)
            cv2.imwrite(str(sonar_path / f"{i:06d}.jpg"), sonar_img)

        with open(seq_path / "light" / "groundtruth.txt", 'w') as f:
            for i in range(30):
                f.write(f"{100},{100},{50},{50}\n")
        with open(seq_path / "sonar" / "groundtruth_rect.txt", 'w') as f:
            for i in range(30):
                f.write(f"{100},{100},{50},{50}\n")

        # Create pipeline
        dataset = AquaSOTDataset(str(tmp_path), sequence_names=['uuv_test'], use_rect_sonar=True)
        sampler = MultiModalSampler(
            datasets=[dataset],
            samples_per_epoch=4,
            max_gap=5,
            num_search_frames=1,
            num_template_frames=1,
        )

        loader = LTRLoader(
            'train',
            sampler,
            training=True,
            batch_size=2,
            num_workers=0,
            drop_last=True,
            stack_dim=1,
        )

        assert len(loader) == 2  # 4 samples / batch_size 2

        # Get one batch
        batch = next(iter(loader))

        # Check batch structure
        assert 'rgb_template_images' in batch
        assert 'rgb_search_images' in batch
        assert 'sonar_template_images' in batch


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
