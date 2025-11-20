"""
Migration Validation Tests

Validates that the Phase 1-3 migration is complete and functional.
Tests basic data flow from dataset through models.

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
from aquasot3d.models.backbones.vit import ViTBackbone


@pytest.fixture
def temp_dataset():
    """Create temporary dataset for validation."""
    temp_dir = Path(tempfile.mkdtemp())

    try:
        seq_dir = temp_dir / "val_seq"
        rgb_dir = seq_dir / "light"
        sonar_dir = seq_dir / "sonar"
        rgb_img_dir = rgb_dir / "img"
        sonar_img_dir = sonar_dir / "img_rect"

        rgb_img_dir.mkdir(parents=True, exist_ok=True)
        sonar_img_dir.mkdir(parents=True, exist_ok=True)

        # Create 15 frames
        for i in range(1, 16):
            rgb_img = Image.new('RGB', (1920, 1080), color=(i * 15, i * 15, i * 15))
            rgb_img.save(rgb_img_dir / f"{i}.jpg")

            sonar_img = Image.new('L', (682, 376), color=(i * 15))
            sonar_img.save(sonar_img_dir / f"{i}.jpg")

        rgb_gt = [[100 + i*5, 200 + i*5, 300, 400] for i in range(1, 16)]
        sonar_gt = [[50 + i*3, 100 + i*3, 150, 200] for i in range(1, 16)]

        with open(rgb_dir / "groundtruth.txt", 'w') as f:
            for bbox in rgb_gt:
                f.write(','.join(map(str, bbox)) + '\n')

        with open(sonar_dir / "groundtruth_rect.txt", 'w') as f:
            for bbox in sonar_gt:
                f.write(','.join(map(str, bbox)) + '\n')

        yield temp_dir

    finally:
        shutil.rmtree(temp_dir)


class TestMigrationValidation:
    """Validate complete migration is functional."""

    def test_no_deprecated_imports(self):
        """Verify deprecated classes are removed from imports."""
        from aquasot3d.dataloader import datasets

        # TrackingDataset should not exist
        assert not hasattr(datasets, 'TrackingDataset')

        # FrameSequenceDataset should exist
        assert hasattr(datasets, 'FrameSequenceDataset')

    def test_data_pipeline_complete(self, temp_dataset):
        """Test complete data pipeline with new format."""
        # Create dataset
        base_dataset = RGBSonarDataset(
            data_root=temp_dataset,
            sequence_name="val_seq",
            return_occluded=True
        )

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

        # Get batch
        batch = next(iter(dataloader))

        # Verify format
        assert 'rgb_frames' in batch
        assert 'sonar_frames' in batch
        assert batch['rgb_frames'].shape == (2, 5, 3, 512, 512)
        assert batch['sonar_frames'].shape == (2, 5, 1, 512, 512)

    def test_vit_processes_frame_sequence(self, temp_dataset):
        """Test ViT can process frame sequences."""
        # Create dataset
        base_dataset = RGBSonarDataset(
            data_root=temp_dataset,
            sequence_name="val_seq",
            return_occluded=True,
            modalities=['rgb']
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=224,  # ViT standard size
            frame_stride=1
        )

        # Get sample
        sample = dataset[5]
        rgb_frames = sample['rgb_frames']  # [T, 3, H, W]

        # Create ViT
        vit = ViTBackbone(
            model_name='vit_base_patch16_224',
            pretrained=False,
            in_channels=3
        )
        vit.eval()

        # Add batch dimension [B, T, 3, H, W]
        rgb_frames = rgb_frames.unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            outputs = vit(rgb_frames)

        # Verify output
        assert 'features' in outputs
        assert outputs['is_sequence'] == True
        assert outputs['batch_size'] == 1
        assert outputs['sequence_length'] == 5
        # Features should be [B*T, N, C]
        assert outputs['features'].shape[0] == 5  # B*T = 1*5

    def test_config_has_new_parameters(self):
        """Verify config has new frame-level parameters."""
        from aquasot3d.utils.config import load_config

        config = load_config('configs/base/default.yaml')

        # New frame-level training parameters should exist
        assert hasattr(config.data, 'frame_size')
        assert hasattr(config.data, 'sequence_length')
        assert hasattr(config.data, 'frame_stride')

        # Evaluation parameters should exist (used by EvaluationSequenceDataset)
        # CRITICAL: template_size/search_size are now evaluation-only parameters (Audit 2025112006)
        # They are used by eval.py for EvaluationSequenceDataset which maintains template-search paradigm
        assert hasattr(config.data, 'template_size')
        assert hasattr(config.data, 'search_size')

        # Deprecated training parameter should not exist
        assert not hasattr(config.data, 'template_search_gap')

    def test_end_to_end_data_to_vit(self, temp_dataset):
        """Test end-to-end from dataloader to ViT."""
        # Setup pipeline
        base_dataset = RGBSonarDataset(
            data_root=temp_dataset,
            sequence_name="val_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=3,
            frame_size=224,
            frame_stride=1
        )

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn_sequence
        )

        # Create model
        vit_rgb = ViTBackbone(
            model_name='vit_base_patch16_224',
            pretrained=False,
            in_channels=3
        )
        vit_sonar = ViTBackbone(
            model_name='vit_base_patch16_224',
            pretrained=False,
            in_channels=1
        )
        vit_rgb.eval()
        vit_sonar.eval()

        # Process batch
        batch = next(iter(dataloader))

        with torch.no_grad():
            # Process RGB frames
            rgb_out = vit_rgb(batch['rgb_frames'])
            # Process Sonar frames
            sonar_out = vit_sonar(batch['sonar_frames'])

        # Verify outputs
        # Input: [B=2, T=3, C, H, W] -> Output: [B*T=6, N, C]
        assert rgb_out['features'].shape[0] == 6
        assert sonar_out['features'].shape[0] == 6
        assert rgb_out['features'].shape[2] == 768  # ViT-Base embedding

        # Verify we can reshape back
        B, T = 2, 3
        N, C = rgb_out['features'].shape[1], rgb_out['features'].shape[2]

        # Reshape to [B, T, N, C]
        rgb_feat_seq = rgb_out['features'].reshape(B, T, N, C)
        assert rgb_feat_seq.shape == (2, 3, N, 768)


class TestNoRegressions:
    """Test that critical functionality is preserved."""

    def test_validity_flags_preserved(self, temp_dataset):
        """Test 0000 handling is preserved."""
        base_dataset = RGBSonarDataset(
            data_root=temp_dataset,
            sequence_name="val_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512
        )

        sample = dataset[5]

        # Validity flags should exist
        assert 'is_valid_rgb' in sample
        assert 'is_valid_sonar' in sample
        assert sample['is_valid_rgb'].shape == (5,)
        assert sample['is_valid_sonar'].shape == (5,)

    def test_single_modality_preserved(self, temp_dataset):
        """Test single-modality training is preserved."""
        base_dataset = RGBSonarDataset(
            data_root=temp_dataset,
            sequence_name="val_seq",
            return_occluded=True,
            modalities=['rgb']
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=5,
            frame_size=512
        )

        sample = dataset[5]

        # RGB should exist, Sonar should be None
        assert sample['rgb_frames'] is not None
        assert sample['sonar_frames'] is None

    def test_bbox_scaling_preserved(self, temp_dataset):
        """Test bbox scaling fixes are preserved."""
        base_dataset = RGBSonarDataset(
            data_root=temp_dataset,
            sequence_name="val_seq",
            return_occluded=True
        )

        dataset = FrameSequenceDataset(
            base_dataset=base_dataset,
            sequence_length=3,
            frame_size=512
        )

        sample = dataset[5]

        # Bboxes should be scaled
        assert sample['rgb_bboxes'].shape == (3, 4)
        assert sample['rgb_bboxes'].min() >= 0
        assert sample['rgb_bboxes'].max() <= 384

        # Metadata should have scale factors
        assert len(sample['rgb_metadata']) == 3
        assert 'scale_x' in sample['rgb_metadata'][0]
        assert 'scale_y' in sample['rgb_metadata'][0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
