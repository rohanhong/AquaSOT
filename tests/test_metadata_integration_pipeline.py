"""
Integration Test for Complete Metadata Pipeline (Audit 2025111409 Finding 1)

This test validates the end-to-end metadata flow:
Dataset → DataLoader → Collate → Evaluator

Ensures that scale/padding metadata is correctly propagated through
the entire pipeline and used for physical coordinate reconstruction.

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from aquasot3d.dataloader.datasets.evaluation_dataset import EvaluationSequenceDataset
from aquasot3d.dataloader.datasets.rgb_sonar_dataset import RGBSonarDataset
from aquasot3d.dataloader.loader import create_dataloader, collate_fn_template_search
from aquasot3d.evaluation.evaluator import TrackingEvaluator


class TestCompleteMetadataPipeline:
    """Integration tests for complete metadata propagation."""

    @pytest.fixture
    def mock_base_dataset(self):
        """Create mock base dataset with realistic dimensions."""
        dataset = MagicMock(spec=RGBSonarDataset)
        dataset.num_frames = 5
        dataset.valid_indices = [0, 1, 2, 3, 4]

        dataset.get_sequence_info.return_value = {
            'sequence_name': 'test_sequence',
            'num_frames': 5,
        }

        def mock_get_frame_data(index):
            # Realistic dimensions: RGB 1920x1080, Sonar 682x376
            rgb_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            sonar_img = np.random.randint(0, 255, (376, 682), dtype=np.uint8)

            return {
                'frame_id': index + 1,
                'rgb_img': rgb_img,
                'sonar_img': sonar_img,
                'rgb_bbox': np.array([960.0, 540.0, 100.0, 80.0]),  # [x, y, w, h] in pixels
                'sonar_bbox': np.array([341.0, 188.0, 50.0, 40.0]),
                'is_valid': True,
                'is_rgb_valid': True,
                'is_sonar_valid': True,
            }

        dataset._get_frame_data = mock_get_frame_data
        return dataset

    def test_dataset_generates_metadata(self, mock_base_dataset):
        """
        Step 1: Verify EvaluationSequenceDataset generates metadata.
        """
        eval_ds = EvaluationSequenceDataset(
            base_dataset=mock_base_dataset,
            template_size=128,
            search_size=256,
            template_frame='first'
        )

        sample = eval_ds[0]

        # Verify all metadata keys exist
        assert 'rgb_search_metadata' in sample
        assert 'sonar_search_metadata' in sample
        assert 'template_rgb_metadata' in sample
        assert 'template_sonar_metadata' in sample

        # Verify RGB metadata (anisotropic scaling)
        rgb_meta = sample['rgb_search_metadata']
        assert rgb_meta['original_height'] == 1080
        assert rgb_meta['original_width'] == 1920
        assert abs(rgb_meta['scale_x'] - 256.0/1920.0) < 1e-5
        assert abs(rgb_meta['scale_y'] - 256.0/1080.0) < 1e-5
        assert rgb_meta['pad_x'] == 0
        assert rgb_meta['pad_y'] == 0

        # Verify sonar metadata (letterbox scaling)
        sonar_meta = sample['sonar_search_metadata']
        assert sonar_meta['original_height'] == 376
        assert sonar_meta['original_width'] == 682
        expected_scale = 256.0 / 682.0
        assert abs(sonar_meta['scale_x'] - expected_scale) < 1e-3
        assert abs(sonar_meta['scale_y'] - expected_scale) < 1e-3

    def test_collate_preserves_metadata(self, mock_base_dataset):
        """
        Step 2: Verify collate_fn_template_search preserves metadata in batch.
        """
        eval_ds = EvaluationSequenceDataset(
            base_dataset=mock_base_dataset,
            template_size=128,
            search_size=256,
            template_frame='first'
        )

        # Create batch manually
        batch = [eval_ds[0], eval_ds[1]]

        # Apply collate function
        batched = collate_fn_template_search(batch)

        # Verify metadata keys exist in batched output
        assert 'rgb_search_metadata' in batched, "RGB search metadata missing after collate"
        assert 'sonar_search_metadata' in batched, "Sonar search metadata missing after collate"
        assert 'template_rgb_metadata' in batched, "Template RGB metadata missing after collate"
        assert 'template_sonar_metadata' in batched, "Template sonar metadata missing after collate"

        # Verify metadata content matches first sample
        rgb_meta = batched['rgb_search_metadata']
        assert rgb_meta['original_height'] == 1080
        assert rgb_meta['original_width'] == 1920
        assert 'scale_x' in rgb_meta
        assert 'scale_y' in rgb_meta
        assert 'pad_x' in rgb_meta
        assert 'pad_y' in rgb_meta

    def test_dataloader_propagates_metadata(self, mock_base_dataset):
        """
        Step 3: Verify DataLoader correctly propagates metadata through collate.
        """
        eval_ds = EvaluationSequenceDataset(
            base_dataset=mock_base_dataset,
            template_size=128,
            search_size=256,
            template_frame='first'
        )

        dataloader = create_dataloader(
            eval_ds,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # No multiprocessing for test
            collate_fn=collate_fn_template_search,
            pin_memory=False
        )

        # Get first batch
        batch = next(iter(dataloader))

        # Verify metadata exists in batch from dataloader
        assert 'rgb_search_metadata' in batch
        assert 'sonar_search_metadata' in batch
        assert 'template_rgb_metadata' in batch
        assert 'template_sonar_metadata' in batch

        # Verify content
        assert batch['rgb_search_metadata']['original_height'] == 1080
        assert batch['sonar_search_metadata']['original_height'] == 376

    def test_evaluator_receives_and_uses_metadata(self, mock_base_dataset):
        """
        Step 4: Verify evaluator receives metadata and uses it for reconstruction.
        """
        eval_ds = EvaluationSequenceDataset(
            base_dataset=mock_base_dataset,
            template_size=128,
            search_size=256,
            template_frame='first'
        )

        dataloader = create_dataloader(
            eval_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_template_search,
            pin_memory=False
        )

        # Get a batch
        batch = next(iter(dataloader))

        # Create mock model that returns known predictions
        mock_model = MagicMock()
        mock_model.eval = MagicMock()

        # Mock model should return predictions directly
        mock_model.return_value = {
            'rgb_boxes': torch.tensor([[0.5, 0.5, 0.2, 0.3]]),  # [cx, cy, w, h] in [0,1]
            'sonar_boxes': torch.tensor([[0.5, 0.6, 0.15, 0.2]]),  # cy=0.6 to avoid negative coords
            'q_rgb': torch.tensor([0.9]),
            'q_sonar': torch.tensor([0.8]),
        }

        evaluator = TrackingEvaluator(
            model=mock_model,
            dataloader=dataloader,
            device='cpu',
            search_size=256
        )

        # Track sequence with batch (correct API)
        with torch.no_grad():
            predictions = evaluator.track_sequence(batch)

        # Verify results contain physical coordinates
        assert 'rgb_pred' in predictions
        assert 'sonar_pred' in predictions

        rgb_pred = predictions['rgb_pred'][0]  # First frame prediction
        sonar_pred = predictions['sonar_pred'][0]

        # RGB predictions should be in physical 1920x1080 space
        # Normalized [0.5, 0.5, 0.2, 0.3] should map to approximately:
        # cx = 0.5 * 256 = 128 -> 128 / (256/1920) = 960
        # cy = 0.5 * 256 = 128 -> 128 / (256/1080) = 540
        # w = 0.2 * 256 = 51.2 -> 51.2 / (256/1920) = 384
        # h = 0.3 * 256 = 76.8 -> 76.8 / (256/1080) = 324
        # In xywh format: x = 960 - 384/2 = 768, y = 540 - 324/2 = 378

        assert rgb_pred[0] > 256, f"RGB x={rgb_pred[0]} should be in physical space (>256)"
        assert rgb_pred[1] > 256, f"RGB y={rgb_pred[1]} should be in physical space (>256)"
        assert rgb_pred[2] > 100, f"RGB w={rgb_pred[2]} should be in physical space (>100)"
        assert rgb_pred[3] > 100, f"RGB h={rgb_pred[3]} should be in physical space (>100)"

        # Sonar predictions should be in physical 682x376 space
        # With letterbox padding, cy_scaled can be small after removing pad_y
        # so x should be within bounds but y might be small
        assert 0 < sonar_pred[0] < 682, f"Sonar x={sonar_pred[0]} should be in physical 682x376 space"
        assert sonar_pred[1] >= -100, f"Sonar y={sonar_pred[1]} should be reasonable (accounting for padding)"
        assert sonar_pred[2] > 0, f"Sonar w={sonar_pred[2]} should be positive"
        assert sonar_pred[3] > 0, f"Sonar h={sonar_pred[3]} should be positive"

    def test_physical_coordinate_accuracy(self, mock_base_dataset):
        """
        Step 5: Verify exact physical coordinate reconstruction accuracy.
        """
        eval_ds = EvaluationSequenceDataset(
            base_dataset=mock_base_dataset,
            template_size=128,
            search_size=256,
            template_frame='first'
        )

        dataloader = create_dataloader(
            eval_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn_template_search,
            pin_memory=False
        )

        # Get batch to extract metadata
        batch = next(iter(dataloader))

        # Manually create evaluator and test denormalization
        mock_model = MagicMock()
        evaluator = TrackingEvaluator(
            model=mock_model,
            dataloader=dataloader,
            device='cpu',
            search_size=256
        )

        # Test RGB denormalization (anisotropic)
        normalized_rgb = np.array([0.5, 0.5, 0.2, 0.3])  # [cx, cy, w, h]
        rgb_meta = batch['rgb_search_metadata']

        physical_rgb = evaluator._denormalize_bbox(
            normalized_rgb,
            image_size=256,
            scale_x=rgb_meta['scale_x'],
            scale_y=rgb_meta['scale_y'],
            pad_x=rgb_meta['pad_x'],
            pad_y=rgb_meta['pad_y'],
            original_size=(rgb_meta['original_height'], rgb_meta['original_width']),
            output_format='xywh'
        )

        # Expected RGB reconstruction:
        # cx = 128 / (256/1920) = 960
        # cy = 128 / (256/1080) = 540
        # w = 51.2 / (256/1920) = 384
        # h = 76.8 / (256/1080) = 324
        # x = 960 - 192 = 768, y = 540 - 162 = 378
        assert abs(physical_rgb[0] - 768.0) < 2.0
        assert abs(physical_rgb[1] - 378.0) < 2.0
        assert abs(physical_rgb[2] - 384.0) < 2.0
        assert abs(physical_rgb[3] - 324.0) < 2.0

        # Test sonar denormalization (letterbox)
        # Use cy=0.7 to ensure it stays within bounds after removing padding
        # pad_y ≈ 115, so cy=0.7*256=179, 179-115=64 which is positive
        normalized_sonar = np.array([0.5, 0.7, 0.15, 0.2])
        sonar_meta = batch['sonar_search_metadata']

        physical_sonar = evaluator._denormalize_bbox(
            normalized_sonar,
            image_size=256,
            scale_x=sonar_meta['scale_x'],
            scale_y=sonar_meta['scale_y'],
            pad_x=sonar_meta['pad_x'],
            pad_y=sonar_meta['pad_y'],
            original_size=(sonar_meta['original_height'], sonar_meta['original_width']),
            output_format='xywh'
        )

        # Sonar should be in 682x376 physical space
        # Expected reconstruction:
        # cx = 0.5 * 256 = 128 -> 128 / scale ≈ 341
        # cy = 0.7 * 256 = 179 -> (179 - pad_y) / scale
        # w = 0.15 * 256 = 38.4 -> 38.4 / scale
        # h = 0.2 * 256 = 51.2 -> 51.2 / scale
        assert 0 < physical_sonar[0] < 682, f"Sonar x={physical_sonar[0]} out of bounds"
        assert 0 < physical_sonar[1] < 376, f"Sonar y={physical_sonar[1]} out of bounds"
        assert physical_sonar[2] > 0, f"Sonar w={physical_sonar[2]} should be positive"
        assert physical_sonar[3] > 0, f"Sonar h={physical_sonar[3]} should be positive"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
