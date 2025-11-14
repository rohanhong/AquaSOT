"""
Integration Tests for Evaluation Metadata Propagation (Audit 2025111409 Finding 1)

This test suite validates that scale/padding metadata flows correctly from
the evaluation dataset through to the evaluator's coordinate reconstruction.

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from aquasot3d.dataloader.datasets.evaluation_dataset import EvaluationSequenceDataset
from aquasot3d.dataloader.datasets.rgb_sonar_dataset import RGBSonarDataset


class TestEvaluationDatasetMetadata:
    """Test metadata propagation in evaluation dataset."""

    @pytest.fixture
    def mock_base_dataset(self):
        """Create mock base dataset with known dimensions."""
        dataset = MagicMock(spec=RGBSonarDataset)
        dataset.num_frames = 10
        dataset.valid_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # Mock sequence info
        dataset.get_sequence_info.return_value = {
            'sequence_name': 'test_seq',
            'num_frames': 10,
        }

        # Mock frame data with known dimensions
        def mock_get_frame_data(index):
            # RGB: 1920x1080, Sonar: 682x376
            rgb_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            sonar_img = np.random.randint(0, 255, (376, 682), dtype=np.uint8)

            return {
                'frame_id': index + 1,
                'rgb_img': rgb_img,
                'sonar_img': sonar_img,
                'rgb_bbox': np.array([100.0, 200.0, 50.0, 60.0]),  # [x, y, w, h]
                'sonar_bbox': np.array([50.0, 100.0, 30.0, 40.0]),
                'is_valid': True,
                'is_rgb_valid': True,
                'is_sonar_valid': True,
            }

        dataset._get_frame_data = mock_get_frame_data
        return dataset

    def test_prepare_image_returns_metadata(self, mock_base_dataset):
        """
        Test that _prepare_image returns transformation metadata.

        This is the core fix: _prepare_image must return both the tensor
        and metadata dict with scale/padding information.
        """
        eval_ds = EvaluationSequenceDataset(
            base_dataset=mock_base_dataset,
            template_size=128,
            search_size=256,
            template_frame='first'
        )

        # Get a sample
        sample = eval_ds[0]

        # Verify metadata keys exist
        assert 'rgb_search_metadata' in sample, "Missing RGB search metadata"
        assert 'sonar_search_metadata' in sample, "Missing sonar search metadata"
        assert 'template_rgb_metadata' in sample, "Missing template RGB metadata"
        assert 'template_sonar_metadata' in sample, "Missing template sonar metadata"

        # Verify RGB metadata structure (anisotropic resize)
        rgb_meta = sample['rgb_search_metadata']
        assert 'original_height' in rgb_meta
        assert 'original_width' in rgb_meta
        assert 'scale_x' in rgb_meta
        assert 'scale_y' in rgb_meta
        assert 'pad_x' in rgb_meta
        assert 'pad_y' in rgb_meta

        # Verify RGB dimensions
        assert rgb_meta['original_height'] == 1080
        assert rgb_meta['original_width'] == 1920

        # RGB uses direct resize (anisotropic)
        # scale_x = 256 / 1920 ≈ 0.133
        # scale_y = 256 / 1080 ≈ 0.237
        assert abs(rgb_meta['scale_x'] - (256.0 / 1920.0)) < 1e-5
        assert abs(rgb_meta['scale_y'] - (256.0 / 1080.0)) < 1e-5
        assert rgb_meta['pad_x'] == 0
        assert rgb_meta['pad_y'] == 0

        # Verify sonar metadata (letterbox resize - isotropic)
        sonar_meta = sample['sonar_search_metadata']
        assert sonar_meta['original_height'] == 376
        assert sonar_meta['original_width'] == 682

        # Sonar uses letterbox (isotropic scaling)
        # aspect ratio 682/376 ≈ 1.814, target 256x256
        # scale should be min(256/682, 256/376) = 256/682 ≈ 0.375
        expected_scale = 256.0 / 682.0
        assert abs(sonar_meta['scale_x'] - expected_scale) < 1e-3
        assert abs(sonar_meta['scale_y'] - expected_scale) < 1e-3

        # Padding should be non-zero for sonar (letterbox)
        # new_H = 376 * 0.375 ≈ 141, so pad_y = 256 - 141 = 115
        new_h = int(376 * expected_scale)
        expected_pad_y = 256 - new_h
        assert sonar_meta['pad_y'] == expected_pad_y
        assert sonar_meta['pad_x'] == 0

    def test_metadata_matches_image_dimensions(self, mock_base_dataset):
        """
        Test that metadata dimensions match actual processed images.
        """
        eval_ds = EvaluationSequenceDataset(
            base_dataset=mock_base_dataset,
            template_size=128,
            search_size=256,
            template_frame='first'
        )

        sample = eval_ds[0]

        # Verify processed image shapes match target sizes
        assert sample['rgb_search'].shape == (3, 256, 256)
        assert sample['sonar_search'].shape == (1, 256, 256)
        assert sample['rgb_template'].shape == (3, 128, 128)
        assert sample['sonar_template'].shape == (1, 128, 128)

    def test_none_image_returns_dummy_metadata(self, mock_base_dataset):
        """
        Test that None images (occluded frames) return dummy metadata.
        """
        # Override mock to return None images
        def mock_get_frame_data_occluded(index):
            return {
                'frame_id': index + 1,
                'rgb_img': None,
                'sonar_img': None,
                'rgb_bbox': np.zeros(4),
                'sonar_bbox': np.zeros(4),
                'is_valid': False,
                'is_rgb_valid': False,
                'is_sonar_valid': False,
            }

        mock_base_dataset._get_frame_data = mock_get_frame_data_occluded

        eval_ds = EvaluationSequenceDataset(
            base_dataset=mock_base_dataset,
            template_size=128,
            search_size=256,
            template_frame='first'
        )

        sample = eval_ds[0]

        # Verify dummy metadata exists
        assert sample['rgb_search_metadata']['scale_x'] == 1.0
        assert sample['rgb_search_metadata']['scale_y'] == 1.0
        assert sample['rgb_search_metadata']['pad_x'] == 0
        assert sample['rgb_search_metadata']['pad_y'] == 0


class TestCoordinateReconstruction:
    """Test physical coordinate reconstruction using metadata."""

    def test_denormalize_with_rgb_anisotropic_scaling(self):
        """
        Test coordinate reconstruction for RGB with anisotropic scaling.

        RGB images are directly resized, which can introduce different
        scale factors along X and Y axes.
        """
        from aquasot3d.evaluation.evaluator import TrackingEvaluator

        # Create mock model and dataloader
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_dataloader = MagicMock()

        evaluator = TrackingEvaluator(
            model=mock_model,
            dataloader=mock_dataloader,
            device='cpu',
            search_size=256
        )

        # Simulate normalized bbox prediction [cx, cy, w, h] in [0, 1]
        normalized_bbox = np.array([0.5, 0.5, 0.2, 0.3])

        # RGB metadata: 1920x1080 -> 256x256
        scale_x = 256.0 / 1920.0
        scale_y = 256.0 / 1080.0

        # Denormalize
        physical_bbox = evaluator._denormalize_bbox(
            normalized_bbox,
            image_size=256,
            scale_x=scale_x,
            scale_y=scale_y,
            pad_x=0,
            pad_y=0,
            original_size=(1080, 1920)
        )

        # Expected reconstruction:
        # Step 1: Denormalize to 256x256 space
        # cx = 0.5 * 256 = 128
        # cy = 0.5 * 256 = 128
        # w = 0.2 * 256 = 51.2
        # h = 0.3 * 256 = 76.8

        # Step 2: Remove padding (none for RGB)
        # (no change)

        # Step 3: Scale back to original resolution
        # cx_orig = 128 / scale_x = 128 / (256/1920) = 960
        # cy_orig = 128 / scale_y = 128 / (256/1080) = 540
        # w_orig = 51.2 / scale_x = 51.2 / (256/1920) = 384
        # h_orig = 76.8 / scale_y = 76.8 / (256/1080) = 324

        # Step 4: Convert to [x, y, w, h] format (default output_format='xywh')
        # x = cx - w/2 = 960 - 384/2 = 768
        # y = cy - h/2 = 540 - 324/2 = 378
        # w = 384
        # h = 324

        expected_x = 768.0
        expected_y = 378.0
        expected_w = 384.0
        expected_h = 324.0

        assert abs(physical_bbox[0] - expected_x) < 1.0
        assert abs(physical_bbox[1] - expected_y) < 1.0
        assert abs(physical_bbox[2] - expected_w) < 1.0
        assert abs(physical_bbox[3] - expected_h) < 1.0

    def test_denormalize_with_sonar_letterbox_padding(self):
        """
        Test coordinate reconstruction for sonar with letterbox padding.

        Sonar images use isotropic scaling + top padding to preserve
        aspect ratio and maintain sensor origin at bottom-center.
        """
        from aquasot3d.evaluation.evaluator import TrackingEvaluator

        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_dataloader = MagicMock()

        evaluator = TrackingEvaluator(
            model=mock_model,
            dataloader=mock_dataloader,
            device='cpu',
            search_size=256
        )

        # Simulate normalized bbox prediction
        normalized_bbox = np.array([0.5, 0.5, 0.15, 0.2])

        # Sonar metadata: 682x376 -> 256x256 with letterbox
        # scale = 256/682 ≈ 0.375
        # new_h = 376 * 0.375 ≈ 141
        # pad_y = 256 - 141 = 115
        scale = 256.0 / 682.0
        new_h = int(376 * scale)
        pad_y = 256 - new_h

        physical_bbox = evaluator._denormalize_bbox(
            normalized_bbox,
            image_size=256,
            scale_x=scale,
            scale_y=scale,
            pad_x=0,
            pad_y=pad_y,
            original_size=(376, 682)
        )

        # Expected reconstruction:
        # Step 1: Denormalize to 256x256 space
        # cx = 0.5 * 256 = 128
        # cy = 0.5 * 256 = 128
        # w = 0.15 * 256 = 38.4
        # h = 0.2 * 256 = 51.2

        # Step 2: Remove padding
        # cy_scaled = 128 - pad_y = 128 - 115 = 13

        # Step 3: Scale back (isotropic)
        # cx_orig = 128 / scale ≈ 341
        # cy_orig = 13 / scale ≈ 35
        # w_orig = 38.4 / scale ≈ 102
        # h_orig = 51.2 / scale ≈ 137

        # Step 4: Convert to [x, y, w, h] format
        # x = cx - w/2
        # y = cy - h/2
        cx_orig = 128.0 / scale
        cy_orig = (128.0 - pad_y) / scale
        w_orig = 38.4 / scale
        h_orig = 51.2 / scale

        expected_x = cx_orig - w_orig / 2
        expected_y = cy_orig - h_orig / 2
        expected_w = w_orig
        expected_h = h_orig

        assert abs(physical_bbox[0] - expected_x) < 2.0
        assert abs(physical_bbox[1] - expected_y) < 2.0
        assert abs(physical_bbox[2] - expected_w) < 2.0
        assert abs(physical_bbox[3] - expected_h) < 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
