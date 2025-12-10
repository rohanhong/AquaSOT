"""
Test Missing Target Detection and Handling

Tests for (0,0,0,0) bbox handling in zero/single/dual modality scenarios.

Author: AquaSOT-3D Team
Date: 2025-11-11
"""

import pytest
import torch
import numpy as np

from aquasot3d.models.tracker import build_tracker


class TestMissingTargetDetection:
    """Test detection of missing targets (0,0,0,0 format)"""

    def test_detect_missing_bbox(self):
        """Test detection of (0,0,0,0) bounding box"""
        # Missing bbox
        bbox_zero = np.array([0.0, 0.0, 0.0, 0.0])
        assert np.allclose(bbox_zero, 0.0)
        assert bbox_zero.sum() == 0.0

        # Valid bbox
        bbox_valid = np.array([100, 100, 50, 50])
        assert not np.allclose(bbox_valid, 0.0)

    def test_detect_missing_bbox_tensor(self):
        """Test detection with PyTorch tensors"""
        # Missing bbox
        bbox_zero = torch.tensor([0.0, 0.0, 0.0, 0.0])
        is_missing = torch.allclose(bbox_zero, torch.zeros_like(bbox_zero))
        assert is_missing

        # Valid bbox
        bbox_valid = torch.tensor([0.5, 0.5, 0.3, 0.3])
        is_missing = torch.allclose(bbox_valid, torch.zeros_like(bbox_valid))
        assert not is_missing


class TestZeroModalityHandling:
    """Test zero modality scenario (both RGB and Sonar missing)"""

    def test_zero_modality_forward(self):
        """Test tracker handles zero modality (both missing)"""
        config = {
            'rgb_backbone_config': {'type': 'vit_rgb', 'model_name': 'vit_tiny_patch16_224', 'pretrained': False, 'output_dim': 192},
            'sonar_backbone_config': {'type': 'sonar', 'encoder_type': 'conv', 'output_dim': 128},
            'fusion_config': {'type': 'dcma_lite', 'output_dim': 192},
            'use_temporal': False,
        }

        tracker = build_tracker('AquaSOT3DTracker', **config)
        tracker.eval()

        # Both modalities have (0,0,0,0) ground truth
        batch = {
            'rgb_template': torch.randn(1, 3, 128, 128),
            'rgb_search': torch.randn(1, 3, 256, 256),
            'sonar_template': torch.randn(1, 1, 376, 682),
            'sonar_search': torch.randn(1, 1, 376, 682),
            'rgb_boxes': torch.zeros(1, 4),  # Missing
            'sonar_boxes': torch.zeros(1, 4),  # Missing
        }

        with torch.no_grad():
            # Should still produce predictions even with missing targets
            outputs = tracker(
                batch['rgb_template'],
                batch['rgb_search'],
                batch['sonar_template'],
                batch['sonar_search']
            )

        # Verify predictions exist
        assert 'fusion_boxes' in outputs
        assert outputs['fusion_boxes'].shape == (1, 4)


class TestSingleModalityHandling:
    """Test single modality scenarios"""

    def test_rgb_only_modality(self):
        """Test when only RGB has valid target"""
        config = {
            'rgb_backbone_config': {'type': 'vit_rgb', 'model_name': 'vit_tiny_patch16_224', 'pretrained': False, 'output_dim': 192},
            'sonar_backbone_config': {'type': 'sonar', 'encoder_type': 'conv', 'output_dim': 128},
            'fusion_config': {'type': 'dcma_lite', 'output_dim': 192},
            'use_temporal': False,
        }

        tracker = build_tracker('AquaSOT3DTracker', **config)
        tracker.eval()

        batch = {
            'rgb_template': torch.randn(1, 3, 128, 128),
            'rgb_search': torch.randn(1, 3, 256, 256),
            'sonar_template': torch.randn(1, 1, 376, 682),
            'sonar_search': torch.randn(1, 1, 376, 682),
            'rgb_boxes': torch.tensor([[0.5, 0.5, 0.3, 0.3]]),  # Valid
            'sonar_boxes': torch.zeros(1, 4),  # Missing
        }

        with torch.no_grad():
            outputs = tracker(
                batch['rgb_template'],
                batch['rgb_search'],
                batch['sonar_template'],
                batch['sonar_search']
            )

        assert 'fusion_boxes' in outputs
        # Fusion should rely more on RGB since sonar is missing
        assert not torch.allclose(outputs['fusion_boxes'], torch.zeros(1, 4))

    def test_sonar_only_modality(self):
        """Test when only Sonar has valid target"""
        config = {
            'rgb_backbone_config': {'type': 'vit_rgb', 'model_name': 'vit_tiny_patch16_224', 'pretrained': False, 'output_dim': 192},
            'sonar_backbone_config': {'type': 'sonar', 'encoder_type': 'conv', 'output_dim': 128},
            'fusion_config': {'type': 'dcma_lite', 'output_dim': 192},
            'use_temporal': False,
        }

        tracker = build_tracker('AquaSOT3DTracker', **config)
        tracker.eval()

        batch = {
            'rgb_template': torch.randn(1, 3, 128, 128),
            'rgb_search': torch.randn(1, 3, 256, 256),
            'sonar_template': torch.randn(1, 1, 376, 682),
            'sonar_search': torch.randn(1, 1, 376, 682),
            'rgb_boxes': torch.zeros(1, 4),  # Missing
            'sonar_boxes': torch.tensor([[0.5, 0.5, 0.3, 0.3]]),  # Valid
        }

        with torch.no_grad():
            outputs = tracker(
                batch['rgb_template'],
                batch['rgb_search'],
                batch['sonar_template'],
                batch['sonar_search']
            )

        assert 'fusion_boxes' in outputs
        # Fusion should rely more on sonar since RGB is missing
        assert not torch.allclose(outputs['fusion_boxes'], torch.zeros(1, 4))


class TestDualModalityHandling:
    """Test dual modality (both RGB and Sonar have valid targets)"""

    def test_dual_modality_forward(self):
        """Test normal dual modality operation"""
        config = {
            'rgb_backbone_config': {'type': 'vit_rgb', 'model_name': 'vit_tiny_patch16_224', 'pretrained': False, 'output_dim': 192},
            'sonar_backbone_config': {'type': 'sonar', 'encoder_type': 'conv', 'output_dim': 128},
            'fusion_config': {'type': 'dcma_lite', 'output_dim': 192},
            'use_temporal': False,
        }

        tracker = build_tracker('AquaSOT3DTracker', **config)
        tracker.eval()

        batch = {
            'rgb_template': torch.randn(1, 3, 128, 128),
            'rgb_search': torch.randn(1, 3, 256, 256),
            'sonar_template': torch.randn(1, 1, 376, 682),
            'sonar_search': torch.randn(1, 1, 376, 682),
            'rgb_boxes': torch.tensor([[0.5, 0.5, 0.3, 0.3]]),  # Valid
            'sonar_boxes': torch.tensor([[0.5, 0.5, 0.3, 0.3]]),  # Valid
        }

        with torch.no_grad():
            outputs = tracker(
                batch['rgb_template'],
                batch['rgb_search'],
                batch['sonar_template'],
                batch['sonar_search']
            )

        assert 'fusion_boxes' in outputs
        assert 'rgb_boxes' in outputs
        assert 'sonar_boxes' in outputs


class TestSequenceIntegrity:
    """Test complete sequence processing with missing targets"""

    def test_sequence_with_intermittent_missing(self):
        """Test processing full sequence with some (0,0,0,0) frames"""
        config = {
            'rgb_backbone_config': {'type': 'vit_rgb', 'model_name': 'vit_tiny_patch16_224', 'pretrained': False, 'output_dim': 192},
            'sonar_backbone_config': {'type': 'sonar', 'encoder_type': 'conv', 'output_dim': 128},
            'fusion_config': {'type': 'dcma_lite', 'output_dim': 192},
            'use_temporal': False,
        }

        tracker = build_tracker('AquaSOT3DTracker', **config)
        tracker.eval()

        # Simulate sequence: [valid, missing, valid, missing, valid]
        sequence_length = 5
        gt_boxes = [
            torch.tensor([[0.5, 0.5, 0.3, 0.3]]),  # Valid
            torch.zeros(1, 4),                      # Missing
            torch.tensor([[0.6, 0.6, 0.3, 0.3]]),  # Valid
            torch.zeros(1, 4),                      # Missing
            torch.tensor([[0.7, 0.7, 0.3, 0.3]]),  # Valid
        ]

        predictions = []
        with torch.no_grad():
            for i in range(sequence_length):
                batch = {
                    'rgb_template': torch.randn(1, 3, 128, 128),
                    'rgb_search': torch.randn(1, 3, 256, 256),
                    'sonar_template': torch.randn(1, 1, 376, 682),
                    'sonar_search': torch.randn(1, 1, 376, 682),
                }

                outputs = tracker(
                    batch['rgb_template'],
                    batch['rgb_search'],
                    batch['sonar_template'],
                    batch['sonar_search']
                )
                predictions.append(outputs['fusion_boxes'])

        # All frames should have predictions
        assert len(predictions) == sequence_length
        for pred in predictions:
            assert pred.shape == (1, 4)
