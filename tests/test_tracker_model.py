"""
Unit Tests for AquaSOT-3D Tracker Model

Tests for the complete tracking system including:
- Tracker initialization with different configurations
- Forward pass logic
- Multi-task output organization
- Loss computation
- Temporal state management

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import pytest
import torch
import torch.nn as nn

from aquasot3d.models.tracker import AquaSOT3DTracker, build_tracker


class TestAquaSOT3DTracker:
    """Tests for AquaSOT3DTracker module."""

    @pytest.fixture
    def minimal_config(self):
        """Minimal configuration for testing."""
        return {
            'rgb_backbone_config': {
                'type': 'vit_rgb',
                'model_name': 'vit_tiny_patch16_224',
                'pretrained': False,
                'output_dim': 192
            },
            'sonar_backbone_config': {
                'type': 'sonar',
                'encoder_type': 'conv',
                'output_dim': 128
            },
            'fusion_config': {
                'type': 'g2_dcma_lite',
                'output_dim': 192,
                'num_heads': 4,
                'num_points': 2
            },
            'temporal_config': None,  # Disable temporal for minimal test
            'use_temporal': False,
            'use_weak3d': False
        }

    @pytest.fixture
    def full_config(self):
        """Full configuration with all features enabled."""
        return {
            'rgb_backbone_config': {
                'type': 'vit_rgb',
                'model_name': 'vit_tiny_patch16_224',
                'pretrained': False,
                'output_dim': 192
            },
            'sonar_backbone_config': {
                'type': 'sonar',
                'encoder_type': 'conv',
                'output_dim': 128
            },
            'fusion_config': {
                'type': 'g2_dcma_lite',
                'output_dim': 192,
                'num_heads': 4,
                'num_points': 2
            },
            'temporal_config': {
                'type': 'transformer',
                'num_layers': 2,
                'num_heads': 4
            },
            'weak3d_config': {
                'hidden_dim': 128,
                'num_radial_bins': 50
            },
            'use_temporal': True,
            'use_weak3d': True
        }

    def test_minimal_tracker_initialization(self, minimal_config):
        """Test tracker initialization with minimal configuration."""
        tracker = AquaSOT3DTracker(**minimal_config)

        assert isinstance(tracker, AquaSOT3DTracker)
        assert tracker.rgb_feat_dim == 192
        assert tracker.sonar_feat_dim == 128
        assert not tracker.use_temporal
        assert not tracker.use_weak3d
        assert tracker.temporal_head is None
        assert tracker.weak3d_head is None

    def test_full_tracker_initialization(self, full_config):
        """Test tracker initialization with all features."""
        tracker = AquaSOT3DTracker(**full_config)

        assert isinstance(tracker, AquaSOT3DTracker)
        assert tracker.use_temporal
        assert tracker.use_weak3d
        assert tracker.temporal_head is not None
        assert tracker.weak3d_head is not None
        assert tracker.rgb_projector is not None
        assert tracker.sonar_projector is not None

    def test_minimal_forward_pass(self, minimal_config):
        """Test forward pass with minimal configuration."""
        tracker = AquaSOT3DTracker(**minimal_config)
        tracker.eval()

        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        with torch.no_grad():
            outputs = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                return_losses=False
            )

        # Check essential outputs
        assert 'fusion_boxes' in outputs
        assert 'rgb_boxes' in outputs
        assert 'sonar_boxes' in outputs
        assert outputs['fusion_boxes'].shape == (batch_size, 4)
        assert outputs['rgb_boxes'].shape == (batch_size, 4)
        assert outputs['sonar_boxes'].shape == (batch_size, 4)

        # Check quality scores
        assert 'fusion_quality' in outputs
        assert outputs['fusion_quality'].shape == (batch_size,)

        # 3D center should not be present
        assert 'center_3d' not in outputs

    def test_full_forward_pass(self, full_config):
        """Test forward pass with all features enabled."""
        tracker = AquaSOT3DTracker(**full_config)
        tracker.eval()

        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        with torch.no_grad():
            outputs = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                return_losses=False
            )

        # Check all outputs including 3D
        assert 'fusion_boxes' in outputs
        assert 'rgb_boxes' in outputs
        assert 'sonar_boxes' in outputs
        assert 'center_3d' in outputs

        assert outputs['fusion_boxes'].shape == (batch_size, 4)
        assert outputs['center_3d'].shape == (batch_size, 3)

    def test_forward_with_losses(self, full_config):
        """Test forward pass with loss computation."""
        tracker = AquaSOT3DTracker(**full_config)
        tracker.train()

        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        # Ground truth boxes
        gt_boxes = {
            'rgb': torch.randn(batch_size, 4),
            'sonar': torch.randn(batch_size, 4),
            'fusion': torch.randn(batch_size, 4)
        }
        # Ensure positive width/height
        gt_boxes['rgb'][:, 2:] = gt_boxes['rgb'][:, 2:].abs() + 0.1
        gt_boxes['sonar'][:, 2:] = gt_boxes['sonar'][:, 2:].abs() + 0.1
        gt_boxes['fusion'][:, 2:] = gt_boxes['fusion'][:, 2:].abs() + 0.1

        outputs = tracker(
            rgb_template, rgb_search,
            sonar_template, sonar_search,
            gt_boxes=gt_boxes,
            return_losses=True
        )

        # Check losses are present
        assert 'losses' in outputs
        losses = outputs['losses']

        # 2D box losses
        assert 'rgb_box_loss' in losses
        assert 'sonar_box_loss' in losses
        assert 'fusion_box_loss' in losses

        # Consistency losses
        assert 'component_consistency' in losses
        assert 'representation_consistency' in losses

        # 3D losses (if weak3d is enabled)
        if tracker.use_weak3d:
            assert 'X_consistency_loss' in losses
            assert 'rgb_cyclic_loss' in losses
            assert 'sonar_cyclic_loss' in losses

        # All losses should be scalars
        for loss_name, loss_value in losses.items():
            assert loss_value.numel() == 1
            assert torch.isfinite(loss_value)

    def test_forward_with_intermediate_features(self, minimal_config):
        """Test forward pass returning intermediate features."""
        tracker = AquaSOT3DTracker(**minimal_config)
        tracker.eval()

        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        with torch.no_grad():
            outputs = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                return_losses=False,
                return_intermediate=True
            )

        # Check intermediate features
        assert 'intermediate' in outputs
        intermediate = outputs['intermediate']

        assert 'rgb_features' in intermediate
        assert 'sonar_features' in intermediate
        assert 'fused_features' in intermediate

        # Features can be either tensors or lists (multi-scale)
        # Extract first element if list
        rgb_feat = intermediate['rgb_features']
        if isinstance(rgb_feat, list):
            rgb_feat = rgb_feat[0]
        sonar_feat = intermediate['sonar_features']
        if isinstance(sonar_feat, list):
            sonar_feat = sonar_feat[0]
        fused_feat = intermediate['fused_features']
        if isinstance(fused_feat, list):
            fused_feat = fused_feat[0]

        # Features can be either 3D (B, N, C) for ViT or 4D (B, C, H, W) for Conv
        assert rgb_feat.dim() in [3, 4]
        assert sonar_feat.dim() in [3, 4]
        assert fused_feat.dim() in [3, 4]

        # Verify batch size dimension
        assert rgb_feat.shape[0] == batch_size
        assert sonar_feat.shape[0] == batch_size
        assert fused_feat.shape[0] == batch_size

    def test_temporal_state_management(self, full_config):
        """Test temporal state reset and update."""
        tracker = AquaSOT3DTracker(**full_config)

        # Initial state should be None
        assert tracker.temporal_state is None

        # After forward pass, state might be updated
        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        tracker.eval()
        with torch.no_grad():
            outputs = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search
            )

        # Reset state
        tracker.reset_temporal_state()
        assert tracker.temporal_state is None

    def test_gradient_flow(self, full_config):
        """Test gradient flow through tracker."""
        tracker = AquaSOT3DTracker(**full_config)
        tracker.train()

        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        gt_boxes = {
            'rgb': torch.randn(batch_size, 4),
            'sonar': torch.randn(batch_size, 4),
            'fusion': torch.randn(batch_size, 4)
        }
        gt_boxes['rgb'][:, 2:] = gt_boxes['rgb'][:, 2:].abs() + 0.1
        gt_boxes['sonar'][:, 2:] = gt_boxes['sonar'][:, 2:].abs() + 0.1
        gt_boxes['fusion'][:, 2:] = gt_boxes['fusion'][:, 2:].abs() + 0.1

        outputs = tracker(
            rgb_template, rgb_search,
            sonar_template, sonar_search,
            gt_boxes=gt_boxes,
            return_losses=True
        )

        # Compute total loss
        total_loss = sum(outputs['losses'].values())

        # Backward pass
        total_loss.backward()

        # Check gradients exist
        has_grad = False
        for name, param in tracker.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

        assert has_grad, "No gradients computed"

    def test_build_tracker_factory(self, minimal_config):
        """Test build_tracker factory function."""
        tracker = build_tracker('AquaSOT3DTracker', **minimal_config)

        assert isinstance(tracker, AquaSOT3DTracker)

    def test_different_temporal_types(self):
        """Test tracker with different temporal head types."""
        base_config = {
            'rgb_backbone_config': {
                'type': 'vit_rgb',
                'model_name': 'vit_tiny_patch16_224',
                'pretrained': False,
                'output_dim': 192
            },
            'sonar_backbone_config': {
                'type': 'sonar',
                'encoder_type': 'conv',
                'output_dim': 128
            },
            'fusion_config': {
                'type': 'g2_dcma_lite',
                'output_dim': 192
            },
            'use_temporal': True,
            'use_weak3d': False
        }

        temporal_types = ['transformer', 'artrack']  # Skip mamba (requires CUDA)

        for temp_type in temporal_types:
            config = base_config.copy()
            config['temporal_config'] = {
                'type': temp_type,
                'num_layers': 2,
                'num_heads': 4
            }

            tracker = AquaSOT3DTracker(**config)
            assert tracker.temporal_head is not None

    def test_output_consistency(self, minimal_config):
        """Test output consistency across multiple calls."""
        tracker = AquaSOT3DTracker(**minimal_config)
        tracker.eval()

        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        with torch.no_grad():
            outputs1 = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search
            )
            outputs2 = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search
            )

        # Outputs should be identical (deterministic in eval mode)
        assert torch.allclose(outputs1['fusion_boxes'], outputs2['fusion_boxes'])
        assert torch.allclose(outputs1['rgb_boxes'], outputs2['rgb_boxes'])
        assert torch.allclose(outputs1['sonar_boxes'], outputs2['sonar_boxes'])

    def test_batch_size_variations(self, minimal_config):
        """Test tracker with different batch sizes."""
        tracker = AquaSOT3DTracker(**minimal_config)
        tracker.eval()

        for batch_size in [1, 2, 4]:
            rgb_template = torch.randn(batch_size, 3, 128, 128)
            rgb_search = torch.randn(batch_size, 3, 256, 256)
            sonar_template = torch.randn(batch_size, 1, 376, 682)  # Correct: H=376, W=682
            sonar_search = torch.randn(batch_size, 1, 376, 682)    # Correct: H=376, W=682

            with torch.no_grad():
                outputs = tracker(
                    rgb_template, rgb_search,
                    sonar_template, sonar_search
                )

            assert outputs['fusion_boxes'].shape[0] == batch_size
            assert outputs['rgb_boxes'].shape[0] == batch_size
            assert outputs['sonar_boxes'].shape[0] == batch_size

    def test_component_integration(self, full_config):
        """Test that all components are properly integrated."""
        tracker = AquaSOT3DTracker(**full_config)

        # Check backbone integration
        assert hasattr(tracker, 'backbone')
        assert tracker.backbone is not None

        # Check fusion integration
        assert hasattr(tracker, 'fusion')
        assert tracker.fusion is not None

        # Check temporal integration
        assert hasattr(tracker, 'temporal_head')
        if tracker.use_temporal:
            assert tracker.temporal_head is not None

        # Check heads integration
        assert hasattr(tracker, 'rgb_box_head')
        assert hasattr(tracker, 'sonar_box_head')
        assert hasattr(tracker, 'fusion_box_head')
        assert hasattr(tracker, 'fusion_head')

        # Check 3D components
        if tracker.use_weak3d:
            assert hasattr(tracker, 'weak3d_head')
            assert hasattr(tracker, 'rgb_projector')
            assert hasattr(tracker, 'sonar_projector')

    def test_disable_weak3d(self):
        """Test tracker with weak 3D estimation disabled."""
        config = {
            'rgb_backbone_config': {
                'type': 'vit_rgb',
                'model_name': 'vit_tiny_patch16_224',
                'pretrained': False,
                'output_dim': 192
            },
            'sonar_backbone_config': {
                'type': 'sonar',
                'encoder_type': 'conv',
                'output_dim': 128
            },
            'fusion_config': {
                'type': 'g2_dcma_lite',
                'output_dim': 192
            },
            'use_temporal': False,
            'use_weak3d': False
        }

        tracker = AquaSOT3DTracker(**config)

        assert not tracker.use_weak3d
        assert tracker.weak3d_head is None
        assert tracker.rgb_projector is None
        assert tracker.sonar_projector is None

        # Forward pass should not produce 3D outputs
        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        tracker.eval()
        with torch.no_grad():
            outputs = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search
            )

        assert 'center_3d' not in outputs

    def test_quality_scores_output(self, minimal_config):
        """Test quality score predictions."""
        tracker = AquaSOT3DTracker(**minimal_config)
        tracker.eval()

        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        with torch.no_grad():
            outputs = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search
            )

        # Check quality scores
        assert 'fusion_quality' in outputs
        assert 'rgb_quality' in outputs
        assert 'sonar_quality' in outputs

        # Quality scores should be in valid range [0, 1]
        assert torch.all(outputs['fusion_quality'] >= 0)
        assert torch.all(outputs['fusion_quality'] <= 1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
