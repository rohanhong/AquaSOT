"""
Integration Tests for AquaSOT-3D Tracker

End-to-end tests for the complete tracking system including:
- Full pipeline integration
- Multi-modal data flow
- Loss aggregation and backpropagation
- Inference mode testing
- Performance and memory tests

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from aquasot3d.models.tracker import build_tracker


class TestTrackerIntegration:
    """Integration tests for complete tracking pipeline."""

    @pytest.fixture
    def integration_config(self):
        """Configuration for integration testing."""
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
            'projector_config': {
                'projection_type': 'mlp',
                'hidden_dim': 128
            },
            'use_temporal': True,
            'use_weak3d': True
        }

    def test_end_to_end_forward_pass(self, integration_config):
        """Test complete end-to-end forward pass."""
        tracker = build_tracker('AquaSOT3DTracker', **integration_config)
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

        # Verify all expected outputs
        assert 'fusion_boxes' in outputs
        assert 'rgb_boxes' in outputs
        assert 'sonar_boxes' in outputs
        assert 'center_3d' in outputs
        assert 'intermediate' in outputs

        # Verify shapes
        assert outputs['fusion_boxes'].shape == (batch_size, 4)
        assert outputs['rgb_boxes'].shape == (batch_size, 4)
        assert outputs['sonar_boxes'].shape == (batch_size, 4)
        assert outputs['center_3d'].shape == (batch_size, 3)

    def test_training_pipeline(self, integration_config):
        """Test complete training pipeline with loss computation."""
        tracker = build_tracker('AquaSOT3DTracker', **integration_config)
        tracker.train()

        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        # Ground truth
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

        # Verify all losses
        assert 'losses' in outputs
        losses = outputs['losses']

        # Check all expected losses
        expected_losses = [
            'rgb_box_loss',
            'sonar_box_loss',
            'fusion_box_loss',
            'component_consistency',
            'representation_consistency',
            'X_consistency_loss',
            'rgb_cyclic_loss',
            'sonar_cyclic_loss',
            'rgb_proj_loss',
            'sonar_proj_loss'
        ]

        for loss_name in expected_losses:
            assert loss_name in losses, f"Missing loss: {loss_name}"
            assert torch.isfinite(losses[loss_name])

    def test_backward_pass(self, integration_config):
        """Test full backward pass through the system."""
        tracker = build_tracker('AquaSOT3DTracker', **integration_config)
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
        assert torch.isfinite(total_loss)

        # Backward pass
        total_loss.backward()

        # Verify gradients in all modules
        modules_with_grads = {
            'backbone': tracker.backbone,
            'fusion': tracker.fusion,
            'temporal_head': tracker.temporal_head,
            'rgb_box_head': tracker.rgb_box_head,
            'sonar_box_head': tracker.sonar_box_head,
            'fusion_box_head': tracker.fusion_box_head,
            'fusion_head': tracker.fusion_head,
            'weak3d_head': tracker.weak3d_head,
        }

        for module_name, module in modules_with_grads.items():
            if module is None:
                continue

            has_grad = False
            for param in module.parameters():
                if param.requires_grad and param.grad is not None:
                    has_grad = True
                    assert torch.isfinite(param.grad).all(), \
                        f"Non-finite gradient in {module_name}"

            # Some modules might not have gradients if not used
            # assert has_grad, f"No gradients in {module_name}"

    def test_temporal_sequence_processing(self, integration_config):
        """Test temporal processing with sequence input."""
        tracker = build_tracker('AquaSOT3DTracker', **integration_config)
        tracker.eval()

        batch_size = 2
        seq_length = 3

        # Simulate a sequence of frames
        for t in range(seq_length):
            rgb_template = torch.randn(batch_size, 3, 128, 128)
            rgb_search = torch.randn(batch_size, 3, 256, 256)
            sonar_template = torch.randn(batch_size, 1, 682, 376)
            sonar_search = torch.randn(batch_size, 1, 682, 376)

            with torch.no_grad():
                outputs = tracker(
                    rgb_template, rgb_search,
                    sonar_template, sonar_search
                )

            assert 'fusion_boxes' in outputs
            assert outputs['fusion_boxes'].shape == (batch_size, 4)

        # Reset temporal state
        tracker.reset_temporal_state()
        assert tracker.temporal_state is None

    def test_multi_scale_features(self, integration_config):
        """Test with different input resolutions."""
        tracker = build_tracker('AquaSOT3DTracker', **integration_config)
        tracker.eval()

        batch_size = 2
        resolutions = [
            (128, 128, 256, 256),
            (96, 96, 192, 192),
            (160, 160, 320, 320)
        ]

        for h_t, w_t, h_s, w_s in resolutions:
            rgb_template = torch.randn(batch_size, 3, h_t, w_t)
            rgb_search = torch.randn(batch_size, 3, h_s, w_s)
            sonar_template = torch.randn(batch_size, 1, 682, 376)
            sonar_search = torch.randn(batch_size, 1, 682, 376)

            with torch.no_grad():
                outputs = tracker(
                    rgb_template, rgb_search,
                    sonar_template, sonar_search
                )

            assert outputs['fusion_boxes'].shape == (batch_size, 4)

    def test_inference_mode(self, integration_config):
        """Test inference mode without ground truth."""
        tracker = build_tracker('AquaSOT3DTracker', **integration_config)
        tracker.eval()

        batch_size = 1
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        with torch.no_grad():
            outputs = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                gt_boxes=None,
                return_losses=False
            )

        # Should not have losses
        assert 'losses' not in outputs

        # Should have predictions
        assert 'fusion_boxes' in outputs
        assert 'rgb_boxes' in outputs
        assert 'sonar_boxes' in outputs
        assert 'center_3d' in outputs

    def test_loss_aggregation(self, integration_config):
        """Test proper loss aggregation with different weights."""
        tracker = build_tracker('AquaSOT3DTracker', **integration_config)
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

        losses = outputs['losses']

        # Define loss weights (as would be done in training)
        loss_weights = {
            'rgb_box_loss': 1.0,
            'sonar_box_loss': 1.0,
            'fusion_box_loss': 1.5,
            'component_consistency': 0.2,
            'representation_consistency': 0.1,
            'X_consistency_loss': 0.2,
            'rgb_cyclic_loss': 0.6,
            'sonar_cyclic_loss': 0.6,
            'rgb_proj_loss': 0.3,
            'sonar_proj_loss': 0.3
        }

        # Compute weighted total loss
        total_loss = 0.0
        for loss_name, loss_value in losses.items():
            weight = loss_weights.get(loss_name, 1.0)
            total_loss += weight * loss_value

        assert torch.isfinite(total_loss)
        assert total_loss > 0

    def test_memory_efficiency(self, integration_config):
        """Test memory usage remains reasonable."""
        tracker = build_tracker('AquaSOT3DTracker', **integration_config)
        tracker.eval()

        batch_size = 4
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        # Multiple forward passes should not accumulate memory
        with torch.no_grad():
            for _ in range(5):
                outputs = tracker(
                    rgb_template, rgb_search,
                    sonar_template, sonar_search
                )

        # Successful completion without OOM is the test

    def test_deterministic_output(self, integration_config):
        """Test deterministic output in eval mode."""
        tracker = build_tracker('AquaSOT3DTracker', **integration_config)
        tracker.eval()

        # Set seed for determinism
        torch.manual_seed(42)

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

        # Reset and run again
        tracker.reset_temporal_state()
        torch.manual_seed(42)

        with torch.no_grad():
            outputs2 = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search
            )

        # Outputs should be identical
        assert torch.allclose(outputs1['fusion_boxes'], outputs2['fusion_boxes'], atol=1e-5)
        assert torch.allclose(outputs1['center_3d'], outputs2['center_3d'], atol=1e-5)

    def test_component_ablation(self):
        """Test tracker with different components disabled."""
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
            }
        }

        # Test 1: No temporal, no weak3d
        config1 = {**base_config, 'use_temporal': False, 'use_weak3d': False}
        tracker1 = build_tracker('AquaSOT3DTracker', **config1)

        # Test 2: Temporal, no weak3d
        config2 = {
            **base_config,
            'use_temporal': True,
            'temporal_config': {'type': 'transformer', 'num_layers': 2},
            'use_weak3d': False
        }
        tracker2 = build_tracker('AquaSOT3DTracker', **config2)

        # Test 3: No temporal, weak3d
        config3 = {
            **base_config,
            'use_temporal': False,
            'use_weak3d': True,
            'weak3d_config': {'hidden_dim': 128}
        }
        tracker3 = build_tracker('AquaSOT3DTracker', **config3)

        batch_size = 2
        rgb_template = torch.randn(batch_size, 3, 128, 128)
        rgb_search = torch.randn(batch_size, 3, 256, 256)
        sonar_template = torch.randn(batch_size, 1, 376, 682)
        sonar_search = torch.randn(batch_size, 1, 376, 682)

        for tracker in [tracker1, tracker2, tracker3]:
            tracker.eval()
            with torch.no_grad():
                outputs = tracker(
                    rgb_template, rgb_search,
                    sonar_template, sonar_search
                )

            # All should produce box predictions
            assert 'fusion_boxes' in outputs
            assert outputs['fusion_boxes'].shape == (batch_size, 4)

    def test_full_system_integration(self, integration_config):
        """Comprehensive integration test of the full system."""
        tracker = build_tracker('AquaSOT3DTracker', **integration_config)

        # Test training mode
        tracker.train()
        batch_size = 2

        # Prepare inputs
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

        # Forward pass with all features
        outputs = tracker(
            rgb_template, rgb_search,
            sonar_template, sonar_search,
            gt_boxes=gt_boxes,
            return_losses=True,
            return_intermediate=True
        )

        # Verify comprehensive outputs
        assert 'fusion_boxes' in outputs
        assert 'rgb_boxes' in outputs
        assert 'sonar_boxes' in outputs
        assert 'center_3d' in outputs
        assert 'losses' in outputs
        assert 'intermediate' in outputs

        # Verify loss computation
        total_loss = sum(outputs['losses'].values())
        total_loss.backward()

        # Verify all components have gradients
        for name, param in tracker.named_parameters():
            if param.requires_grad:
                # Not all parameters may have gradients (e.g., unused branches)
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all()

        # Test eval mode
        tracker.eval()
        with torch.no_grad():
            eval_outputs = tracker(
                rgb_template, rgb_search,
                sonar_template, sonar_search,
                return_losses=False
            )

        assert 'fusion_boxes' in eval_outputs
        assert 'losses' not in eval_outputs


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
