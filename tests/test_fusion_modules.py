"""
Comprehensive Test Suite for Fusion Modules

Tests all components of the DCMA fusion framework:
- Deformable Attention (deformable_attention.py)
- Consistency Loss (consistency_loss.py)
- DCMA Integration (dcma.py)

Run with: pytest tests/test_fusion_modules.py -v
"""

import pytest
import torch
import torch.nn as nn

from aquasot3d.models.fusion import (
    # Deformable Attention
    DeformableOffsetNetwork,
    BilinearSampler,
    DeformableCrossAttention,
    MultiScaleDeformableAttention,
    build_deformable_cross_attention,
    build_multi_scale_deformable_attention,
    # Consistency Loss
    ComponentConsistencyLoss,
    InfoNCELoss,
    DualConsistencyLoss,
    ProjectionHead,
    build_dual_consistency_loss,
    build_projection_head,
    # DCMA
    DCMA,
    DCMALite,
    build_dcma,
    build_dcma_lite
)


# ============================================================================
# Test Deformable Attention Components
# ============================================================================

class TestDeformableOffsetNetwork:
    """Test DeformableOffsetNetwork."""

    def test_initialization(self):
        """Test module initialization."""
        offset_net = DeformableOffsetNetwork(
            embed_dim=256,
            num_heads=8,
            num_points=4
        )
        assert offset_net.num_heads == 8
        assert offset_net.num_points == 4

    def test_forward_shape(self):
        """Test output shape correctness."""
        B, N, C = 2, 100, 256
        query_features = torch.randn(B, N, C)

        offset_net = DeformableOffsetNetwork(embed_dim=C, num_heads=8, num_points=4)
        offsets = offset_net(query_features)

        assert offsets.shape == (B, N, 8, 4, 2)
        # Offsets should be in [-1, 1] due to tanh
        assert (offsets >= -1).all() and (offsets <= 1).all()


class TestBilinearSampler:
    """Test BilinearSampler."""

    def test_initialization(self):
        """Test module initialization."""
        sampler = BilinearSampler(padding_mode='zeros')
        assert sampler.padding_mode == 'zeros'

    def test_forward_shape(self):
        """Test output shape correctness."""
        B, C, H, W = 2, 256, 32, 32
        feature_map = torch.randn(B, C, H, W)
        sampling_locations = torch.rand(B, 50, 2) * 2 - 1  # Random locations in [-1, 1]

        sampler = BilinearSampler()
        sampled_features = sampler(feature_map, sampling_locations)

        assert sampled_features.shape == (B, 50, C)

    def test_sampling_correctness(self):
        """Test sampling produces reasonable values."""
        B, C, H, W = 1, 3, 16, 16
        feature_map = torch.ones(B, C, H, W) * 5.0

        # Sample at center (0, 0)
        sampling_locations = torch.tensor([[[0.0, 0.0]]])

        sampler = BilinearSampler()
        sampled = sampler(feature_map, sampling_locations)

        # Should be close to 5.0
        assert torch.allclose(sampled, torch.tensor([[[5.0, 5.0, 5.0]]]), atol=0.1)


class TestDeformableCrossAttention:
    """Test complete DeformableCrossAttention module."""

    def test_initialization(self):
        """Test module initialization."""
        attention = DeformableCrossAttention(
            embed_dim=256,
            num_heads=8,
            num_points=4
        )
        assert attention.embed_dim == 256
        assert attention.num_heads == 8
        assert attention.num_points == 4

    def test_forward_shape(self):
        """Test output shape correctness."""
        B, C = 2, 256
        query_features = torch.randn(B, C, 64, 64)
        key_features = torch.randn(B, C, 32, 32)

        attention = DeformableCrossAttention(embed_dim=C, num_heads=8, num_points=4)
        output = attention(query_features, key_features)

        assert 'fused_features' in output
        assert output['fused_features'].shape == (B, C, 64, 64)
        assert 'offsets' in output
        assert 'sampling_locations' in output

    def test_gradient_flow(self):
        """Test gradient flow through deformable attention."""
        B, C = 2, 256
        query_features = torch.randn(B, C, 32, 32, requires_grad=True)
        key_features = torch.randn(B, C, 16, 16, requires_grad=True)

        attention = DeformableCrossAttention(embed_dim=C, num_heads=8, num_points=4)
        output = attention(query_features, key_features)

        loss = output['fused_features'].mean()
        loss.backward()

        assert query_features.grad is not None
        assert key_features.grad is not None


class TestMultiScaleDeformableAttention:
    """Test MultiScaleDeformableAttention."""

    def test_initialization(self):
        """Test module initialization."""
        attention = MultiScaleDeformableAttention(
            embed_dim=256,
            num_heads=8,
            num_levels=3,
            num_points=4
        )
        assert attention.num_levels == 3

    def test_forward_shape(self):
        """Test output shape with pyramid inputs."""
        B, C = 2, 256
        query_pyramid = [
            torch.randn(B, C, 64, 64),
            torch.randn(B, C, 32, 32),
            torch.randn(B, C, 16, 16)
        ]
        key_pyramid = [
            torch.randn(B, C, 64, 64),
            torch.randn(B, C, 32, 32),
            torch.randn(B, C, 16, 16)
        ]

        attention = MultiScaleDeformableAttention(
            embed_dim=C,
            num_heads=8,
            num_levels=3,
            num_points=4
        )
        output = attention(query_pyramid, key_pyramid)

        assert 'fused_features' in output
        assert output['fused_features'].shape == (B, C, 64, 64)
        assert 'fused_pyramid' in output
        assert len(output['fused_pyramid']) == 3


# ============================================================================
# Test Consistency Loss Components
# ============================================================================

class TestComponentConsistencyLoss:
    """Test ComponentConsistencyLoss."""

    def test_initialization(self):
        """Test module initialization."""
        loss_fn = ComponentConsistencyLoss(loss_type='l1', reduction='mean')
        assert loss_fn.loss_type == 'l1'
        assert loss_fn.reduction == 'mean'

    def test_loss_computation(self):
        """Test loss computation."""
        pred_rgb = torch.randn(4, 64)
        pred_sonar = torch.randn(4, 64)

        loss_fn = ComponentConsistencyLoss(loss_type='l1')
        loss = loss_fn(pred_rgb, pred_sonar)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # L1 loss is non-negative

    def test_with_valid_mask(self):
        """Test loss with valid mask."""
        pred_rgb = torch.randn(4, 64)
        pred_sonar = torch.randn(4, 64)
        valid_mask = torch.rand(4, 64) > 0.5

        loss_fn = ComponentConsistencyLoss(loss_type='l1')
        loss = loss_fn(pred_rgb, pred_sonar, valid_mask)

        assert loss >= 0


class TestInfoNCELoss:
    """Test InfoNCE contrastive loss."""

    def test_initialization(self):
        """Test module initialization."""
        loss_fn = InfoNCELoss(temperature=0.07, normalize=True)
        assert loss_fn.temperature == 0.07
        assert loss_fn.normalize is True

    def test_loss_computation(self):
        """Test InfoNCE loss computation."""
        B, D = 8, 128
        embeddings_rgb = torch.randn(B, D)
        embeddings_sonar = torch.randn(B, D)

        loss_fn = InfoNCELoss(temperature=0.07)
        output = loss_fn(embeddings_rgb, embeddings_sonar)

        assert 'loss' in output
        assert output['loss'] >= 0
        assert 'rgb_to_sonar_acc' in output
        assert 0 <= output['rgb_to_sonar_acc'] <= 1

    def test_perfect_alignment(self):
        """Test loss with perfectly aligned embeddings."""
        B, D = 4, 128
        embeddings = torch.randn(B, D)

        loss_fn = InfoNCELoss(temperature=0.1)
        output = loss_fn(embeddings, embeddings)

        # With identical embeddings, accuracy should be high
        assert output['rgb_to_sonar_acc'] >= 0.8


class TestDualConsistencyLoss:
    """Test combined DualConsistencyLoss."""

    def test_initialization(self):
        """Test module initialization."""
        loss_fn = DualConsistencyLoss(
            component_weight=1.0,
            representation_weight=1.0
        )
        assert loss_fn.component_weight == 1.0
        assert loss_fn.representation_weight == 1.0

    def test_loss_computation(self):
        """Test dual consistency loss computation."""
        B = 4
        pred_rgb_component = torch.randn(B, 64)
        pred_sonar_component = torch.randn(B, 64)
        embeddings_rgb = torch.randn(B, 128)
        embeddings_sonar = torch.randn(B, 128)

        loss_fn = DualConsistencyLoss()
        output = loss_fn(
            pred_rgb_component,
            pred_sonar_component,
            embeddings_rgb,
            embeddings_sonar
        )

        assert 'total_loss' in output
        assert 'component_loss' in output
        assert 'infonce_loss' in output
        assert output['total_loss'] >= 0


class TestProjectionHead:
    """Test ProjectionHead for contrastive learning."""

    def test_initialization(self):
        """Test module initialization."""
        proj_head = ProjectionHead(input_dim=256, output_dim=128, num_layers=2)
        assert proj_head is not None

    def test_forward_shape(self):
        """Test output shape correctness."""
        B, C = 4, 256
        features_2d = torch.randn(B, C)
        features_4d = torch.randn(B, C, 16, 16)

        proj_head = ProjectionHead(input_dim=C, output_dim=128)

        # Test 2D features
        output_2d = proj_head(features_2d)
        assert output_2d.shape == (B, 128)

        # Test 4D features (should apply global pooling)
        output_4d = proj_head(features_4d)
        assert output_4d.shape == (B, 128)


# ============================================================================
# Test DCMA Integration
# ============================================================================

class TestDCMA:
    """Test complete DCMA module."""

    def test_initialization(self):
        """Test module initialization."""
        dcma = DCMA(embed_dim=256, num_heads=8)
        assert dcma.embed_dim == 256
        assert dcma.num_heads == 8

    def test_forward_single_scale(self):
        """Test forward pass with single-scale features."""
        B, C = 2, 256
        rgb_features = torch.randn(B, C, 64, 64)
        sonar_features = torch.randn(B, C, 32, 32)

        dcma = DCMA(embed_dim=C, num_heads=8, use_multi_scale=False)
        output = dcma(rgb_features, sonar_features)

        assert 'fused_features' in output
        assert output['fused_features'].shape == (B, C, 64, 64)
        assert 'consistency_losses' in output
        assert 'deformable_info' in output
        assert 'rgb_view_fused' in output
        assert 'sonar_view_fused' in output

    def test_forward_multi_scale(self):
        """Test forward pass with multi-scale features."""
        B, C = 2, 256
        rgb_pyramid = [
            torch.randn(B, C, 64, 64),
            torch.randn(B, C, 32, 32),
            torch.randn(B, C, 16, 16)
        ]
        sonar_pyramid = [
            torch.randn(B, C, 64, 64),
            torch.randn(B, C, 32, 32),
            torch.randn(B, C, 16, 16)
        ]

        dcma = DCMA(embed_dim=C, num_heads=8, use_multi_scale=True, num_levels=3)
        output = dcma(
            rgb_features=rgb_pyramid[0],
            sonar_features=sonar_pyramid[0],
            rgb_pyramid=rgb_pyramid,
            sonar_pyramid=sonar_pyramid
        )

        assert 'fused_features' in output
        assert output['fused_features'].shape == (B, C, 64, 64)

    def test_gradient_flow_e2e(self):
        """Test end-to-end gradient flow."""
        B, C = 2, 256
        rgb_features = torch.randn(B, C, 32, 32, requires_grad=True)
        sonar_features = torch.randn(B, C, 16, 16, requires_grad=True)

        dcma = DCMA(embed_dim=C, num_heads=8)
        output = dcma(rgb_features, sonar_features)

        # Backprop through fusion output
        loss_fusion = output['fused_features'].mean()
        loss_fusion.backward(retain_graph=True)

        assert rgb_features.grad is not None
        assert sonar_features.grad is not None

    def test_consistency_losses(self):
        """Test consistency loss computation."""
        B, C = 2, 256
        rgb_features = torch.randn(B, C, 32, 32)
        sonar_features = torch.randn(B, C, 16, 16)

        dcma = DCMA(embed_dim=C, num_heads=8)
        output = dcma(rgb_features, sonar_features, return_consistency=True)

        consistency_losses = output['consistency_losses']
        assert 'total_loss' in consistency_losses
        assert 'component_loss' in consistency_losses
        assert 'infonce_loss' in consistency_losses
        assert consistency_losses['total_loss'] >= 0

    def test_confidence_estimation(self):
        """Test that confidence scores are computed correctly."""
        B, C = 2, 256
        rgb_features = torch.randn(B, C, 32, 32)
        sonar_features = torch.randn(B, C, 16, 16)

        dcma = DCMA(embed_dim=C, num_heads=8)
        output = dcma(rgb_features, sonar_features)

        assert 'rgb_confidence' in output
        assert 'sonar_confidence' in output
        assert 'combined_confidence' in output

        # Confidence should be in [0, 1]
        assert (output['rgb_confidence'] >= 0).all() and (output['rgb_confidence'] <= 1).all()
        assert (output['sonar_confidence'] >= 0).all() and (output['sonar_confidence'] <= 1).all()


class TestDCMALite:
    """Test lightweight DCMA variant."""

    def test_initialization(self):
        """Test module initialization."""
        dcma_lite = DCMALite(embed_dim=256, num_heads=4)
        assert dcma_lite.dcma.num_heads == 4

    def test_forward_shape(self):
        """Test output shape correctness."""
        B, C = 2, 256
        rgb_features = torch.randn(B, C, 32, 32)
        sonar_features = torch.randn(B, C, 16, 16)

        dcma_lite = DCMALite(embed_dim=C, num_heads=4)
        output = dcma_lite(rgb_features, sonar_features)

        assert 'fused_features' in output
        assert output['fused_features'].shape == (B, C, 32, 32)


# ============================================================================
# Test Factory Functions
# ============================================================================

class TestFactoryFunctions:
    """Test factory functions for building modules from config."""

    def test_build_deformable_cross_attention(self):
        """Test building DeformableCrossAttention from config."""
        config = {'embed_dim': 256, 'num_heads': 8, 'num_points': 4}
        attention = build_deformable_cross_attention(config)
        assert attention.embed_dim == 256
        assert attention.num_points == 4

    def test_build_dual_consistency_loss(self):
        """Test building DualConsistencyLoss from config."""
        config = {
            'component_weight': 1.0,
            'representation_weight': 0.5,
            'infonce_temperature': 0.07
        }
        loss_fn = build_dual_consistency_loss(config)
        assert loss_fn.component_weight == 1.0
        assert loss_fn.representation_weight == 0.5

    def test_build_dcma(self):
        """Test building DCMA from config."""
        config = {
            'embed_dim': 256,
            'num_heads': 8,
            'use_multi_scale': False
        }
        dcma = build_dcma(config)
        assert dcma.embed_dim == 256
        assert dcma.num_heads == 8

    def test_build_dcma_lite(self):
        """Test building lightweight DCMA from config."""
        config = {'embed_dim': 256, 'num_heads': 4}
        dcma_lite = build_dcma_lite(config)
        assert dcma_lite.dcma.num_heads == 4


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete fusion pipeline."""

    def test_complete_pipeline(self):
        """Test complete fusion pipeline from RGB+Sonar to fused output."""
        B, C = 2, 256
        rgb_features = torch.randn(B, C, 64, 64)
        sonar_features = torch.randn(B, C, 32, 32)

        # Build complete DCMA
        dcma = DCMA(
            embed_dim=C,
            num_heads=8,
            num_deformable_points=4,
            use_multi_scale=False
        )

        # Forward pass
        output = dcma(rgb_features, sonar_features, return_consistency=True)

        # Verify all outputs
        assert 'fused_features' in output
        assert 'initial_fused' in output
        assert 'consistency_losses' in output
        assert 'deformable_info' in output
        assert 'rgb_view_fused' in output
        assert 'sonar_view_fused' in output

        # Verify shapes
        assert output['fused_features'].shape == (B, C, 64, 64)

        # Verify losses
        consistency = output['consistency_losses']
        assert consistency['total_loss'] >= 0
        assert consistency['component_loss'] >= 0
        assert consistency['infonce_loss'] >= 0

    def test_multi_scale_pipeline(self):
        """Test multi-scale fusion pipeline."""
        B, C = 2, 256
        rgb_pyramid = [
            torch.randn(B, C, 64, 64),
            torch.randn(B, C, 32, 32),
            torch.randn(B, C, 16, 16)
        ]
        sonar_pyramid = [
            torch.randn(B, C, 64, 64),
            torch.randn(B, C, 32, 32),
            torch.randn(B, C, 16, 16)
        ]

        dcma = DCMA(
            embed_dim=C,
            num_heads=8,
            use_multi_scale=True,
            num_levels=3
        )

        output = dcma(
            rgb_features=rgb_pyramid[0],
            sonar_features=sonar_pyramid[0],
            rgb_pyramid=rgb_pyramid,
            sonar_pyramid=sonar_pyramid
        )

        assert 'fused_features' in output
        assert 'level_weights' in output
        assert output['fused_features'].shape == (B, C, 64, 64)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
