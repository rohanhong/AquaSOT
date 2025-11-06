"""
Comprehensive Test Suite for Fusion Modules (Phase 4)

Tests all components of the G²-DCMA fusion framework:
- Geometric Gating (geometric_gating.py)
- Deformable Attention (deformable_attention.py)
- Consistency Loss (consistency_loss.py)
- G²-DCMA Integration (g2_dcma.py)

Run with: pytest tests/test_fusion_modules.py -v
"""

import pytest
import torch
import torch.nn as nn

from aquasot3d.models.fusion import (
    # Geometric Gating
    RadialProfileExtractor,
    ScalePyramidExtractor,
    GeometricGatingFunction,
    GeometryGatedAttention,
    build_geometry_gated_attention,
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
    # G²-DCMA
    G2DCMA,
    G2DCMALite,
    build_g2_dcma,
    build_g2_dcma_lite
)


# ============================================================================
# Test Geometric Gating Components
# ============================================================================

class TestRadialProfileExtractor:
    """Test RadialProfileExtractor for sonar radial encoding."""

    def test_initialization(self):
        """Test module initialization."""
        extractor = RadialProfileExtractor(num_bins=50, feature_dim=256)
        assert extractor.num_bins == 50
        assert extractor.feature_dim == 256
        assert extractor.learnable_weights is True

    def test_forward_shape(self):
        """Test output shape correctness."""
        B, C, H, W = 2, 256, 32, 32
        sonar_features = torch.randn(B, C, H, W)

        extractor = RadialProfileExtractor(num_bins=50, feature_dim=C)
        output = extractor(sonar_features)

        assert output.shape == (B, 64), f"Expected shape (2, 64), got {output.shape}"

    def test_radial_bins_extraction(self):
        """Test radial bin extraction."""
        B, C, H, W = 2, 256, 32, 32
        sonar_features = torch.randn(B, C, H, W)

        extractor = RadialProfileExtractor(num_bins=25, feature_dim=C)
        radial_profile = extractor.extract_radial_bins(sonar_features)

        assert radial_profile.shape == (B, 25)
        assert not torch.isnan(radial_profile).any()
        assert not torch.isinf(radial_profile).any()

    def test_different_sensor_positions(self):
        """Test different sensor position modes."""
        B, C, H, W = 2, 256, 32, 32
        sonar_features = torch.randn(B, C, H, W)

        extractor = RadialProfileExtractor(num_bins=50, feature_dim=C)

        # Bottom center (default)
        profile1 = extractor.extract_radial_bins(sonar_features, sensor_position="bottom_center")
        assert profile1.shape == (B, 50)

        # Center
        profile2 = extractor.extract_radial_bins(sonar_features, sensor_position="center")
        assert profile2.shape == (B, 50)

        # Profiles should be different
        assert not torch.allclose(profile1, profile2)


class TestScalePyramidExtractor:
    """Test ScalePyramidExtractor for RGB scale encoding."""

    def test_initialization(self):
        """Test module initialization."""
        extractor = ScalePyramidExtractor(feature_dim=256, num_scales=3)
        assert extractor.feature_dim == 256
        assert extractor.num_scales == 3
        assert extractor.use_attention is True

    def test_forward_shape(self):
        """Test output shape correctness."""
        B, C, H, W = 2, 256, 64, 64
        rgb_features = torch.randn(B, C, H, W)

        extractor = ScalePyramidExtractor(feature_dim=C, num_scales=3)
        output = extractor(rgb_features)

        assert output.shape == (B, 64), f"Expected shape (2, 64), got {output.shape}"

    def test_with_pyramid_levels(self):
        """Test with explicit pyramid levels."""
        B, C = 2, 256
        pyramid_levels = [
            torch.randn(B, C, 64, 64),
            torch.randn(B, C, 32, 32),
            torch.randn(B, C, 16, 16)
        ]

        extractor = ScalePyramidExtractor(feature_dim=C, num_scales=3)
        output = extractor(pyramid_levels[0], pyramid_levels=pyramid_levels)

        assert output.shape == (B, 64)
        assert not torch.isnan(output).any()


class TestGeometricGatingFunction:
    """Test GeometricGatingFunction."""

    def test_initialization(self):
        """Test module initialization."""
        gating_fn = GeometricGatingFunction(
            radial_dim=64,
            scale_dim=64,
            hidden_dim=128,
            num_heads=8
        )
        assert gating_fn.num_heads == 8

    def test_forward_shape(self):
        """Test output shape correctness."""
        B = 4
        radial_encoding = torch.randn(B, 64)
        scale_encoding = torch.randn(B, 64)

        gating_fn = GeometricGatingFunction(radial_dim=64, scale_dim=64, num_heads=8)
        gating_weights = gating_fn(radial_encoding, scale_encoding)

        assert gating_weights.shape == (B, 8)
        # Gating weights should be in [0, 1] due to sigmoid
        assert (gating_weights >= 0).all() and (gating_weights <= 1).all()


class TestGeometryGatedAttention:
    """Test complete GeometryGatedAttention module."""

    def test_initialization(self):
        """Test module initialization."""
        attention = GeometryGatedAttention(embed_dim=256, num_heads=8)
        assert attention.embed_dim == 256
        assert attention.num_heads == 8

    def test_forward_shape(self):
        """Test output shape correctness."""
        B, C = 2, 256
        rgb_features = torch.randn(B, C, 64, 64)
        sonar_features = torch.randn(B, C, 32, 32)

        attention = GeometryGatedAttention(embed_dim=C, num_heads=8)
        output = attention(rgb_features, sonar_features)

        assert 'fused_features' in output
        assert output['fused_features'].shape == (B, C, 64, 64)
        assert 'gating_weights' in output
        assert output['gating_weights'].shape == (B, 8)

    def test_gradient_flow(self):
        """Test gradient flow through the module."""
        B, C = 2, 256
        rgb_features = torch.randn(B, C, 64, 64, requires_grad=True)
        sonar_features = torch.randn(B, C, 32, 32, requires_grad=True)

        attention = GeometryGatedAttention(embed_dim=C, num_heads=8)
        output = attention(rgb_features, sonar_features)

        loss = output['fused_features'].mean()
        loss.backward()

        assert rgb_features.grad is not None
        assert sonar_features.grad is not None


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
# Test G²-DCMA Integration
# ============================================================================

class TestG2DCMA:
    """Test complete G²-DCMA module."""

    def test_initialization(self):
        """Test module initialization."""
        g2_dcma = G2DCMA(embed_dim=256, num_heads=8)
        assert g2_dcma.embed_dim == 256
        assert g2_dcma.num_heads == 8

    def test_forward_single_scale(self):
        """Test forward pass with single-scale features."""
        B, C = 2, 256
        rgb_features = torch.randn(B, C, 64, 64)
        sonar_features = torch.randn(B, C, 32, 32)

        g2_dcma = G2DCMA(embed_dim=C, num_heads=8, use_multi_scale=False)
        output = g2_dcma(rgb_features, sonar_features)

        assert 'fused_features' in output
        assert output['fused_features'].shape == (B, C, 64, 64)
        assert 'consistency_losses' in output
        assert 'geometric_info' in output
        assert 'deformable_info' in output

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

        g2_dcma = G2DCMA(embed_dim=C, num_heads=8, use_multi_scale=True, num_levels=3)
        output = g2_dcma(
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

        g2_dcma = G2DCMA(embed_dim=C, num_heads=8)
        output = g2_dcma(rgb_features, sonar_features)

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

        g2_dcma = G2DCMA(embed_dim=C, num_heads=8)
        output = g2_dcma(rgb_features, sonar_features, return_consistency=True)

        consistency_losses = output['consistency_losses']
        assert 'total_loss' in consistency_losses
        assert 'component_loss' in consistency_losses
        assert 'infonce_loss' in consistency_losses
        assert consistency_losses['total_loss'] >= 0


class TestG2DCMALite:
    """Test lightweight G²-DCMA variant."""

    def test_initialization(self):
        """Test module initialization."""
        g2_dcma_lite = G2DCMALite(embed_dim=256, num_heads=4)
        assert g2_dcma_lite.g2_dcma.num_heads == 4

    def test_forward_shape(self):
        """Test output shape correctness."""
        B, C = 2, 256
        rgb_features = torch.randn(B, C, 32, 32)
        sonar_features = torch.randn(B, C, 16, 16)

        g2_dcma_lite = G2DCMALite(embed_dim=C, num_heads=4)
        output = g2_dcma_lite(rgb_features, sonar_features)

        assert 'fused_features' in output
        assert output['fused_features'].shape == (B, C, 32, 32)


# ============================================================================
# Test Factory Functions
# ============================================================================

class TestFactoryFunctions:
    """Test factory functions for building modules from config."""

    def test_build_geometry_gated_attention(self):
        """Test building GeometryGatedAttention from config."""
        config = {'embed_dim': 256, 'num_heads': 8, 'dropout': 0.1}
        attention = build_geometry_gated_attention(config)
        assert attention.embed_dim == 256
        assert attention.num_heads == 8

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

    def test_build_g2_dcma(self):
        """Test building G²-DCMA from config."""
        config = {
            'embed_dim': 256,
            'num_heads': 8,
            'num_radial_bins': 50,
            'num_scales': 3,
            'use_multi_scale': False
        }
        g2_dcma = build_g2_dcma(config)
        assert g2_dcma.embed_dim == 256
        assert g2_dcma.num_heads == 8

    def test_build_g2_dcma_lite(self):
        """Test building lightweight G²-DCMA from config."""
        config = {'embed_dim': 256, 'num_heads': 4}
        g2_dcma_lite = build_g2_dcma_lite(config)
        assert g2_dcma_lite.g2_dcma.num_heads == 4


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

        # Build complete G²-DCMA
        g2_dcma = G2DCMA(
            embed_dim=C,
            num_heads=8,
            num_radial_bins=50,
            num_scales=3,
            num_deformable_points=4,
            use_multi_scale=False
        )

        # Forward pass
        output = g2_dcma(rgb_features, sonar_features, return_consistency=True)

        # Verify all outputs
        assert 'fused_features' in output
        assert 'geometry_gated' in output
        assert 'deformable_attended' in output
        assert 'consistency_losses' in output
        assert 'geometric_info' in output
        assert 'deformable_info' in output

        # Verify shapes
        assert output['fused_features'].shape == (B, C, 64, 64)
        assert output['geometric_info']['gating_weights'].shape == (B, 8)

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

        g2_dcma = G2DCMA(
            embed_dim=C,
            num_heads=8,
            use_multi_scale=True,
            num_levels=3
        )

        output = g2_dcma(
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
