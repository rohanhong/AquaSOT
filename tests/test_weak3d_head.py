"""
Unit Tests for Weak 3D Center Head

Tests for Weak3DHead, extract_radial_activation, and related utilities.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import pytest
import torch
import torch.nn as nn

from aquasot3d.models.heads import (
    Weak3DHead,
    extract_radial_activation,
    build_weak3d_head,
)


class TestRadialActivationExtraction:
    """Tests for radial activation extraction function."""

    def test_extract_radial_activation_basic(self):
        """Test basic radial activation extraction."""
        batch_size = 4
        sonar_img = torch.randn(batch_size, 1, 376, 682)
        sonar_feat = torch.randn(batch_size, 256, 14, 14)
        num_bins = 50

        profile = extract_radial_activation(sonar_img, sonar_feat, num_bins)

        assert profile.shape == (batch_size, num_bins)

    def test_extract_radial_activation_different_sizes(self):
        """Test with different input sizes."""
        batch_size = 2
        # Different feature map size
        sonar_img = torch.randn(batch_size, 1, 376, 682)
        sonar_feat = torch.randn(batch_size, 128, 7, 7)
        num_bins = 30

        profile = extract_radial_activation(sonar_img, sonar_feat, num_bins)

        assert profile.shape == (batch_size, num_bins)

    def test_extract_radial_activation_single_batch(self):
        """Test with single sample."""
        sonar_img = torch.randn(1, 1, 376, 682)
        sonar_feat = torch.randn(1, 256, 14, 14)
        num_bins = 50

        profile = extract_radial_activation(sonar_img, sonar_feat, num_bins)

        assert profile.shape == (1, num_bins)

    def test_extract_radial_activation_values(self):
        """Test that extracted values are reasonable."""
        batch_size = 4
        sonar_img = torch.randn(batch_size, 1, 376, 682)
        sonar_feat = torch.randn(batch_size, 256, 14, 14)
        num_bins = 50

        profile = extract_radial_activation(sonar_img, sonar_feat, num_bins)

        # Check that profile contains valid values (not all zeros, not NaN)
        assert not torch.all(profile == 0)
        assert not torch.any(torch.isnan(profile))

    def test_extract_radial_activation_gradient_flow(self):
        """Test gradient flow through radial extraction."""
        batch_size = 4
        sonar_img = torch.randn(batch_size, 1, 376, 682)
        sonar_feat = torch.randn(batch_size, 256, 14, 14, requires_grad=True)
        num_bins = 50

        profile = extract_radial_activation(sonar_img, sonar_feat, num_bins)
        loss = profile.mean()

        loss.backward()

        assert sonar_feat.grad is not None
        assert not torch.all(sonar_feat.grad == 0)


class TestWeak3DHead:
    """Tests for Weak3DHead module."""

    @pytest.fixture
    def weak3d_head(self):
        """Create Weak3DHead for testing."""
        return Weak3DHead(
            rgb_feat_dim=768,
            sonar_feat_dim=384,
            num_radial_bins=50,
            hidden_dim=256
        )

    def test_weak3d_head_initialization(self, weak3d_head):
        """Test Weak3DHead initialization."""
        assert isinstance(weak3d_head, Weak3DHead)
        assert weak3d_head.rgb_feat_dim == 768
        assert weak3d_head.sonar_feat_dim == 384
        assert weak3d_head.num_radial_bins == 50
        assert weak3d_head.hidden_dim == 256

    def test_weak3d_head_forward_1d(self, weak3d_head):
        """Test forward pass with 1D features."""
        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)
        sonar_radial = torch.randn(batch_size, 50)

        outputs = weak3d_head(rgb_features, sonar_features, sonar_radial)

        assert 'center_3d' in outputs
        assert 'X_rgb' in outputs
        assert 'X_sonar' in outputs
        assert 'X_fused' in outputs
        assert 'X_weights' in outputs
        assert 'Y' in outputs
        assert 'Z' in outputs
        assert 'Z_weighted' in outputs
        assert 'consistency_loss' in outputs

        assert outputs['center_3d'].shape == (batch_size, 3)
        assert outputs['X_rgb'].shape == (batch_size, 1)
        assert outputs['X_sonar'].shape == (batch_size, 1)
        assert outputs['X_fused'].shape == (batch_size, 1)
        assert outputs['X_weights'].shape == (batch_size, 2)
        assert outputs['Y'].shape == (batch_size, 1)
        assert outputs['Z'].shape == (batch_size, 1)
        assert outputs['Z_weighted'].shape == (batch_size, 1)

    def test_weak3d_head_forward_3d(self, weak3d_head):
        """Test forward pass with 3D features (sequence)."""
        batch_size = 4
        seq_len = 196
        rgb_features = torch.randn(batch_size, seq_len, 768)
        sonar_features = torch.randn(batch_size, seq_len // 2, 384)
        sonar_radial = torch.randn(batch_size, 50)

        outputs = weak3d_head(rgb_features, sonar_features, sonar_radial)

        assert outputs['center_3d'].shape == (batch_size, 3)

    def test_weak3d_head_forward_4d(self, weak3d_head):
        """Test forward pass with 4D features (spatial)."""
        batch_size = 4
        rgb_features = torch.randn(batch_size, 768, 14, 14)
        sonar_features = torch.randn(batch_size, 384, 7, 7)
        sonar_radial = torch.randn(batch_size, 50)

        outputs = weak3d_head(rgb_features, sonar_features, sonar_radial)

        assert outputs['center_3d'].shape == (batch_size, 3)

    def test_weak3d_head_fusion_weights_sum_to_one(self, weak3d_head):
        """Test that X fusion weights sum to one."""
        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)
        sonar_radial = torch.randn(batch_size, 50)

        outputs = weak3d_head(rgb_features, sonar_features, sonar_radial)
        weights = outputs['X_weights']

        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_weak3d_head_consistency_loss(self, weak3d_head):
        """Test consistency loss computation."""
        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)
        sonar_radial = torch.randn(batch_size, 50)

        outputs = weak3d_head(rgb_features, sonar_features, sonar_radial, compute_consistency_loss=True)

        assert 'consistency_loss' in outputs
        assert outputs['consistency_loss'].numel() == 1
        assert outputs['consistency_loss'] >= 0

    def test_weak3d_head_without_consistency_loss(self, weak3d_head):
        """Test forward without consistency loss."""
        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)
        sonar_radial = torch.randn(batch_size, 50)

        outputs = weak3d_head(rgb_features, sonar_features, sonar_radial, compute_consistency_loss=False)

        assert 'consistency_loss' not in outputs or outputs['consistency_loss'] == 0

    def test_weak3d_head_z_components(self, weak3d_head):
        """Test Z estimation components."""
        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)
        sonar_radial = torch.randn(batch_size, 50)

        outputs = weak3d_head(rgb_features, sonar_features, sonar_radial)

        Z = outputs['Z']
        Z_weighted = outputs['Z_weighted']

        # Z should be different from just weighted (due to MLP)
        if weak3d_head.use_residual_z:
            # With residual, Z includes both components
            assert not torch.allclose(Z, Z_weighted, atol=1e-2)

    def test_weak3d_head_gradient_flow(self, weak3d_head):
        """Test gradient flow through Weak3DHead."""
        rgb_features = torch.randn(4, 768, requires_grad=True)
        sonar_features = torch.randn(4, 384, requires_grad=True)
        sonar_radial = torch.randn(4, 50, requires_grad=True)

        outputs = weak3d_head(rgb_features, sonar_features, sonar_radial)
        loss = outputs['center_3d'].mean()

        loss.backward()

        assert rgb_features.grad is not None
        assert sonar_features.grad is not None
        assert sonar_radial.grad is not None
        assert not torch.all(rgb_features.grad == 0)
        assert not torch.all(sonar_features.grad == 0)
        assert not torch.all(sonar_radial.grad == 0)

    def test_weak3d_head_residual_z(self):
        """Test Z estimation with and without residual."""
        head_with_residual = Weak3DHead(
            rgb_feat_dim=768,
            sonar_feat_dim=384,
            use_residual_z=True
        )
        head_without_residual = Weak3DHead(
            rgb_feat_dim=768,
            sonar_feat_dim=384,
            use_residual_z=False
        )

        rgb_features = torch.randn(4, 768)
        sonar_features = torch.randn(4, 384)
        sonar_radial = torch.randn(4, 50)

        outputs_with = head_with_residual(rgb_features, sonar_features, sonar_radial)
        outputs_without = head_without_residual(rgb_features, sonar_features, sonar_radial)

        # Both should produce valid Z
        assert outputs_with['Z'].shape == (4, 1)
        assert outputs_without['Z'].shape == (4, 1)

    def test_weak3d_head_custom_bins(self):
        """Test Weak3DHead with different number of radial bins."""
        head = Weak3DHead(
            rgb_feat_dim=768,
            sonar_feat_dim=384,
            num_radial_bins=30
        )

        rgb_features = torch.randn(4, 768)
        sonar_features = torch.randn(4, 384)
        sonar_radial = torch.randn(4, 30)  # Matching num_bins

        outputs = head(rgb_features, sonar_features, sonar_radial)

        assert outputs['center_3d'].shape == (4, 3)


class Test3DLossComputation:
    """Tests for 3D loss computation."""

    @pytest.fixture
    def weak3d_head(self):
        """Create Weak3DHead for testing."""
        return Weak3DHead(
            rgb_feat_dim=768,
            sonar_feat_dim=384
        )

    def test_3d_loss_l1(self, weak3d_head):
        """Test L1 3D loss."""
        pred_center = torch.randn(4, 3)
        target_center = torch.randn(4, 3)

        loss = weak3d_head.compute_3d_loss(pred_center, target_center, loss_type='l1')

        assert loss.numel() == 1
        assert loss >= 0

    def test_3d_loss_l2(self, weak3d_head):
        """Test L2 3D loss."""
        pred_center = torch.randn(4, 3)
        target_center = torch.randn(4, 3)

        loss = weak3d_head.compute_3d_loss(pred_center, target_center, loss_type='l2')

        assert loss.numel() == 1
        assert loss >= 0

    def test_3d_loss_smooth_l1(self, weak3d_head):
        """Test Smooth L1 3D loss."""
        pred_center = torch.randn(4, 3)
        target_center = torch.randn(4, 3)

        loss = weak3d_head.compute_3d_loss(pred_center, target_center, loss_type='smooth_l1')

        assert loss.numel() == 1
        assert loss >= 0

    def test_3d_loss_perfect_match(self, weak3d_head):
        """Test 3D loss for perfect match."""
        center = torch.randn(4, 3)

        loss = weak3d_head.compute_3d_loss(center, center, loss_type='l1')

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_3d_loss_gradient_flow(self, weak3d_head):
        """Test gradient flow through 3D loss."""
        pred_center = torch.randn(4, 3, requires_grad=True)
        target_center = torch.randn(4, 3)

        loss = weak3d_head.compute_3d_loss(pred_center, target_center, loss_type='l1')

        loss.backward()

        assert pred_center.grad is not None
        assert not torch.all(pred_center.grad == 0)


class TestWeak3DHeadFactory:
    """Tests for build_weak3d_head factory function."""

    def test_build_weak3d_head_default(self):
        """Test building Weak3DHead with default parameters."""
        head = build_weak3d_head(rgb_feat_dim=768, sonar_feat_dim=384)
        assert isinstance(head, Weak3DHead)
        assert head.rgb_feat_dim == 768
        assert head.sonar_feat_dim == 384

    def test_build_weak3d_head_custom(self):
        """Test building Weak3DHead with custom parameters."""
        head = build_weak3d_head(
            rgb_feat_dim=512,
            sonar_feat_dim=256,
            num_radial_bins=30,
            hidden_dim=128,
            use_residual_z=False
        )
        assert isinstance(head, Weak3DHead)
        assert head.rgb_feat_dim == 512
        assert head.sonar_feat_dim == 256
        assert head.num_radial_bins == 30
        assert head.hidden_dim == 128
        assert not head.use_residual_z


class TestWeak3DHeadIntegration:
    """Integration tests for Weak3DHead with other modules."""

    def test_weak3d_with_radial_extraction(self):
        """Test Weak3DHead with radial activation extraction."""
        head = Weak3DHead(rgb_feat_dim=768, sonar_feat_dim=384)

        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384, 14, 14)

        # Simulate sonar image and extract radial profile
        sonar_img = torch.randn(batch_size, 1, 376, 682)
        sonar_radial = extract_radial_activation(sonar_img, sonar_features, num_bins=50)

        # Predict 3D center
        outputs = head(rgb_features, sonar_features, sonar_radial)

        assert outputs['center_3d'].shape == (batch_size, 3)

    def test_weak3d_with_fusion_head(self):
        """Test Weak3DHead integration with FusionHead."""
        from aquasot3d.models.heads import build_fusion_head

        fusion_head = build_fusion_head(rgb_channels=768, sonar_channels=384, hidden_dim=256)
        weak3d_head = build_weak3d_head(rgb_feat_dim=256, sonar_feat_dim=256)

        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)
        sonar_radial = torch.randn(batch_size, 50)

        # First fuse features
        fusion_outputs = fusion_head(rgb_features, sonar_features)
        fused_features = fusion_outputs['fused_features']

        # Then predict 3D with fused features (using same for both branches)
        weak3d_outputs = weak3d_head(fused_features, fused_features, sonar_radial)

        assert weak3d_outputs['center_3d'].shape == (batch_size, 3)

    def test_complete_prediction_pipeline(self):
        """Test complete multi-modal 3D prediction pipeline."""
        from aquasot3d.models.heads import build_box_head, build_fusion_head

        # Create all heads
        rgb_box_head = build_box_head('rgb', in_channels=768)
        sonar_box_head = build_box_head('sonar', in_channels=384)
        fusion_head = build_fusion_head(rgb_channels=768, sonar_channels=384)
        fusion_box_head = build_box_head('fusion', in_channels=256)
        weak3d_head = build_weak3d_head(rgb_feat_dim=768, sonar_feat_dim=384)

        # Simulate features
        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features_4d = torch.randn(batch_size, 384, 7, 7)
        sonar_img = torch.randn(batch_size, 1, 376, 682)

        # Extract radial profile
        sonar_radial = extract_radial_activation(sonar_img, sonar_features_4d)

        # Flatten for 1D heads
        sonar_features_1d = sonar_features_4d.mean(dim=[2, 3])

        # Independent 2D predictions
        rgb_boxes = rgb_box_head(rgb_features)
        sonar_boxes = sonar_box_head(sonar_features_1d)

        # Fused prediction
        fusion_outputs = fusion_head(rgb_features, sonar_features_1d)
        fusion_boxes = fusion_box_head(fusion_outputs['fused_features'])

        # 3D prediction
        weak3d_outputs = weak3d_head(rgb_features, sonar_features_1d, sonar_radial)

        # Verify all outputs
        assert rgb_boxes['pred_boxes'].shape == (batch_size, 4)
        assert sonar_boxes['pred_boxes'].shape == (batch_size, 4)
        assert fusion_boxes['pred_boxes'].shape == (batch_size, 4)
        assert weak3d_outputs['center_3d'].shape == (batch_size, 3)

    def test_weak3d_with_temporal_features(self):
        """Test Weak3DHead with temporally aggregated features."""
        head = Weak3DHead(rgb_feat_dim=768, sonar_feat_dim=384)

        # Simulate temporally aggregated features (3D: B, T, C)
        batch_size = 4
        temporal_len = 5
        rgb_features = torch.randn(batch_size, temporal_len, 768)
        sonar_features = torch.randn(batch_size, temporal_len, 384)
        sonar_radial = torch.randn(batch_size, 50)

        outputs = head(rgb_features, sonar_features, sonar_radial)

        assert outputs['center_3d'].shape == (batch_size, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
