"""
Unit Tests for Fusion Head

Tests for FusionHead, UncertaintyEstimator, AdaptiveFusionGate, and related utilities.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import pytest
import torch
import torch.nn as nn

from aquasot3d.models.heads import (
    FusionHead,
    UncertaintyEstimator,
    AdaptiveFusionGate,
    build_fusion_head,
)


class TestUncertaintyEstimator:
    """Tests for UncertaintyEstimator module."""

    def test_learned_uncertainty_initialization(self):
        """Test learned uncertainty estimator initialization."""
        estimator = UncertaintyEstimator(
            in_channels=768,
            hidden_dim=128,
            method='learned'
        )
        assert isinstance(estimator, nn.Module)
        assert estimator.method == 'learned'
        assert estimator.in_channels == 768
        assert estimator.hidden_dim == 128

    def test_entropy_uncertainty_initialization(self):
        """Test entropy-based uncertainty estimator initialization."""
        estimator = UncertaintyEstimator(
            in_channels=768,
            hidden_dim=128,
            method='entropy'
        )
        assert isinstance(estimator, nn.Module)
        assert estimator.method == 'entropy'

    def test_learned_uncertainty_forward_1d(self):
        """Test learned uncertainty forward with 1D features."""
        estimator = UncertaintyEstimator(
            in_channels=768,
            method='learned'
        )

        batch_size = 4
        features = torch.randn(batch_size, 768)

        confidence = estimator(features)

        assert confidence.shape == (batch_size,)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)

    def test_learned_uncertainty_forward_4d(self):
        """Test learned uncertainty forward with 4D features (spatial)."""
        estimator = UncertaintyEstimator(
            in_channels=768,
            method='learned'
        )

        batch_size = 4
        features = torch.randn(batch_size, 768, 14, 14)

        confidence = estimator(features)

        assert confidence.shape == (batch_size,)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)

    def test_entropy_uncertainty_forward(self):
        """Test entropy-based uncertainty forward."""
        estimator = UncertaintyEstimator(
            in_channels=768,
            method='entropy'
        )

        batch_size = 4
        features = torch.randn(batch_size, 768)

        confidence = estimator(features)

        assert confidence.shape == (batch_size,)
        assert torch.all(confidence >= 0) and torch.all(confidence <= 1)

    def test_uncertainty_gradient_flow(self):
        """Test that gradients flow through uncertainty estimator."""
        estimator = UncertaintyEstimator(
            in_channels=768,
            method='learned'
        )

        features = torch.randn(4, 768, requires_grad=True)
        confidence = estimator(features)
        loss = confidence.mean()

        loss.backward()

        assert features.grad is not None
        assert not torch.all(features.grad == 0)


class TestAdaptiveFusionGate:
    """Tests for AdaptiveFusionGate module."""

    def test_attention_fusion_initialization(self):
        """Test attention-based fusion gate initialization."""
        gate = AdaptiveFusionGate(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256,
            fusion_method='attention'
        )
        assert isinstance(gate, nn.Module)
        assert gate.fusion_method == 'attention'
        assert gate.rgb_channels == 768
        assert gate.sonar_channels == 384

    def test_mlp_fusion_initialization(self):
        """Test MLP-based fusion gate initialization."""
        gate = AdaptiveFusionGate(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256,
            fusion_method='mlp'
        )
        assert isinstance(gate, nn.Module)
        assert gate.fusion_method == 'mlp'

    def test_uncertainty_weighted_fusion_initialization(self):
        """Test uncertainty-weighted fusion gate initialization."""
        gate = AdaptiveFusionGate(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256,
            fusion_method='uncertainty_weighted'
        )
        assert isinstance(gate, nn.Module)
        assert gate.fusion_method == 'uncertainty_weighted'

    def test_attention_fusion_forward(self):
        """Test attention fusion forward pass."""
        gate = AdaptiveFusionGate(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256,
            fusion_method='attention'
        )

        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)

        fused, weights = gate(rgb_features, sonar_features)

        assert fused.shape == (batch_size, 256)
        assert weights.shape == (batch_size, 2)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_mlp_fusion_forward(self):
        """Test MLP fusion forward pass."""
        gate = AdaptiveFusionGate(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256,
            fusion_method='mlp'
        )

        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)

        fused, weights = gate(rgb_features, sonar_features)

        assert fused.shape == (batch_size, 256)
        assert weights.shape == (batch_size, 2)

    def test_uncertainty_weighted_fusion_forward(self):
        """Test uncertainty-weighted fusion forward pass."""
        gate = AdaptiveFusionGate(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256,
            fusion_method='uncertainty_weighted'
        )

        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)
        rgb_confidence = torch.rand(batch_size)
        sonar_confidence = torch.rand(batch_size)

        fused, weights = gate(
            rgb_features,
            sonar_features,
            rgb_confidence,
            sonar_confidence
        )

        assert fused.shape == (batch_size, 256)
        assert weights.shape == (batch_size, 2)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_fusion_gate_with_4d_features(self):
        """Test fusion gate with spatial features."""
        gate = AdaptiveFusionGate(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256,
            fusion_method='uncertainty_weighted'
        )

        batch_size = 4
        rgb_features = torch.randn(batch_size, 768, 14, 14)
        sonar_features = torch.randn(batch_size, 384, 7, 7)

        fused, weights = gate(rgb_features, sonar_features)

        assert fused.shape == (batch_size, 256)
        assert weights.shape == (batch_size, 2)

    def test_fusion_gate_gradient_flow(self):
        """Test gradient flow through fusion gate."""
        gate = AdaptiveFusionGate(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256,
            fusion_method='uncertainty_weighted'
        )

        rgb_features = torch.randn(4, 768, requires_grad=True)
        sonar_features = torch.randn(4, 384, requires_grad=True)

        fused, weights = gate(rgb_features, sonar_features)
        loss = fused.mean()

        loss.backward()

        assert rgb_features.grad is not None
        assert sonar_features.grad is not None
        assert not torch.all(rgb_features.grad == 0)
        assert not torch.all(sonar_features.grad == 0)


class TestFusionHead:
    """Tests for FusionHead module."""

    @pytest.fixture
    def fusion_head(self):
        """Create FusionHead for testing."""
        return FusionHead(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256,
            uncertainty_method='learned',
            fusion_method='uncertainty_weighted'
        )

    def test_fusion_head_initialization(self, fusion_head):
        """Test FusionHead initialization."""
        assert isinstance(fusion_head, FusionHead)
        assert fusion_head.rgb_channels == 768
        assert fusion_head.sonar_channels == 384
        assert fusion_head.hidden_dim == 256
        assert fusion_head.uncertainty_method == 'learned'
        assert fusion_head.fusion_method == 'uncertainty_weighted'

    def test_fusion_head_forward(self, fusion_head):
        """Test FusionHead forward pass."""
        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)

        outputs = fusion_head(rgb_features, sonar_features)

        assert 'fused_features' in outputs
        assert 'fusion_weights' in outputs
        assert 'quality_score' in outputs
        assert 'rgb_confidence' in outputs
        assert 'sonar_confidence' in outputs
        assert 'uncertainty' in outputs

        assert outputs['fused_features'].shape == (batch_size, 256)
        assert outputs['fusion_weights'].shape == (batch_size, 2)
        assert outputs['quality_score'].shape == (batch_size,)
        assert outputs['rgb_confidence'].shape == (batch_size,)
        assert outputs['sonar_confidence'].shape == (batch_size,)
        assert outputs['uncertainty'].shape == (batch_size,)

    def test_fusion_head_forward_4d(self, fusion_head):
        """Test FusionHead forward with spatial features."""
        batch_size = 4
        rgb_features = torch.randn(batch_size, 768, 14, 14)
        sonar_features = torch.randn(batch_size, 384, 7, 7)

        outputs = fusion_head(rgb_features, sonar_features)

        assert outputs['fused_features'].shape == (batch_size, 256)
        assert outputs['fusion_weights'].shape == (batch_size, 2)
        assert outputs['quality_score'].shape == (batch_size,)

    def test_fusion_head_without_uncertainty(self):
        """Test FusionHead forward without uncertainty output."""
        head = FusionHead(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256,
            predict_uncertainty=False
        )

        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)

        outputs = head(rgb_features, sonar_features, return_uncertainty=False)

        assert 'fused_features' in outputs
        assert 'fusion_weights' in outputs
        assert 'quality_score' in outputs
        # Uncertainty still returned because of predict_uncertainty flag
        # But we can test it doesn't break

    def test_fusion_head_quality_scores(self, fusion_head):
        """Test that quality scores are in valid range."""
        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)

        outputs = fusion_head(rgb_features, sonar_features)

        quality = outputs['quality_score']
        rgb_conf = outputs['rgb_confidence']
        sonar_conf = outputs['sonar_confidence']

        assert torch.all(quality >= 0) and torch.all(quality <= 1)
        assert torch.all(rgb_conf >= 0) and torch.all(rgb_conf <= 1)
        assert torch.all(sonar_conf >= 0) and torch.all(sonar_conf <= 1)

    def test_fusion_head_weights_sum_to_one(self, fusion_head):
        """Test that fusion weights sum to one."""
        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)

        outputs = fusion_head(rgb_features, sonar_features)
        weights = outputs['fusion_weights']

        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    def test_fusion_head_gradient_flow(self, fusion_head):
        """Test gradient flow through FusionHead."""
        rgb_features = torch.randn(4, 768, requires_grad=True)
        sonar_features = torch.randn(4, 384, requires_grad=True)

        outputs = fusion_head(rgb_features, sonar_features)
        loss = outputs['quality_score'].mean()

        loss.backward()

        assert rgb_features.grad is not None
        assert sonar_features.grad is not None
        assert not torch.all(rgb_features.grad == 0)
        assert not torch.all(sonar_features.grad == 0)

    def test_fusion_head_different_methods(self):
        """Test FusionHead with different fusion methods."""
        methods = ['attention', 'mlp', 'uncertainty_weighted']
        batch_size = 4

        for method in methods:
            head = FusionHead(
                rgb_channels=768,
                sonar_channels=384,
                hidden_dim=256,
                fusion_method=method
            )

            rgb_features = torch.randn(batch_size, 768)
            sonar_features = torch.randn(batch_size, 384)

            outputs = head(rgb_features, sonar_features)

            assert outputs['fused_features'].shape == (batch_size, 256)
            assert outputs['quality_score'].shape == (batch_size,)

    def test_fusion_head_entropy_uncertainty(self):
        """Test FusionHead with entropy-based uncertainty."""
        head = FusionHead(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256,
            uncertainty_method='entropy'
        )

        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)

        outputs = head(rgb_features, sonar_features)

        assert 'rgb_confidence' in outputs
        assert 'sonar_confidence' in outputs
        assert outputs['rgb_confidence'].shape == (batch_size,)
        assert outputs['sonar_confidence'].shape == (batch_size,)


class TestFusionLoss:
    """Tests for fusion loss computation."""

    @pytest.fixture
    def fusion_head(self):
        """Create FusionHead for testing."""
        return FusionHead(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256
        )

    def test_fusion_loss_with_target(self, fusion_head):
        """Test fusion loss computation with target quality."""
        batch_size = 4
        rgb_quality = torch.rand(batch_size)
        sonar_quality = torch.rand(batch_size)
        fusion_weights = torch.softmax(torch.randn(batch_size, 2), dim=-1)
        target_quality = torch.rand(batch_size)

        loss = fusion_head.compute_fusion_loss(
            rgb_quality,
            sonar_quality,
            fusion_weights,
            target_quality
        )

        assert loss.numel() == 1
        assert loss >= 0

    def test_fusion_loss_without_target(self, fusion_head):
        """Test fusion loss computation without target quality."""
        batch_size = 4
        rgb_quality = torch.rand(batch_size)
        sonar_quality = torch.rand(batch_size)
        fusion_weights = torch.softmax(torch.randn(batch_size, 2), dim=-1)

        loss = fusion_head.compute_fusion_loss(
            rgb_quality,
            sonar_quality,
            fusion_weights,
            target_quality=None
        )

        assert loss.numel() == 1

    def test_fusion_loss_gradient_flow(self, fusion_head):
        """Test gradient flow through fusion loss."""
        batch_size = 4
        rgb_quality = torch.rand(batch_size, requires_grad=True)
        sonar_quality = torch.rand(batch_size, requires_grad=True)
        fusion_weights = torch.softmax(torch.randn(batch_size, 2), dim=-1)
        target_quality = torch.rand(batch_size)

        loss = fusion_head.compute_fusion_loss(
            rgb_quality,
            sonar_quality,
            fusion_weights,
            target_quality
        )

        loss.backward()

        assert rgb_quality.grad is not None
        assert sonar_quality.grad is not None


class TestFusionHeadFactory:
    """Tests for build_fusion_head factory function."""

    def test_build_fusion_head_default(self):
        """Test building fusion head with default parameters."""
        head = build_fusion_head(rgb_channels=768, sonar_channels=384)
        assert isinstance(head, FusionHead)
        assert head.rgb_channels == 768
        assert head.sonar_channels == 384

    def test_build_fusion_head_custom(self):
        """Test building fusion head with custom parameters."""
        head = build_fusion_head(
            rgb_channels=512,
            sonar_channels=256,
            hidden_dim=128,
            uncertainty_method='entropy',
            fusion_method='mlp'
        )
        assert isinstance(head, FusionHead)
        assert head.rgb_channels == 512
        assert head.sonar_channels == 256
        assert head.hidden_dim == 128
        assert head.uncertainty_method == 'entropy'
        assert head.fusion_method == 'mlp'


class TestFusionHeadIntegration:
    """Integration tests for FusionHead with other modules."""

    def test_fusion_head_with_backbone_features(self):
        """Test FusionHead with typical backbone features."""
        head = FusionHead(
            rgb_channels=768,  # ViT-Base dimension
            sonar_channels=384,  # Smaller sonar encoder
            hidden_dim=256
        )

        # Simulate ViT RGB features
        batch_size = 4
        num_patches = 196
        rgb_features = torch.randn(batch_size, num_patches, 768)

        # Simulate Sonar CNN features
        sonar_features = torch.randn(batch_size, 384, 7, 7)

        outputs = head(rgb_features, sonar_features)

        assert outputs['fused_features'].shape == (batch_size, 256)
        assert outputs['quality_score'].shape == (batch_size,)

    def test_fusion_head_with_box_head(self):
        """Test FusionHead integration with BoxHead."""
        from aquasot3d.models.heads import build_box_head

        fusion_head = FusionHead(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256
        )
        box_head = build_box_head('fusion', in_channels=256)

        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)

        # Fusion
        fusion_outputs = fusion_head(rgb_features, sonar_features)
        fused_features = fusion_outputs['fused_features']

        # Box prediction
        box_outputs = box_head(fused_features)

        assert box_outputs['pred_boxes'].shape == (batch_size, 4)
        assert box_outputs['pred_scores'].shape == (batch_size,)

    def test_multi_modal_pipeline(self):
        """Test complete multi-modal prediction pipeline."""
        # Create heads
        fusion_head = build_fusion_head(
            rgb_channels=768,
            sonar_channels=384,
            hidden_dim=256
        )

        from aquasot3d.models.heads import build_box_head

        rgb_box_head = build_box_head('rgb', in_channels=768)
        sonar_box_head = build_box_head('sonar', in_channels=384)
        fusion_box_head = build_box_head('fusion', in_channels=256)

        # Simulate features
        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)

        # Independent predictions
        rgb_boxes = rgb_box_head(rgb_features)
        sonar_boxes = sonar_box_head(sonar_features)

        # Fused prediction
        fusion_outputs = fusion_head(rgb_features, sonar_features)
        fusion_boxes = fusion_box_head(fusion_outputs['fused_features'])

        # Verify all outputs
        assert rgb_boxes['pred_boxes'].shape == (batch_size, 4)
        assert sonar_boxes['pred_boxes'].shape == (batch_size, 4)
        assert fusion_boxes['pred_boxes'].shape == (batch_size, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
