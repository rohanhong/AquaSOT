"""
Unit Tests for Geometric Cyclic Projector

Tests for GeometricCyclicProjector, RGBProjector, SonarProjector, and utilities.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from aquasot3d.models.projector import (
    GeometricCyclicProjector,
    RGBProjector,
    SonarProjector,
    build_geometric_projector,
)


class TestGeometricCyclicProjector:
    """Tests for GeometricCyclicProjector module."""

    @pytest.fixture
    def projector_mlp(self):
        """Create MLP projector for testing."""
        return GeometricCyclicProjector(
            input_dim=0,
            hidden_dim=256,
            projection_type='mlp'
        )

    @pytest.fixture
    def projector_linear(self):
        """Create linear projector for testing."""
        return GeometricCyclicProjector(
            input_dim=0,
            hidden_dim=256,
            projection_type='linear'
        )

    @pytest.fixture
    def projector_perspective(self):
        """Create perspective projector for testing."""
        return GeometricCyclicProjector(
            projection_type='perspective'
        )

    def test_mlp_projector_initialization(self, projector_mlp):
        """Test MLP projector initialization."""
        assert isinstance(projector_mlp, GeometricCyclicProjector)
        assert projector_mlp.projection_type == 'mlp'
        assert projector_mlp.hidden_dim == 256

    def test_linear_projector_initialization(self, projector_linear):
        """Test linear projector initialization."""
        assert isinstance(projector_linear, GeometricCyclicProjector)
        assert projector_linear.projection_type == 'linear'

    def test_perspective_projector_initialization(self, projector_perspective):
        """Test perspective projector initialization."""
        assert isinstance(projector_perspective, GeometricCyclicProjector)
        assert projector_perspective.projection_type == 'perspective'
        assert hasattr(projector_perspective, 'focal_length')
        assert hasattr(projector_perspective, 'principal_point')

    def test_3d_to_2d_projection_mlp(self, projector_mlp):
        """Test 3D to 2D projection with MLP."""
        batch_size = 4
        center_3d = torch.randn(batch_size, 3)

        bbox_2d = projector_mlp.project_3d_to_2d(center_3d)

        assert bbox_2d.shape == (batch_size, 4)
        # Width and height should be positive
        assert torch.all(bbox_2d[:, 2] > 0)
        assert torch.all(bbox_2d[:, 3] > 0)

    def test_3d_to_2d_projection_linear(self, projector_linear):
        """Test 3D to 2D projection with linear."""
        batch_size = 4
        center_3d = torch.randn(batch_size, 3)

        bbox_2d = projector_linear.project_3d_to_2d(center_3d)

        assert bbox_2d.shape == (batch_size, 4)

    def test_3d_to_2d_projection_perspective(self, projector_perspective):
        """Test 3D to 2D projection with perspective."""
        batch_size = 4
        center_3d = torch.randn(batch_size, 3)
        center_3d[:, 2] = center_3d[:, 2].abs() + 0.5  # Ensure positive Z

        bbox_2d = projector_perspective.project_3d_to_2d(center_3d)

        assert bbox_2d.shape == (batch_size, 4)

    def test_2d_to_3d_projection_mlp(self, projector_mlp):
        """Test 2D to 3D projection with MLP."""
        batch_size = 4
        bbox_2d = torch.randn(batch_size, 4)
        bbox_2d[:, 2:] = bbox_2d[:, 2:].abs() + 0.1  # Positive width/height

        center_3d = projector_mlp.project_2d_to_3d(bbox_2d)

        assert center_3d.shape == (batch_size, 3)

    def test_2d_to_3d_projection_perspective(self, projector_perspective):
        """Test 2D to 3D projection with perspective."""
        batch_size = 4
        bbox_2d = torch.randn(batch_size, 4)
        bbox_2d[:, 2:] = bbox_2d[:, 2:].abs() + 0.1

        center_3d = projector_perspective.project_2d_to_3d(bbox_2d)

        assert center_3d.shape == (batch_size, 3)

    def test_forward_without_cyclic_loss(self, projector_mlp):
        """Test forward pass without cyclic loss."""
        batch_size = 4
        center_3d = torch.randn(batch_size, 3)

        outputs = projector_mlp(center_3d, compute_cyclic_loss=False)

        assert 'bbox_2d' in outputs
        assert outputs['bbox_2d'].shape == (batch_size, 4)
        assert 'cyclic_loss' not in outputs

    def test_forward_with_cyclic_loss(self, projector_mlp):
        """Test forward pass with cyclic loss."""
        batch_size = 4
        center_3d = torch.randn(batch_size, 3)

        outputs = projector_mlp(center_3d, compute_cyclic_loss=True)

        assert 'bbox_2d' in outputs
        assert 'center_3d_reconstructed' in outputs
        assert 'cyclic_loss' in outputs
        assert outputs['bbox_2d'].shape == (batch_size, 4)
        assert outputs['center_3d_reconstructed'].shape == (batch_size, 3)
        assert outputs['cyclic_loss'].numel() == 1
        assert outputs['cyclic_loss'] >= 0

    def test_cyclic_consistency(self, projector_mlp):
        """Test cyclic consistency: 3D→2D→3D should be close to identity."""
        batch_size = 4
        center_3d_original = torch.randn(batch_size, 3)

        # Forward: 3D → 2D
        bbox_2d = projector_mlp.project_3d_to_2d(center_3d_original)

        # Backward: 2D → 3D
        center_3d_reconstructed = projector_mlp.project_2d_to_3d(bbox_2d)

        # Check shapes
        assert center_3d_reconstructed.shape == center_3d_original.shape

        # Cyclic loss should be finite (not checking exact value as it depends on initialization)
        cyclic_loss = F.l1_loss(center_3d_reconstructed, center_3d_original)
        assert torch.isfinite(cyclic_loss)

    def test_projection_loss_l1(self, projector_mlp):
        """Test L1 projection loss."""
        pred_bbox = torch.randn(4, 4)
        target_bbox = torch.randn(4, 4)

        loss = projector_mlp.compute_projection_loss(pred_bbox, target_bbox, loss_type='l1')

        assert loss.numel() == 1
        assert loss >= 0

    def test_projection_loss_perfect_match(self, projector_mlp):
        """Test projection loss for perfect match."""
        bbox = torch.randn(4, 4)

        loss = projector_mlp.compute_projection_loss(bbox, bbox, loss_type='l1')

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_projector_with_context(self):
        """Test projector with context features."""
        projector = GeometricCyclicProjector(
            input_dim=256,
            hidden_dim=256,
            use_context=True,
            projection_type='mlp'
        )

        batch_size = 4
        center_3d = torch.randn(batch_size, 3)
        context = torch.randn(batch_size, 256)

        bbox_2d = projector.project_3d_to_2d(center_3d, context)

        assert bbox_2d.shape == (batch_size, 4)

    def test_gradient_flow(self, projector_mlp):
        """Test gradient flow through projector."""
        center_3d = torch.randn(4, 3, requires_grad=True)

        outputs = projector_mlp(center_3d, compute_cyclic_loss=True)
        loss = outputs['cyclic_loss']

        loss.backward()

        assert center_3d.grad is not None
        assert not torch.all(center_3d.grad == 0)


class TestSpecializedProjectors:
    """Tests for specialized projector variants."""

    def test_rgb_projector_initialization(self):
        """Test RGBProjector initialization."""
        projector = RGBProjector()

        assert isinstance(projector, RGBProjector)
        assert isinstance(projector, GeometricCyclicProjector)
        # RGB defaults to perspective
        assert projector.projection_type == 'perspective'

    def test_sonar_projector_initialization(self):
        """Test SonarProjector initialization."""
        projector = SonarProjector()

        assert isinstance(projector, SonarProjector)
        assert isinstance(projector, GeometricCyclicProjector)
        # Sonar defaults to MLP
        assert projector.projection_type == 'mlp'

    def test_rgb_projector_forward(self):
        """Test RGBProjector forward pass."""
        projector = RGBProjector()

        batch_size = 4
        center_3d = torch.randn(batch_size, 3)
        center_3d[:, 2] = center_3d[:, 2].abs() + 0.5  # Positive Z

        outputs = projector(center_3d, compute_cyclic_loss=True)

        assert 'bbox_2d' in outputs
        assert 'cyclic_loss' in outputs

    def test_sonar_projector_forward(self):
        """Test SonarProjector forward pass."""
        projector = SonarProjector()

        batch_size = 4
        center_3d = torch.randn(batch_size, 3)

        outputs = projector(center_3d, compute_cyclic_loss=True)

        assert 'bbox_2d' in outputs
        assert 'cyclic_loss' in outputs


class TestProjectorFactory:
    """Tests for build_geometric_projector factory function."""

    def test_build_rgb_projector(self):
        """Test building RGB projector."""
        projector = build_geometric_projector('rgb')
        assert isinstance(projector, RGBProjector)

    def test_build_sonar_projector(self):
        """Test building Sonar projector."""
        projector = build_geometric_projector('sonar')
        assert isinstance(projector, SonarProjector)

    def test_build_generic_projector(self):
        """Test building generic projector."""
        projector = build_geometric_projector('generic')
        assert isinstance(projector, GeometricCyclicProjector)
        assert not isinstance(projector, (RGBProjector, SonarProjector))

    def test_build_projector_with_kwargs(self):
        """Test building projector with custom parameters."""
        projector = build_geometric_projector('rgb', hidden_dim=128)
        assert projector.hidden_dim == 128


class TestProjectorIntegration:
    """Integration tests for projectors with other modules."""

    def test_projector_with_weak3d_head(self):
        """Test projector with Weak3DHead output."""
        from aquasot3d.models.heads import build_weak3d_head

        weak3d_head = build_weak3d_head(rgb_feat_dim=768, sonar_feat_dim=384)
        rgb_projector = build_geometric_projector('rgb')
        sonar_projector = build_geometric_projector('sonar')

        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)
        sonar_radial = torch.randn(batch_size, 50)

        # Predict 3D center
        weak3d_outputs = weak3d_head(rgb_features, sonar_features, sonar_radial)
        center_3d = weak3d_outputs['center_3d']

        # Project to 2D
        rgb_bbox = rgb_projector(center_3d, compute_cyclic_loss=True)
        sonar_bbox = sonar_projector(center_3d, compute_cyclic_loss=True)

        assert rgb_bbox['bbox_2d'].shape == (batch_size, 4)
        assert sonar_bbox['bbox_2d'].shape == (batch_size, 4)
        assert 'cyclic_loss' in rgb_bbox
        assert 'cyclic_loss' in sonar_bbox

    def test_complete_pipeline(self):
        """Test complete pipeline: features → 3D → 2D projection."""
        from aquasot3d.models.heads import build_weak3d_head, build_box_head

        # Create modules
        weak3d_head = build_weak3d_head(rgb_feat_dim=768, sonar_feat_dim=384)
        rgb_box_head = build_box_head('rgb', in_channels=768)
        rgb_projector = build_geometric_projector('rgb')

        batch_size = 4
        rgb_features = torch.randn(batch_size, 768)
        sonar_features = torch.randn(batch_size, 384)
        sonar_radial = torch.randn(batch_size, 50)

        # Direct 2D prediction
        direct_bbox = rgb_box_head(rgb_features)

        # Indirect 2D prediction via 3D
        weak3d_outputs = weak3d_head(rgb_features, sonar_features, sonar_radial)
        projected_bbox = rgb_projector(weak3d_outputs['center_3d'])

        # Both should produce valid 2D boxes
        assert direct_bbox['pred_boxes'].shape == (batch_size, 4)
        assert projected_bbox['bbox_2d'].shape == (batch_size, 4)

    def test_multi_modal_projection(self):
        """Test separate RGB and Sonar projections from same 3D center."""
        rgb_proj = build_geometric_projector('rgb')
        sonar_proj = build_geometric_projector('sonar')

        batch_size = 4
        center_3d = torch.randn(batch_size, 3)
        center_3d[:, 2] = center_3d[:, 2].abs() + 0.5  # Positive Z

        rgb_outputs = rgb_proj(center_3d, compute_cyclic_loss=True)
        sonar_outputs = sonar_proj(center_3d, compute_cyclic_loss=True)

        # Both modalities project the same 3D center
        assert rgb_outputs['bbox_2d'].shape == (batch_size, 4)
        assert sonar_outputs['bbox_2d'].shape == (batch_size, 4)

        # Different projection types may yield different 2D boxes
        # (this is expected due to different viewpoints)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
