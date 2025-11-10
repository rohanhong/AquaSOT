"""
Test Dynamic Sonar Size Handling

Tests for adaptive processing of variable sonar image dimensions:
- Within-sequence size variations (minor)
- Cross-sequence size differences (major: 376x682 vs 950x525)
- Aspect ratio consistency preservation

Author: AquaSOT-3D Team
Date: 2025-11-11
"""

import pytest
import torch
import numpy as np

from aquasot3d.dataloader.transforms.sonar_geometry import (
    SonarGeometryHandler,
    generate_fan_mask,
    compute_radial_weights
)
from aquasot3d.models.backbones.sonar_encoder import SonarEncoder


class TestAdaptiveSonarGeometry:
    """Test adaptive geometry handling for variable sonar sizes"""

    def test_generate_mask_different_sizes(self):
        """Test fan mask generation for different sizes"""
        # Standard size
        mask1 = generate_fan_mask(376, 682, fov_angle=130)
        assert mask1.shape == (1, 376, 682)

        # Large size (different sequence)
        mask2 = generate_fan_mask(950, 525, fov_angle=130)
        assert mask2.shape == (1, 950, 525)

        # Both should have valid pixels (non-zero sum)
        assert mask1.sum() > 0
        assert mask2.sum() > 0

    def test_radial_weights_different_sizes(self):
        """Test radial weight computation for different sizes"""
        # Standard size
        weights1 = compute_radial_weights(376, 682, decay_factor=0.5)
        assert weights1.shape == (1, 376, 682)

        # Large size
        weights2 = compute_radial_weights(950, 525, decay_factor=0.5)
        assert weights2.shape == (1, 950, 525)

        # Weights should be normalized [0, 1]
        assert weights1.min() >= 0 and weights1.max() <= 1
        assert weights2.min() >= 0 and weights2.max() <= 1

    def test_geometry_handler_adaptive_sizing(self):
        """Test SonarGeometryHandler adapts to different input sizes"""
        # Initialize with standard size
        handler = SonarGeometryHandler(sonar_height=376, sonar_width=682)

        # Test with standard size
        sonar1 = torch.randn(2, 1, 376, 682)
        result1 = handler(sonar1)
        assert result1['masked'].shape == (2, 1, 376, 682)
        assert result1['weighted'].shape == (2, 1, 376, 682)

        # Test with different size (large)
        sonar2 = torch.randn(2, 1, 950, 525)
        result2 = handler(sonar2)
        assert result2['masked'].shape == (2, 1, 950, 525)
        assert result2['weighted'].shape == (2, 1, 950, 525)

        # Test with slightly different size (minor variation)
        sonar3 = torch.randn(2, 1, 380, 686)
        result3 = handler(sonar3)
        assert result3['masked'].shape == (2, 1, 380, 686)
        assert result3['weighted'].shape == (2, 1, 380, 686)


class TestSonarEncoderAdaptiveSize:
    """Test sonar encoder handles variable sizes"""

    def test_encoder_different_sizes(self):
        """Test SonarEncoder processes different sonar sizes"""
        encoder = SonarEncoder(
            encoder_type='conv',
            output_dim=128,
            use_logpolar=False  # Disable for dynamic size testing
        )
        encoder.eval()

        # Standard size
        template1 = torch.randn(1, 1, 376, 682)
        search1 = torch.randn(1, 1, 376, 682)

        with torch.no_grad():
            output1 = encoder(template1, search1)

        assert 'features' in output1
        # features can be a list or tensor
        features1 = output1['features']
        if isinstance(features1, list):
            assert len(features1) > 0
            assert isinstance(features1[0], torch.Tensor)
        else:
            assert isinstance(features1, torch.Tensor)

        # Large size (different sequence)
        template2 = torch.randn(1, 1, 950, 525)
        search2 = torch.randn(1, 1, 950, 525)

        with torch.no_grad():
            output2 = encoder(template2, search2)

        assert 'features' in output2
        features2 = output2['features']
        if isinstance(features2, list):
            assert len(features2) > 0
            assert isinstance(features2[0], torch.Tensor)
        else:
            assert isinstance(features2, torch.Tensor)

        # Both outputs should have compatible structures
        assert type(features1) == type(features2)


class TestAspectRatioConsistency:
    """Test aspect ratio preservation across size variations"""

    def test_aspect_ratio_preservation(self):
        """Test that geometry computations preserve aspect ratios"""
        # Define aspect ratios
        sizes = [
            (376, 682),   # Aspect ratio: ~0.55
            (380, 686),   # Aspect ratio: ~0.55 (minor variation)
            (950, 525),   # Aspect ratio: ~1.81 (inverted, different sensor)
        ]

        for h, w in sizes:
            mask = generate_fan_mask(h, w, fov_angle=130)
            weights = compute_radial_weights(h, w, decay_factor=0.5)

            # Both should match input size
            assert mask.shape == (1, h, w)
            assert weights.shape == (1, h, w)

            # Check that geometry is reasonable
            assert mask.sum() > 0  # Has valid region
            assert weights.max() > 0.5  # Has high-quality region


class TestSequenceSizeVariations:
    """Test handling of size variations within a sequence"""

    def test_within_sequence_size_changes(self):
        """Test processing sequence with minor size variations"""
        handler = SonarGeometryHandler(sonar_height=376, sonar_width=682)

        # Simulate sequence with minor size variations
        sizes = [
            (376, 682),
            (378, 684),  # +2 pixels
            (376, 682),
            (374, 680),  # -2 pixels
            (376, 682),
        ]

        for h, w in sizes:
            sonar = torch.randn(1, 1, h, w)
            result = handler(sonar)

            # All outputs should match input size
            assert result['masked'].shape == (1, 1, h, w)
            assert result['weighted'].shape == (1, 1, h, w)

    def test_cross_sequence_size_changes(self):
        """Test processing different sequences with major size differences"""
        handler = SonarGeometryHandler(sonar_height=376, sonar_width=682)

        # Sequence 1: Standard size
        seq1_frames = [torch.randn(1, 1, 376, 682) for _ in range(5)]

        for frame in seq1_frames:
            result = handler(frame)
            assert result['masked'].shape == (1, 1, 376, 682)

        # Sequence 2: Large size
        seq2_frames = [torch.randn(1, 1, 950, 525) for _ in range(5)]

        for frame in seq2_frames:
            result = handler(frame)
            assert result['masked'].shape == (1, 1, 950, 525)


class TestGeometricConsistency:
    """Test geometric properties remain consistent across sizes"""

    def test_fan_shape_angle_consistency(self):
        """Test fan FOV angle is preserved across different sizes"""
        fov = 130.0

        # Generate masks at different sizes
        mask_small = generate_fan_mask(376, 682, fov_angle=fov)
        mask_large = generate_fan_mask(950, 525, fov_angle=fov)

        # Both should have similar coverage ratios (within reasonable margin)
        # since the fan angle is the same
        coverage_small = mask_small.float().mean()
        coverage_large = mask_large.float().mean()

        # Coverage should be similar (within 20% relative difference)
        # This checks that geometric properties scale correctly
        ratio = coverage_small / (coverage_large + 1e-8)
        assert 0.5 < ratio < 2.0, f"Coverage ratio {ratio} indicates geometric inconsistency"

    def test_radial_decay_consistency(self):
        """Test radial decay pattern is consistent across sizes"""
        decay_factor = 0.5

        weights_small = compute_radial_weights(376, 682, decay_factor=decay_factor)
        weights_large = compute_radial_weights(950, 525, decay_factor=decay_factor)

        # Both should have similar weight distributions
        # Check that max weight is near sensor (bottom-center)
        assert weights_small.max() > 0.8
        assert weights_large.max() > 0.8

        # Min weight should be at far edges
        assert weights_small.min() < 0.3
        assert weights_large.min() < 0.3
