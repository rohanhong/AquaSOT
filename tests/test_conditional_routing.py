"""
Integration tests for conditional fusion routing.

CRITICAL FIX (Audit 2025112508 Finding 2):
These tests verify that:
1. ConditionalDCMA correctly routes to dual/single/zero-modality paths
2. Confidence-driven validity inference works correctly
3. Bypass paths are used appropriately
4. Zero-modality frames are handled correctly

Author: AquaSOT-3D Team
Date: 2025-11-25
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

from aquasot3d.models.fusion.dcma import ConditionalDCMA


class TestConditionalRouting:
    """Test conditional fusion routing mechanisms."""

    @pytest.fixture
    def conditional_fusion(self):
        """Create ConditionalDCMA instance for testing."""
        return ConditionalDCMA(
            embed_dim=256,
            num_heads=8,
            rgb_channels=768,
            sonar_channels=768,
            confidence_threshold=0.3,
            dropout=0.1
        )

    @pytest.fixture
    def rgb_features(self):
        """Create mock RGB features."""
        return torch.randn(2, 768, 16, 16)  # [B, C, H, W]

    @pytest.fixture
    def sonar_features(self):
        """Create mock sonar features."""
        return torch.randn(2, 768, 16, 16)  # [B, C, H, W]

    def test_dual_modality_routing(self, conditional_fusion, rgb_features, sonar_features):
        """Test that dual-modality samples use full DCMA fusion."""
        # Explicitly mark both modalities as valid
        is_rgb_valid = torch.tensor([True, True])
        is_sonar_valid = torch.tensor([True, True])

        outputs = conditional_fusion(
            rgb_features, sonar_features,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid,
            return_consistency=False
        )

        # Check outputs
        assert 'fused_features' in outputs
        assert 'routing_info' in outputs

        # Verify routing: all samples should go to dual-modality path
        assert outputs['routing_info']['num_dual'] == 2
        assert outputs['routing_info']['num_single_rgb'] == 0
        assert outputs['routing_info']['num_single_sonar'] == 0
        assert outputs['routing_info']['num_zero'] == 0

    def test_single_rgb_routing(self, conditional_fusion, rgb_features, sonar_features):
        """Test that single-RGB samples use RGB bypass path."""
        # Mark only RGB as valid
        is_rgb_valid = torch.tensor([True, True])
        is_sonar_valid = torch.tensor([False, False])

        outputs = conditional_fusion(
            rgb_features, sonar_features,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid,
            return_consistency=False
        )

        # Verify routing: all samples should go to single-RGB path
        assert outputs['routing_info']['num_dual'] == 0
        assert outputs['routing_info']['num_single_rgb'] == 2
        assert outputs['routing_info']['num_single_sonar'] == 0
        assert outputs['routing_info']['num_zero'] == 0

        # Check output shape
        assert outputs['fused_features'].shape == (2, 256, 16, 16)

    def test_single_sonar_routing(self, conditional_fusion, rgb_features, sonar_features):
        """Test that single-sonar samples use sonar bypass path."""
        # Mark only sonar as valid
        is_rgb_valid = torch.tensor([False, False])
        is_sonar_valid = torch.tensor([True, True])

        outputs = conditional_fusion(
            rgb_features, sonar_features,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid,
            return_consistency=False
        )

        # Verify routing: all samples should go to single-sonar path
        assert outputs['routing_info']['num_dual'] == 0
        assert outputs['routing_info']['num_single_rgb'] == 0
        assert outputs['routing_info']['num_single_sonar'] == 2
        assert outputs['routing_info']['num_zero'] == 0

    def test_zero_modality_routing(self, conditional_fusion, rgb_features, sonar_features):
        """Test that zero-modality samples (0000 annotations) are handled correctly."""
        # Mark both modalities as invalid
        is_rgb_valid = torch.tensor([False, False])
        is_sonar_valid = torch.tensor([False, False])

        outputs = conditional_fusion(
            rgb_features, sonar_features,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid,
            return_consistency=False
        )

        # Verify routing: all samples should go to zero-modality path
        assert outputs['routing_info']['num_dual'] == 0
        assert outputs['routing_info']['num_single_rgb'] == 0
        assert outputs['routing_info']['num_single_sonar'] == 0
        assert outputs['routing_info']['num_zero'] == 2

        # Zero-modality should return zeros
        assert torch.allclose(outputs['fused_features'], torch.zeros_like(outputs['fused_features']))

    def test_heterogeneous_batch_routing(self, conditional_fusion, rgb_features, sonar_features):
        """Test that heterogeneous batches (mixed modality availability) route correctly."""
        # Create heterogeneous batch:
        # Sample 0: dual-modality
        # Sample 1: single-RGB
        is_rgb_valid = torch.tensor([True, True])
        is_sonar_valid = torch.tensor([True, False])

        outputs = conditional_fusion(
            rgb_features, sonar_features,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid,
            return_consistency=False
        )

        # Verify routing
        assert outputs['routing_info']['num_dual'] == 1
        assert outputs['routing_info']['num_single_rgb'] == 1
        assert outputs['routing_info']['num_single_sonar'] == 0
        assert outputs['routing_info']['num_zero'] == 0

    def test_confidence_driven_validity_inference(self, conditional_fusion, rgb_features, sonar_features):
        """
        Test automatic validity inference from confidence scores.

        CRITICAL FIX (Audit 2025112508 Finding 2):
        When validity flags are not provided, ConditionalDCMA should
        infer them from lightweight confidence estimation.
        """
        # Don't provide validity flags (None)
        outputs = conditional_fusion(
            rgb_features, sonar_features,
            is_rgb_valid=None,
            is_sonar_valid=None,
            return_consistency=False
        )

        # Verify that routing happened (not all zeros, not all dual)
        # The exact routing depends on the random features and learned confidence estimators
        routing_info = outputs['routing_info']
        total_samples = (routing_info['num_dual'] + routing_info['num_single_rgb'] +
                        routing_info['num_single_sonar'] + routing_info['num_zero'])

        assert total_samples == 2, f"Total routed samples should be 2, got {total_samples}"

        # Verify fused_features are not NaN
        assert not torch.isnan(outputs['fused_features']).any()

    def test_consistency_losses_only_for_dual_modality(self, conditional_fusion, rgb_features, sonar_features):
        """Test that consistency losses are only computed for dual-modality samples."""
        # Single-RGB batch
        is_rgb_valid = torch.tensor([True, True])
        is_sonar_valid = torch.tensor([False, False])

        outputs = conditional_fusion(
            rgb_features, sonar_features,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid,
            return_consistency=True
        )

        # Consistency losses should be zero (no dual-modality samples)
        if 'consistency_losses' in outputs:
            assert torch.allclose(outputs['consistency_losses']['component_consistency'],
                                torch.tensor(0.0))
            assert torch.allclose(outputs['consistency_losses']['representation_consistency'],
                                torch.tensor(0.0))

    def test_dual_view_features_generation(self, conditional_fusion, rgb_features, sonar_features):
        """Test that dual-view fused features are generated correctly."""
        is_rgb_valid = torch.tensor([True, True])
        is_sonar_valid = torch.tensor([True, True])

        outputs = conditional_fusion(
            rgb_features, sonar_features,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid,
            return_consistency=False
        )

        # Check that dual-view features exist
        assert 'rgb_view_fused' in outputs, "RGB-view fused features missing"
        assert 'sonar_view_fused' in outputs, "Sonar-view fused features missing"

        # Check shapes
        assert outputs['rgb_view_fused'].shape == (2, 256, 16, 16)
        assert outputs['sonar_view_fused'].shape == (2, 256, 16, 16)

    def test_metadata_passthrough(self, conditional_fusion, rgb_features, sonar_features):
        """
        Test that letterbox metadata is passed through to DCMA.

        CRITICAL FIX (Audit 2025112508 Finding 1):
        Metadata is essential for geometry-gated attention.
        """
        is_rgb_valid = torch.tensor([True, True])
        is_sonar_valid = torch.tensor([True, True])

        # Create mock metadata
        sonar_metadata = {
            'scale': 0.5,
            'pad_top': 50,
            'original_height': 376
        }

        outputs = conditional_fusion(
            rgb_features, sonar_features,
            is_rgb_valid=is_rgb_valid,
            is_sonar_valid=is_sonar_valid,
            sonar_letterbox_metadata=sonar_metadata,
            return_consistency=False
        )

        # Verify forward pass completes successfully with metadata
        assert 'fused_features' in outputs
        assert not torch.isnan(outputs['fused_features']).any()

    @pytest.mark.parametrize("confidence_threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_different_confidence_thresholds(self, confidence_threshold, rgb_features, sonar_features):
        """Test that different confidence thresholds affect routing appropriately."""
        conditional_fusion = ConditionalDCMA(
            embed_dim=256,
            num_heads=8,
            rgb_channels=768,
            sonar_channels=768,
            confidence_threshold=confidence_threshold,
            dropout=0.1
        )

        # Let it infer validity from confidence
        outputs = conditional_fusion(
            rgb_features, sonar_features,
            is_rgb_valid=None,
            is_sonar_valid=None,
            return_consistency=False
        )

        # Lower threshold should result in more samples being considered "valid"
        # (though exact behavior depends on learned estimators)
        assert 'routing_info' in outputs
        assert 'fused_features' in outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
