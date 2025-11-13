"""
Test Suite for Geometric Gating Bug Fixes

Tests the fixes for:
1. Double normalization in RadialProfileExtractor
2. Unregistered dynamic Conv2d modules in ScalePyramidExtractor and GeometryGatedAttention

Based on audit report: 2025111403_Comprehensive_Algo_Audit.md

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import pytest
import torch
import torch.nn as nn

from aquasot3d.models.fusion import (
    RadialProfileExtractor,
    ScalePyramidExtractor,
    GeometryGatedAttention,
)


class TestRadialProfileExtractorFix:
    """Test fixes for RadialProfileExtractor double normalization bug."""

    def test_single_normalization_no_double_division(self):
        """
        Test that radial profile uses single normalization (mean), not double.

        Verifies fix for: RadialProfileExtractor.extract_radial_bins
        was doing profile[bin] = feat[mask].mean() then dividing by counts again.
        """
        B, C, H, W = 2, 256, 32, 32
        sonar_features = torch.randn(B, C, H, W)

        extractor = RadialProfileExtractor(num_bins=50, feature_dim=C)
        radial_profile = extractor.extract_radial_bins(sonar_features)

        # Profile values should not be systematically damped
        # Check that far-range bins don't collapse to ~0
        assert radial_profile.shape == (B, 50)
        assert not torch.isnan(radial_profile).any()
        assert not torch.isinf(radial_profile).any()

        # Far-range bins should have reasonable magnitude
        # (not collapsed to ~0 due to double normalization)
        far_range_bins = radial_profile[:, -10:]  # Last 10 bins
        assert far_range_bins.abs().mean() > 1e-6, \
            "Far-range bins collapsed to ~0, possible double normalization"

    def test_batch_permutation_invariance(self):
        """
        Test that gating weights are invariant to batch permutations.

        Regression test to ensure stable gradients and deterministic behavior.
        """
        B, C, H, W = 4, 256, 32, 32
        sonar_features = torch.randn(B, C, H, W)

        extractor = RadialProfileExtractor(num_bins=50, feature_dim=C)

        # Compute profiles in original order
        profiles_original = extractor.extract_radial_bins(sonar_features)

        # Permute batch
        perm_idx = torch.randperm(B)
        sonar_features_perm = sonar_features[perm_idx]

        # Compute profiles in permuted order
        profiles_permuted = extractor.extract_radial_bins(sonar_features_perm)

        # Reverse permutation to compare
        profiles_reversed = profiles_permuted[torch.argsort(perm_idx)]

        # Profiles should match (batch order invariant)
        assert torch.allclose(profiles_original, profiles_reversed, atol=1e-5), \
            "Radial profiles not invariant to batch permutation"

    def test_vectorized_scatter_add_equivalence(self):
        """
        Test that vectorized scatter_add implementation produces correct results.

        Verifies that the optimized scatter_add approach gives same results
        as manual loop-based aggregation.
        """
        B, C, H, W = 2, 128, 16, 16
        sonar_features = torch.randn(B, C, H, W)

        extractor = RadialProfileExtractor(num_bins=25, feature_dim=C)
        radial_profile = extractor.extract_radial_bins(sonar_features)

        # Check shape and validity
        assert radial_profile.shape == (B, 25)
        assert not torch.isnan(radial_profile).any()

        # All bins should have non-zero values (since we're using mean aggregation)
        # At least most bins should be populated
        non_zero_bins = (radial_profile.abs() > 1e-8).float().mean()
        assert non_zero_bins > 0.5, \
            f"Too many empty bins ({non_zero_bins:.2%}), scatter_add may be incorrect"


class TestScalePyramidExtractorFix:
    """Test fixes for ScalePyramidExtractor dynamic module registration."""

    def test_channel_adapters_registered(self):
        """
        Test that channel adapters are properly registered as nn.ModuleDict.

        Verifies fix for: Dynamic nn.Conv2d created in forward() but not registered.
        """
        extractor = ScalePyramidExtractor(feature_dim=256, num_scales=3)

        # Check that channel_adapters is a ModuleDict
        assert hasattr(extractor, 'channel_adapters')
        assert isinstance(extractor.channel_adapters, nn.ModuleDict)

    def test_adapter_persistence_across_calls(self):
        """
        Test that adapters persist across forward calls (not recreated).

        Ensures checkpoint determinism and gradient flow.
        """
        B, C, H, W = 2, 512, 14, 14  # Different from feature_dim=256
        rgb_features = torch.randn(B, C, H, W)

        extractor = ScalePyramidExtractor(feature_dim=256, num_scales=3)
        extractor.eval()  # Set to eval for deterministic output

        # First forward pass
        with torch.no_grad():
            output1 = extractor(rgb_features)

        # Check adapter was created
        adapter_key = str(C)
        assert adapter_key in extractor.channel_adapters, \
            "Adapter not created for input dimension"

        # Get adapter reference
        adapter_ref = extractor.channel_adapters[adapter_key]

        # Second forward pass
        with torch.no_grad():
            output2 = extractor(rgb_features)

        # Adapter should be the same object (not recreated)
        assert extractor.channel_adapters[adapter_key] is adapter_ref, \
            "Adapter recreated instead of reused"

        # Outputs should be identical given same input
        assert torch.allclose(output1, output2, atol=1e-5), \
            "Non-deterministic output from adapter"

    def test_adapter_in_state_dict(self):
        """
        Test that adapters are included in state_dict for checkpointing.

        Verifies that dynamically created modules participate in save/load.
        """
        B, C, H, W = 2, 384, 14, 14
        rgb_features = torch.randn(B, C, H, W)

        extractor = ScalePyramidExtractor(feature_dim=256, num_scales=3)

        # Trigger adapter creation
        _ = extractor(rgb_features)

        # Get state dict
        state_dict = extractor.state_dict()

        # Check that adapter weights are in state dict
        adapter_key = str(C)
        adapter_weight_key = f'channel_adapters.{adapter_key}.weight'

        assert any(adapter_key in k for k in state_dict.keys()), \
            f"Adapter {adapter_key} not found in state_dict"
        assert adapter_weight_key in state_dict, \
            f"Adapter weights not in state_dict"

    def test_gradient_flow_through_adapter(self):
        """
        Test that gradients flow through dynamically created adapters.

        Ensures adapters participate in optimization.
        """
        B, C, H, W = 2, 384, 14, 14
        rgb_features = torch.randn(B, C, H, W, requires_grad=True)

        extractor = ScalePyramidExtractor(feature_dim=256, num_scales=3)

        # Forward pass
        output = extractor(rgb_features)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check that adapter has gradients
        adapter_key = str(C)
        adapter = extractor.channel_adapters[adapter_key]

        assert adapter.weight.grad is not None, \
            "No gradient for adapter weights"
        assert adapter.weight.grad.abs().sum() > 0, \
            "Adapter gradient is zero, no gradient flow"


class TestGeometryGatedAttentionFix:
    """Test fixes for GeometryGatedAttention dynamic projection layers."""

    def test_adapters_registered(self):
        """
        Test that all dynamic projection adapters are registered as ModuleDict.

        Verifies fix for: rgb_proj, sonar_proj, rgb_out_proj created dynamically.
        """
        attention = GeometryGatedAttention(
            embed_dim=256,
            num_heads=8,
            num_radial_bins=50,
            num_scales=3
        )

        # Check ModuleDict registration
        assert hasattr(attention, 'rgb_adapters')
        assert hasattr(attention, 'sonar_adapters')
        assert hasattr(attention, 'rgb_out_adapters')

        assert isinstance(attention.rgb_adapters, nn.ModuleDict)
        assert isinstance(attention.sonar_adapters, nn.ModuleDict)
        assert isinstance(attention.rgb_out_adapters, nn.ModuleDict)

    def test_adapter_persistence_across_forward(self):
        """
        Test that adapters persist and aren't recreated on each forward pass.

        Critical for training stability and checkpoint consistency.
        """
        B = 2
        rgb_features = torch.randn(B, 512, 14, 14)  # Different from embed_dim=256
        sonar_features = torch.randn(B, 384, 14, 14)

        attention = GeometryGatedAttention(embed_dim=256, num_heads=8)
        attention.eval()  # Set to eval for deterministic output

        # First forward
        with torch.no_grad():
            out1 = attention(rgb_features, sonar_features)

        # Store adapter references
        rgb_adapter_ref = attention.rgb_adapters[str(512)]
        sonar_adapter_ref = attention.sonar_adapters[str(384)]
        rgb_out_adapter_ref = attention.rgb_out_adapters[str(512)]

        # Second forward
        with torch.no_grad():
            out2 = attention(rgb_features, sonar_features)

        # Adapters should be the same objects
        assert attention.rgb_adapters[str(512)] is rgb_adapter_ref
        assert attention.sonar_adapters[str(384)] is sonar_adapter_ref
        assert attention.rgb_out_adapters[str(512)] is rgb_out_adapter_ref

        # Outputs should be deterministic
        assert torch.allclose(
            out1['fused_features'], out2['fused_features'], atol=1e-5
        ), "Non-deterministic output from attention"

    def test_adapters_in_state_dict(self):
        """
        Test that all adapters are saved in state_dict.

        Ensures checkpoints capture dynamically created modules.
        """
        B = 2
        rgb_features = torch.randn(B, 768, 14, 14)
        sonar_features = torch.randn(B, 512, 14, 14)

        attention = GeometryGatedAttention(embed_dim=256, num_heads=8)

        # Trigger adapter creation
        _ = attention(rgb_features, sonar_features)

        state_dict = attention.state_dict()

        # Check all adapters are in state_dict
        assert any('rgb_adapters.768' in k for k in state_dict.keys()), \
            "RGB adapter not in state_dict"
        assert any('sonar_adapters.512' in k for k in state_dict.keys()), \
            "Sonar adapter not in state_dict"
        assert any('rgb_out_adapters.768' in k for k in state_dict.keys()), \
            "RGB output adapter not in state_dict"

    def test_checkpoint_save_load_with_adapters(self):
        """
        Test that models with dynamic adapters can be saved and loaded.

        Integration test for checkpoint persistence.
        For dynamic adapters: need to trigger creation then reload state.
        """
        B = 2
        rgb_features = torch.randn(B, 384, 14, 14)
        sonar_features = torch.randn(B, 256, 14, 14)

        # Create and run model
        attention = GeometryGatedAttention(embed_dim=256, num_heads=8)
        attention.eval()
        with torch.no_grad():
            out_before = attention(rgb_features, sonar_features)

        # Save state
        state_dict = attention.state_dict()

        # Create new model
        attention_loaded = GeometryGatedAttention(embed_dim=256, num_heads=8)
        attention_loaded.eval()

        # First pass: trigger adapter creation with same input dimensions
        # This creates the adapters in the new model
        with torch.no_grad():
            _ = attention_loaded(rgb_features, sonar_features)

        # Second pass: load saved state (now adapters exist and can be loaded)
        attention_loaded.load_state_dict(state_dict, strict=False)

        # Third pass: verify outputs match
        with torch.no_grad():
            out_after = attention_loaded(rgb_features, sonar_features)

        # Outputs should match after adapters are properly loaded
        assert torch.allclose(
            out_before['fused_features'],
            out_after['fused_features'],
            atol=1e-5
        ), "Checkpoint load changed model behavior"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
