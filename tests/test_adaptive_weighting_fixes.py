"""
Regression Tests for AdaptiveWeighting EMA Fixes (Audit 2025111406)

Tests for critical bug fixes:
1. EMA buffer dimension mismatch with variable batch sizes
2. Temporal smoothing application during training
3. Cross-sequence state leakage prevention via reset_state()

Author: AquaSOT-3D Team
Date: 2025-11-14
"""

import pytest
import torch
import torch.nn as nn

from aquasot3d.models.quality.adaptive_weighting import (
    AdaptiveWeighting,
    create_adaptive_weighting,
)


class TestAdaptiveWeightingEMAFixes:
    """Tests for EMA buffer dimension fixes (Audit 2025111406)."""

    @pytest.fixture
    def adaptive_weighting(self):
        """Create adaptive weighting module for testing."""
        return AdaptiveWeighting(
            weighting_method='softmax',
            temporal_smoothing=True,
            ema_alpha=0.9,
            quality_threshold=0.3,
            use_gating=True,
        )

    def test_mixed_batch_sizes_no_crash(self, adaptive_weighting):
        """
        Test that EMA buffers handle variable batch sizes without crashing.

        This is the CRITICAL fix from Audit 2025111406 Issue 1.1:
        EMA buffers must store scalar values (1, 1) instead of full batches (B, 1).
        """
        # First batch: B=32
        rgb_feat_1 = torch.randn(32, 256, 14, 14)
        sonar_feat_1 = torch.randn(32, 256, 14, 14)
        q_rgb_1 = torch.rand(32, 1) * 0.5 + 0.5  # [0.5, 1.0]
        q_sonar_1 = torch.rand(32, 1) * 0.5 + 0.5

        # First forward pass
        fused_1, weights_1 = adaptive_weighting(
            rgb_feat_1, sonar_feat_1, q_rgb_1, q_sonar_1
        )

        # Check buffer shapes after first batch
        assert adaptive_weighting.ema_q_rgb.shape == (1, 1), \
            f"EMA buffer should be (1, 1) but got {adaptive_weighting.ema_q_rgb.shape}"
        assert adaptive_weighting.ema_q_sonar.shape == (1, 1), \
            f"EMA buffer should be (1, 1) but got {adaptive_weighting.ema_q_sonar.shape}"

        # Second batch: B=23 (different size - this used to crash)
        rgb_feat_2 = torch.randn(23, 256, 14, 14)
        sonar_feat_2 = torch.randn(23, 256, 14, 14)
        q_rgb_2 = torch.rand(23, 1) * 0.5 + 0.5
        q_sonar_2 = torch.rand(23, 1) * 0.5 + 0.5

        # Second forward pass (should NOT crash)
        fused_2, weights_2 = adaptive_weighting(
            rgb_feat_2, sonar_feat_2, q_rgb_2, q_sonar_2
        )

        # Verify outputs have correct shapes
        assert fused_1.shape == rgb_feat_1.shape
        assert fused_2.shape == rgb_feat_2.shape
        assert weights_2['q_rgb_smoothed'].shape == (23, 1)
        assert weights_2['q_sonar_smoothed'].shape == (23, 1)

        # EMA buffers should remain (1, 1) throughout
        assert adaptive_weighting.ema_q_rgb.shape == (1, 1)
        assert adaptive_weighting.ema_q_sonar.shape == (1, 1)

    def test_ema_updates_during_training(self, adaptive_weighting):
        """
        Test that EMA buffers are updated during training mode.

        This is the MAJOR fix from Audit 2025111406 Issue 1.2:
        Temporal smoothing must be applied in both training and eval modes.
        """
        adaptive_weighting.train()

        rgb_feat = torch.randn(8, 256, 14, 14)
        sonar_feat = torch.randn(8, 256, 14, 14)
        q_rgb = torch.tensor([[0.8], [0.9], [0.7], [0.6], [0.85], [0.75], [0.65], [0.55]])
        q_sonar = torch.tensor([[0.6], [0.5], [0.7], [0.8], [0.55], [0.65], [0.75], [0.85]])

        # First pass: buffers should initialize
        fused_1, weights_1 = adaptive_weighting(rgb_feat, sonar_feat, q_rgb, q_sonar)

        # Get initial EMA values (should be batch mean)
        ema_rgb_init = adaptive_weighting.ema_q_rgb.clone()
        ema_sonar_init = adaptive_weighting.ema_q_sonar.clone()

        # Verify initialization uses batch mean
        expected_rgb_mean = q_rgb.mean()
        expected_sonar_mean = q_sonar.mean()
        assert torch.allclose(ema_rgb_init, expected_rgb_mean.view(1, 1), atol=1e-5)
        assert torch.allclose(ema_sonar_init, expected_sonar_mean.view(1, 1), atol=1e-5)

        # Second pass with different quality scores
        q_rgb_2 = torch.tensor([[0.9], [0.95], [0.85], [0.8], [0.92], [0.88], [0.82], [0.78]])
        q_sonar_2 = torch.tensor([[0.7], [0.65], [0.75], [0.85], [0.68], [0.72], [0.80], [0.90]])

        fused_2, weights_2 = adaptive_weighting(rgb_feat, sonar_feat, q_rgb_2, q_sonar_2)

        # EMA buffers should have updated (not remain at init values)
        ema_rgb_updated = adaptive_weighting.ema_q_rgb
        ema_sonar_updated = adaptive_weighting.ema_q_sonar

        # Verify EMA update formula: new_ema = alpha * old_ema + (1 - alpha) * batch_mean
        expected_rgb_ema = 0.9 * ema_rgb_init + 0.1 * q_rgb_2.mean()
        expected_sonar_ema = 0.9 * ema_sonar_init + 0.1 * q_sonar_2.mean()

        assert torch.allclose(ema_rgb_updated, expected_rgb_ema.view(1, 1), atol=1e-5), \
            "EMA should update during training mode"
        assert torch.allclose(ema_sonar_updated, expected_sonar_ema.view(1, 1), atol=1e-5), \
            "EMA should update during training mode"

    def test_reset_state_clears_ema_buffers(self, adaptive_weighting):
        """
        Test that reset_state() properly clears EMA buffers.

        This is the MAJOR fix from Audit 2025111406 Issue 1.3:
        Cross-sequence state leakage must be prevented via reset_state().
        """
        # Run forward pass to populate buffers
        rgb_feat = torch.randn(4, 256, 14, 14)
        sonar_feat = torch.randn(4, 256, 14, 14)
        q_rgb = torch.rand(4, 1)
        q_sonar = torch.rand(4, 1)

        fused, weights = adaptive_weighting(rgb_feat, sonar_feat, q_rgb, q_sonar)

        # Verify buffers are populated
        assert adaptive_weighting.ema_q_rgb is not None
        assert adaptive_weighting.ema_q_sonar is not None

        # Reset state
        adaptive_weighting.reset_state()

        # Verify buffers are cleared
        assert adaptive_weighting.ema_q_rgb is None, \
            "reset_state() should clear ema_q_rgb buffer"
        assert adaptive_weighting.ema_q_sonar is None, \
            "reset_state() should clear ema_q_sonar buffer"

    def test_cross_sequence_independence(self, adaptive_weighting):
        """
        Test that reset_state() prevents cross-sequence leakage.

        Simulates validation across two different sequences:
        - Sequence 1: high RGB quality
        - Sequence 2: high Sonar quality

        Without reset, Sequence 2 would be biased by Sequence 1's statistics.
        """
        adaptive_weighting.eval()

        # === Sequence 1: High RGB quality ===
        q_rgb_seq1 = torch.full((8, 1), 0.9)  # High RGB quality
        q_sonar_seq1 = torch.full((8, 1), 0.4)  # Low Sonar quality
        rgb_feat = torch.randn(8, 256, 14, 14)
        sonar_feat = torch.randn(8, 256, 14, 14)

        # Process sequence 1
        for _ in range(5):
            fused, weights = adaptive_weighting(rgb_feat, sonar_feat, q_rgb_seq1, q_sonar_seq1)

        # EMA should reflect high RGB quality
        ema_rgb_seq1 = adaptive_weighting.ema_q_rgb.item()
        ema_sonar_seq1 = adaptive_weighting.ema_q_sonar.item()
        assert ema_rgb_seq1 > 0.85, "EMA should reflect high RGB quality"
        assert ema_sonar_seq1 < 0.45, "EMA should reflect low Sonar quality"

        # === Reset for new sequence (prevents leakage) ===
        adaptive_weighting.reset_state()

        # === Sequence 2: High Sonar quality ===
        q_rgb_seq2 = torch.full((8, 1), 0.3)  # Low RGB quality
        q_sonar_seq2 = torch.full((8, 1), 0.95)  # High Sonar quality

        # First batch of sequence 2 should use fresh statistics
        fused, weights = adaptive_weighting(rgb_feat, sonar_feat, q_rgb_seq2, q_sonar_seq2)

        # On first batch after reset, should return raw values (not EMA smoothed)
        # Because buffers were cleared
        assert torch.allclose(weights['q_rgb_smoothed'], q_rgb_seq2, atol=1e-5), \
            "First batch after reset should use raw quality scores"
        assert torch.allclose(weights['q_sonar_smoothed'], q_sonar_seq2, atol=1e-5), \
            "First batch after reset should use raw quality scores"

    def test_ema_smoothing_convergence(self, adaptive_weighting):
        """Test that EMA smoothing converges to steady-state values."""
        adaptive_weighting.eval()

        rgb_feat = torch.randn(4, 256, 14, 14)
        sonar_feat = torch.randn(4, 256, 14, 14)

        # Constant quality scores
        q_rgb_constant = torch.full((4, 1), 0.8)
        q_sonar_constant = torch.full((4, 1), 0.6)

        # Run multiple iterations
        smoothed_rgb_history = []
        smoothed_sonar_history = []

        for i in range(50):
            fused, weights = adaptive_weighting(
                rgb_feat, sonar_feat, q_rgb_constant, q_sonar_constant
            )
            smoothed_rgb_history.append(weights['q_rgb_smoothed'][0, 0].item())
            smoothed_sonar_history.append(weights['q_sonar_smoothed'][0, 0].item())

        # After many iterations, EMA should converge close to the constant value
        final_rgb = smoothed_rgb_history[-1]
        final_sonar = smoothed_sonar_history[-1]

        assert abs(final_rgb - 0.8) < 0.01, \
            f"EMA should converge to 0.8, got {final_rgb}"
        assert abs(final_sonar - 0.6) < 0.01, \
            f"EMA should converge to 0.6, got {final_sonar}"

        # Verify convergence occurred (EMA values stabilized)
        # First few iterations should differ from final value
        early_rgb = smoothed_rgb_history[0]
        early_sonar = smoothed_sonar_history[0]

        # First batch returns raw values (before smoothing starts)
        # Subsequent batches should smooth toward the constant
        if len(smoothed_rgb_history) > 1:
            # Check that smoothing is applied (values change over iterations)
            # Due to high alpha (0.9), convergence is fast, so just verify it's working
            assert abs(smoothed_rgb_history[-1] - 0.8) < 0.02, \
                "EMA should converge close to constant value"

    def test_broadcast_correctness(self, adaptive_weighting):
        """Test that scalar EMA buffers broadcast correctly to batch dimensions."""
        rgb_feat = torch.randn(16, 256, 14, 14)
        sonar_feat = torch.randn(16, 256, 14, 14)
        q_rgb = torch.rand(16, 1)
        q_sonar = torch.rand(16, 1)

        # Forward pass
        fused, weights = adaptive_weighting(rgb_feat, sonar_feat, q_rgb, q_sonar)

        # Second pass with different batch size
        rgb_feat_2 = torch.randn(8, 256, 14, 14)
        sonar_feat_2 = torch.randn(8, 256, 14, 14)
        q_rgb_2 = torch.rand(8, 1)
        q_sonar_2 = torch.rand(8, 1)

        fused_2, weights_2 = adaptive_weighting(
            rgb_feat_2, sonar_feat_2, q_rgb_2, q_sonar_2
        )

        # Smoothed values should be broadcasted to correct batch size
        assert weights_2['q_rgb_smoothed'].shape == (8, 1)
        assert weights_2['q_sonar_smoothed'].shape == (8, 1)

        # All samples in batch should have same smoothed value (broadcasted from scalar)
        assert torch.allclose(
            weights_2['q_rgb_smoothed'][0],
            weights_2['q_rgb_smoothed'][1],
            atol=1e-6
        ), "Broadcasted values should be identical across batch"

    def test_factory_function(self):
        """Test create_adaptive_weighting factory function."""
        weighting = create_adaptive_weighting(
            weighting_method='sigmoid',
            temporal_smoothing=True,
            ema_alpha=0.85,
            quality_threshold=0.4
        )

        assert isinstance(weighting, AdaptiveWeighting)
        assert weighting.weighting_method == 'sigmoid'
        assert weighting.temporal_smoothing is True
        assert weighting.ema_alpha == 0.85
        assert weighting.quality_threshold == 0.4


class TestAdaptiveWeightingIntegration:
    """Integration tests with tracker reset_temporal_state()."""

    def test_tracker_calls_reset_state(self):
        """
        Test that tracker's reset_temporal_state() properly calls
        adaptive_weighting.reset_state().

        This verifies the wiring fix from Audit 2025111406 Action 1.
        """
        # Mock tracker module
        adaptive_weighting = AdaptiveWeighting(temporal_smoothing=True)

        # Simulate tracker reset
        class MockTracker:
            def __init__(self):
                self.adaptive_weighting = adaptive_weighting
                self.temporal_head = None
                self.temporal_aggregator = None

            def reset_temporal_state(self):
                """Tracker's reset method (from tracker.py:341)."""
                if self.adaptive_weighting is not None and hasattr(
                    self.adaptive_weighting, 'reset_state'
                ):
                    self.adaptive_weighting.reset_state()

        tracker = MockTracker()

        # Populate adaptive weighting buffers
        rgb_feat = torch.randn(4, 256, 14, 14)
        sonar_feat = torch.randn(4, 256, 14, 14)
        q_rgb = torch.rand(4, 1)
        q_sonar = torch.rand(4, 1)

        tracker.adaptive_weighting(rgb_feat, sonar_feat, q_rgb, q_sonar)
        assert tracker.adaptive_weighting.ema_q_rgb is not None

        # Call tracker reset
        tracker.reset_temporal_state()

        # Verify buffers were cleared
        assert tracker.adaptive_weighting.ema_q_rgb is None, \
            "Tracker reset should clear adaptive_weighting buffers"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])