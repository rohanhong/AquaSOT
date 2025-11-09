"""
Comprehensive Test Suite for Temporal Modules (Phase 5)

Tests all components of the temporal modeling framework:
- Vision-Mamba Head (mamba_head.py)
- Temporal Aggregator (temporal_aggregator.py)
- Transformer Head (transformer_head.py)

Run with: pytest tests/test_temporal_modules.py -v
"""

import pytest
import torch
import torch.nn as nn
import warnings

from aquasot3d.models.temporal import (
    # Vision-Mamba Head
    SpatialPositionalEncoding,
    OcclusionAwareGating,
    VisionMambaBlock,
    VisionMambaHead,
    build_vision_mamba_head,
    build_vision_mamba_head_lite,
    # Temporal Aggregator
    EMASmoothing,
    VisibilityGating,
    FeatureMemoryBank,
    TemporalAggregator,
    build_temporal_aggregator,
    build_temporal_aggregator_with_memory,
    # Transformer Head
    TemporalPositionalEncoding,
    SpatialPositionalEncoding2D,
    TransformerBlock,
    TransformerTemporalHead,
    TransformerTemporalHeadLite,
    build_transformer_head,
    build_transformer_head_lite,
    build_causal_transformer_head,
    # ARTrack Head
    AutoregressiveBlock,
    TemporalMemoryBank,
    ARTrackTemporalHead,
    ARTrackTemporalHeadLite,
    build_artrack_head,
    build_artrack_head_lite,
    build_artrack_head_without_memory
)


# Check if mamba_ssm is available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    warnings.warn("mamba_ssm not installed. Skipping Mamba-related tests.")

# Check CUDA availability (required for Mamba SSM)
CUDA_AVAILABLE = torch.cuda.is_available()
if MAMBA_AVAILABLE and not CUDA_AVAILABLE:
    warnings.warn("CUDA not available. Mamba tests will be skipped.")


# ============================================================================
# Test Vision-Mamba Components
# ============================================================================

class TestSpatialPositionalEncoding:
    """Test SpatialPositionalEncoding for feature maps."""

    def test_initialization(self):
        """Test module initialization."""
        pos_enc = SpatialPositionalEncoding(feature_dim=256, max_spatial_size=32)
        assert pos_enc.feature_dim == 256
        assert pos_enc.max_spatial_size == 32

    def test_forward_shape(self):
        """Test output shape correctness."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)

        pos_enc = SpatialPositionalEncoding(feature_dim=C, max_spatial_size=32)
        output = pos_enc(features)

        assert output.shape == (B, C, H, W)

    def test_different_spatial_sizes(self):
        """Test with different spatial sizes."""
        pos_enc = SpatialPositionalEncoding(feature_dim=256, max_spatial_size=32)

        # Test with different sizes
        for H, W in [(8, 8), (16, 16), (32, 32)]:
            features = torch.randn(2, 256, H, W)
            output = pos_enc(features)
            assert output.shape == (2, 256, H, W)

    def test_learnable_parameters(self):
        """Test that positional embeddings are learnable."""
        pos_enc = SpatialPositionalEncoding(feature_dim=256, max_spatial_size=32)

        # Check parameters exist
        param_count = sum(p.numel() for p in pos_enc.parameters())
        assert param_count > 0


class TestOcclusionAwareGating:
    """Test OcclusionAwareGating for visibility-based feature gating."""

    def test_initialization(self):
        """Test module initialization."""
        gate = OcclusionAwareGating(feature_dim=256, use_learned_threshold=True)
        assert gate.feature_dim == 256
        assert gate.use_learned_threshold is True

    def test_forward_with_quality_scores(self):
        """Test forward with provided quality scores."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)
        quality_scores = torch.rand(B)  # [0, 1]

        gate = OcclusionAwareGating(feature_dim=C)
        gated_features, gate_weights = gate(features, quality_scores)

        assert gated_features.shape == (B, C, H, W)
        assert gate_weights.shape == (B,)
        assert (gate_weights >= 0).all() and (gate_weights <= 1).all()

    def test_forward_without_quality_scores(self):
        """Test forward without quality scores (predictor mode)."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)

        gate = OcclusionAwareGating(feature_dim=C)
        gated_features, gate_weights = gate(features, quality_scores=None)

        assert gated_features.shape == (B, C, H, W)
        assert gate_weights.shape == (B,)

    def test_temporal_input(self):
        """Test with temporal input [B, T, C, H, W]."""
        B, T, C, H, W = 2, 5, 256, 16, 16
        features = torch.randn(B, T, C, H, W)
        quality_scores = torch.rand(B, T)

        gate = OcclusionAwareGating(feature_dim=C)
        gated_features, gate_weights = gate(features, quality_scores)

        assert gated_features.shape == (B, T, C, H, W)
        assert gate_weights.shape == (B, T)

    def test_threshold_gating(self):
        """Test that threshold gating mechanism works."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)

        # Test with different quality scores
        high_quality = torch.tensor([0.9, 0.8])
        low_quality = torch.tensor([0.1, 0.05])

        gate = OcclusionAwareGating(feature_dim=C, use_learned_threshold=False)

        # High quality should produce non-zero weights
        gated_high, weights_high = gate(features, high_quality)
        # Low quality should produce weights (may not be zero due to encoder, but should be lower)
        gated_low, weights_low = gate(features, low_quality)

        # Gate weights should be in valid range [0, 1]
        assert (weights_high >= 0).all() and (weights_high <= 1).all()
        assert (weights_low >= 0).all() and (weights_low <= 1).all()

        # Gated features should have correct shape
        assert gated_high.shape == (B, C, H, W)
        assert gated_low.shape == (B, C, H, W)


@pytest.mark.skipif(not MAMBA_AVAILABLE or not CUDA_AVAILABLE,
                    reason="mamba_ssm not installed or CUDA not available")
class TestVisionMambaBlock:
    """Test VisionMambaBlock for temporal processing (requires CUDA)."""

    def test_initialization(self):
        """Test module initialization."""
        block = VisionMambaBlock(d_model=256, d_state=16, d_conv=4, expand=2)
        assert block.d_model == 256

    def test_forward_shape(self):
        """Test output shape correctness."""
        device = torch.device('cuda')
        B, L, D = 2, 64, 256
        x = torch.randn(B, L, D).to(device)

        block = VisionMambaBlock(d_model=D, d_state=16).to(device)
        output = block(x)

        assert output.shape == (B, L, D)
        assert output.device.type == 'cuda'

    def test_residual_connection(self):
        """Test that residual connection is working."""
        device = torch.device('cuda')
        B, L, D = 2, 64, 256
        x = torch.randn(B, L, D).to(device)

        block = VisionMambaBlock(d_model=D, d_state=16, dropout=0.0).to(device)
        output = block(x)

        # Output should be different from input (not identity)
        assert not torch.allclose(output, x, atol=1e-5)


@pytest.mark.skipif(not MAMBA_AVAILABLE or not CUDA_AVAILABLE,
                    reason="mamba_ssm not installed or CUDA not available")
class TestVisionMambaHead:
    """Test VisionMambaHead for temporal aggregation (requires CUDA)."""

    def test_initialization(self):
        """Test module initialization."""
        head = VisionMambaHead(
            feature_dim=256,
            num_layers=4,
            bidirectional=True
        )
        assert head.feature_dim == 256
        assert head.num_layers == 4
        assert head.bidirectional is True

    def test_forward_single_frame(self):
        """Test forward with single frame input."""
        device = torch.device('cuda')
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W).to(device)

        head = VisionMambaHead(feature_dim=C, num_layers=2).to(device)
        output, state = head(features)

        assert output.shape == (B, C, H, W)
        assert output.device.type == 'cuda'
        assert state is None

    def test_forward_temporal_sequence(self):
        """Test forward with temporal sequence."""
        device = torch.device('cuda')
        B, T, C, H, W = 2, 5, 256, 16, 16
        features = torch.randn(B, T, C, H, W).to(device)

        head = VisionMambaHead(feature_dim=C, num_layers=2).to(device)
        output, state = head(features)

        assert output.shape == (B, C, H, W)
        assert output.device.type == 'cuda'

    def test_bidirectional_processing(self):
        """Test bidirectional processing."""
        device = torch.device('cuda')
        B, T, C, H, W = 2, 8, 256, 16, 16
        features = torch.randn(B, T, C, H, W).to(device)

        head_bidir = VisionMambaHead(
            feature_dim=C,
            num_layers=2,
            bidirectional=True
        ).to(device)
        output_bidir, _ = head_bidir(features)

        head_unidir = VisionMambaHead(
            feature_dim=C,
            num_layers=2,
            bidirectional=False
        ).to(device)
        output_unidir, _ = head_unidir(features)

        # Outputs should be different
        assert not torch.allclose(output_bidir, output_unidir, atol=1e-5)

    def test_occlusion_gating(self):
        """Test occlusion-aware gating."""
        device = torch.device('cuda')
        B, T, C, H, W = 2, 5, 256, 16, 16
        features = torch.randn(B, T, C, H, W).to(device)
        quality_scores = torch.rand(B, T).to(device)

        head = VisionMambaHead(
            feature_dim=C,
            num_layers=2,
            use_occlusion_gating=True
        ).to(device)
        output, _ = head(features, quality_scores)

        assert output.shape == (B, C, H, W)
        assert output.device.type == 'cuda'

    def test_gradient_flow(self):
        """Test gradient flow through the module."""
        device = torch.device('cuda')
        B, T, C, H, W = 2, 4, 256, 16, 16
        features = torch.randn(B, T, C, H, W, device=device, requires_grad=True)

        head = VisionMambaHead(feature_dim=C, num_layers=2).to(device)
        output, _ = head(features)

        # Compute dummy loss and backward
        loss = output.mean()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()


# ============================================================================
# Test Temporal Aggregator Components
# ============================================================================

class TestEMASmoothing:
    """Test EMASmoothing for temporal feature smoothing."""

    def test_initialization(self):
        """Test module initialization."""
        ema = EMASmoothing(feature_dim=256, momentum=0.1, adaptive=True)
        assert ema.feature_dim == 256
        assert ema.base_momentum == 0.1
        assert ema.adaptive is True

    def test_first_frame(self):
        """Test first frame (no previous features)."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)

        ema = EMASmoothing(feature_dim=C, adaptive=False)
        output = ema(features, feat_prev=None)

        # First frame should return input directly
        assert torch.allclose(output, features)

    def test_smoothing(self):
        """Test EMA smoothing behavior."""
        B, C, H, W = 2, 256, 16, 16
        feat_current = torch.randn(B, C, H, W)
        feat_prev = torch.randn(B, C, H, W)

        ema = EMASmoothing(feature_dim=C, momentum=0.1, adaptive=False)
        output = ema(feat_current, feat_prev)

        # Output should be blend of current and previous
        expected = 0.1 * feat_current + 0.9 * feat_prev
        assert torch.allclose(output, expected, atol=1e-5)

    def test_adaptive_momentum(self):
        """Test adaptive momentum computation."""
        B, C, H, W = 2, 256, 16, 16
        feat_current = torch.randn(B, C, H, W)
        feat_prev = torch.randn(B, C, H, W)

        ema = EMASmoothing(feature_dim=C, momentum=0.1, adaptive=True)
        output = ema(feat_current, feat_prev)

        assert output.shape == (B, C, H, W)
        # With adaptive, output should not be exactly fixed-momentum result
        expected_fixed = 0.1 * feat_current + 0.9 * feat_prev
        assert not torch.allclose(output, expected_fixed, atol=1e-5)


class TestVisibilityGating:
    """Test VisibilityGating for temporal aggregation."""

    def test_initialization(self):
        """Test module initialization."""
        gate = VisibilityGating(feature_dim=256, gate_threshold=0.3)
        assert gate.feature_dim == 256
        assert gate.gate_threshold == 0.3

    def test_hard_gating(self):
        """Test hard binary gating."""
        B, C, H, W = 4, 256, 16, 16
        features = torch.randn(B, C, H, W)
        quality_scores = torch.tensor([0.8, 0.5, 0.2, 0.1])

        gate = VisibilityGating(
            feature_dim=C,
            gate_threshold=0.3,
            use_hard_gating=True
        )
        gate_weights, should_update = gate(features, quality_scores)

        # Check binary gating
        expected_update = torch.tensor([True, True, False, False])
        assert torch.equal(should_update, expected_update)
        assert gate_weights[0] == 1.0 and gate_weights[1] == 1.0
        assert gate_weights[2] == 0.0 and gate_weights[3] == 0.0

    def test_soft_gating(self):
        """Test soft gating."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)
        quality_scores = torch.tensor([0.8, 0.2])

        gate = VisibilityGating(
            feature_dim=C,
            gate_threshold=0.3,
            use_hard_gating=False
        )
        gate_weights, should_update = gate(features, quality_scores)

        # Soft gating uses quality scores directly
        assert torch.allclose(gate_weights, quality_scores)


class TestFeatureMemoryBank:
    """Test FeatureMemoryBank for long-term context."""

    def test_initialization(self):
        """Test module initialization."""
        memory = FeatureMemoryBank(
            feature_dim=256,
            spatial_size=(16, 16),
            memory_size=10
        )
        assert memory.feature_dim == 256
        assert memory.spatial_size == (16, 16)
        assert memory.memory_size == 10

    def test_push_and_query(self):
        """Test push and query operations."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)

        memory = FeatureMemoryBank(
            feature_dim=C,
            spatial_size=(H, W),
            memory_size=5
        )

        # Push features
        memory.push(features)
        assert memory.memory_count == B

        # Query memory
        query = torch.randn(2, C, H, W)
        result = memory.query(query)
        assert result.shape == (2, C, H, W)

    def test_fifo_behavior(self):
        """Test FIFO memory management."""
        B, C, H, W = 1, 256, 16, 16

        memory = FeatureMemoryBank(
            feature_dim=C,
            spatial_size=(H, W),
            memory_size=3
        )

        # Push 5 frames
        for _ in range(5):
            features = torch.randn(B, C, H, W)
            memory.push(features)

        # Should only keep last 3
        assert memory.memory_buffer.size(0) == 3
        assert memory.memory_count == 3

    def test_reset(self):
        """Test memory reset."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)

        memory = FeatureMemoryBank(
            feature_dim=C,
            spatial_size=(H, W),
            memory_size=10
        )

        memory.push(features)
        assert memory.memory_count == B

        memory.reset()
        assert memory.memory_count == 0
        assert memory.memory_buffer.size(0) == 0


class TestTemporalAggregator:
    """Test TemporalAggregator integration."""

    def test_initialization(self):
        """Test module initialization."""
        aggregator = TemporalAggregator(
            feature_dim=256,
            momentum=0.1,
            adaptive_momentum=True
        )
        assert aggregator.feature_dim == 256

    def test_forward_first_frame(self):
        """Test forward on first frame."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)

        aggregator = TemporalAggregator(feature_dim=C)
        aggregator.reset_state()

        output, state = aggregator(features)

        assert output.shape == (B, C, H, W)
        assert 'prev_features' in state
        assert 'frame_count' in state

    def test_temporal_smoothing(self):
        """Test temporal smoothing across frames."""
        B, C, H, W = 2, 256, 16, 16

        aggregator = TemporalAggregator(feature_dim=C, momentum=0.1)
        aggregator.reset_state()

        # Process multiple frames
        outputs = []
        for _ in range(5):
            features = torch.randn(B, C, H, W)
            output, state = aggregator(features)
            outputs.append(output)

        assert len(outputs) == 5
        assert all(o.shape == (B, C, H, W) for o in outputs)

    def test_visibility_gating_effect(self):
        """Test that visibility gating affects output."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)

        # High quality
        high_quality = torch.tensor([0.9, 0.8])
        # Low quality
        low_quality = torch.tensor([0.1, 0.05])

        aggregator = TemporalAggregator(
            feature_dim=C,
            gate_threshold=0.3
        )
        aggregator.reset_state()

        # First frame with high quality
        output1, state1 = aggregator(features, high_quality)

        # Second frame with low quality
        output2, state2 = aggregator(features, low_quality)

        # Check gating info in state
        assert 'gate_weights' in state2
        assert 'should_update' in state2

    def test_with_memory_bank(self):
        """Test temporal aggregator with memory bank."""
        B, C, H, W = 2, 256, 16, 16

        aggregator = TemporalAggregator(
            feature_dim=C,
            use_memory_bank=True,
            memory_size=5,
            spatial_size=(H, W)
        )
        aggregator.reset_state()

        # Process frames
        for _ in range(10):
            features = torch.randn(B, C, H, W)
            output, state = aggregator(features)

        assert output.shape == (B, C, H, W)


# ============================================================================
# Test Transformer Components
# ============================================================================

class TestTemporalPositionalEncoding:
    """Test TemporalPositionalEncoding."""

    def test_initialization(self):
        """Test module initialization."""
        pos_enc = TemporalPositionalEncoding(d_model=256, max_sequence_length=100)
        assert pos_enc.d_model == 256

    def test_forward_shape(self):
        """Test output shape correctness."""
        B, T, D = 2, 10, 256
        x = torch.randn(B, T, D)

        pos_enc = TemporalPositionalEncoding(d_model=D, max_sequence_length=100)
        output = pos_enc(x)

        assert output.shape == (B, T, D)


class TestTransformerBlock:
    """Test TransformerBlock."""

    def test_initialization(self):
        """Test module initialization."""
        block = TransformerBlock(d_model=256, num_heads=8, dim_feedforward=1024)
        assert hasattr(block, 'self_attn')
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')

    def test_forward_shape(self):
        """Test output shape correctness."""
        B, T, D = 2, 10, 256
        x = torch.randn(B, T, D)

        block = TransformerBlock(d_model=D, num_heads=8)
        output = block(x)

        assert output.shape == (B, T, D)

    def test_residual_connection(self):
        """Test residual connection."""
        B, T, D = 2, 10, 256
        x = torch.randn(B, T, D)

        block = TransformerBlock(d_model=D, num_heads=8, dropout=0.0)
        output = block(x)

        # Output should be different from input
        assert not torch.allclose(output, x, atol=1e-5)


class TestTransformerTemporalHead:
    """Test TransformerTemporalHead."""

    def test_initialization(self):
        """Test module initialization."""
        head = TransformerTemporalHead(
            feature_dim=256,
            num_layers=4,
            num_heads=8
        )
        assert head.feature_dim == 256
        assert head.num_layers == 4

    def test_forward_single_frame(self):
        """Test forward with single frame."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)

        head = TransformerTemporalHead(feature_dim=C, num_layers=2, num_heads=8)
        output, state = head(features)

        assert output.shape == (B, C, H, W)

    def test_forward_temporal_sequence(self):
        """Test forward with temporal sequence."""
        B, T, C, H, W = 2, 5, 256, 16, 16
        features = torch.randn(B, T, C, H, W)

        head = TransformerTemporalHead(feature_dim=C, num_layers=2, num_heads=8)
        output, state = head(features)

        assert output.shape == (B, C, H, W)

    def test_causal_masking(self):
        """Test causal attention masking."""
        B, T, C, H, W = 2, 8, 256, 16, 16
        features = torch.randn(B, T, C, H, W)

        head = TransformerTemporalHead(
            feature_dim=C,
            num_layers=2,
            num_heads=8,
            use_causal_mask=True
        )
        output, _ = head(features)

        assert output.shape == (B, C, H, W)

    def test_gradient_flow(self):
        """Test gradient flow."""
        B, T, C, H, W = 2, 4, 256, 16, 16
        features = torch.randn(B, T, C, H, W, requires_grad=True)

        head = TransformerTemporalHead(feature_dim=C, num_layers=2, num_heads=8)
        output, _ = head(features)

        loss = output.mean()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()


# ============================================================================
# Test Factory Functions
# ============================================================================

@pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba_ssm not installed")
class TestFactoryFunctions:
    """Test factory functions for building temporal modules."""

    def test_build_vision_mamba_head(self):
        """Test build_vision_mamba_head factory."""
        head = build_vision_mamba_head(feature_dim=256, num_layers=4)
        assert isinstance(head, VisionMambaHead)
        assert head.feature_dim == 256
        assert head.num_layers == 4

    def test_build_vision_mamba_head_lite(self):
        """Test build_vision_mamba_head_lite factory."""
        head = build_vision_mamba_head_lite(feature_dim=256)
        assert isinstance(head, VisionMambaHead)
        assert head.num_layers == 2
        assert head.bidirectional is False

    def test_build_temporal_aggregator(self):
        """Test build_temporal_aggregator factory."""
        aggregator = build_temporal_aggregator(feature_dim=256, momentum=0.1)
        assert isinstance(aggregator, TemporalAggregator)
        assert aggregator.feature_dim == 256

    def test_build_temporal_aggregator_with_memory(self):
        """Test build_temporal_aggregator_with_memory factory."""
        aggregator = build_temporal_aggregator_with_memory(
            feature_dim=256,
            memory_size=10
        )
        assert isinstance(aggregator, TemporalAggregator)
        assert aggregator.use_memory_bank is True

    def test_build_transformer_head(self):
        """Test build_transformer_head factory."""
        head = build_transformer_head(feature_dim=256, num_layers=4, num_heads=8)
        assert isinstance(head, TransformerTemporalHead)
        assert head.feature_dim == 256

    def test_build_transformer_head_lite(self):
        """Test build_transformer_head_lite factory."""
        head = build_transformer_head_lite(feature_dim=256)
        assert isinstance(head, TransformerTemporalHeadLite)

    def test_build_causal_transformer_head(self):
        """Test build_causal_transformer_head factory."""
        head = build_causal_transformer_head(feature_dim=256)
        assert isinstance(head, TransformerTemporalHead)
        assert head.use_causal_mask is True

    def test_build_artrack_head(self):
        """Test build_artrack_head factory."""
        head = build_artrack_head(feature_dim=256, num_layers=3)
        assert isinstance(head, ARTrackTemporalHead)
        assert head.feature_dim == 256
        assert head.num_layers == 3

    def test_build_artrack_head_lite(self):
        """Test build_artrack_head_lite factory."""
        head = build_artrack_head_lite(feature_dim=256)
        assert isinstance(head, ARTrackTemporalHeadLite)
        assert head.num_layers == 2

    def test_build_artrack_head_without_memory(self):
        """Test build_artrack_head_without_memory factory."""
        head = build_artrack_head_without_memory(feature_dim=256)
        assert isinstance(head, ARTrackTemporalHead)
        assert head.use_cross_attention is False


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.skipif(not MAMBA_AVAILABLE or not CUDA_AVAILABLE,
                    reason="mamba_ssm not installed or CUDA not available")
class TestIntegration:
    """Integration tests for temporal modules (requires CUDA for Mamba)."""

    def test_mamba_with_aggregator_pipeline(self):
        """Test Vision-Mamba head + Temporal aggregator pipeline."""
        device = torch.device('cuda')
        B, T, C, H, W = 2, 8, 256, 16, 16
        features = torch.randn(B, T, C, H, W).to(device)

        # Temporal aggregator
        aggregator = TemporalAggregator(feature_dim=C).to(device)
        aggregator.reset_state()

        # Vision-Mamba head
        mamba_head = VisionMambaHead(feature_dim=C, num_layers=2).to(device)

        # Process through aggregator first
        aggregated_outputs = []
        for t in range(T):
            feat_t = features[:, t]
            agg_out, _ = aggregator(feat_t)
            aggregated_outputs.append(agg_out)

        # Stack and process through Mamba
        aggregated_seq = torch.stack(aggregated_outputs, dim=1)
        final_output, _ = mamba_head(aggregated_seq)

        assert final_output.shape == (B, C, H, W)
        assert final_output.device.type == 'cuda'

    def test_transformer_vs_mamba_consistency(self):
        """Test that Transformer and Mamba produce consistent output shapes."""
        device = torch.device('cuda')
        B, T, C, H, W = 2, 5, 256, 16, 16
        features = torch.randn(B, T, C, H, W).to(device)

        # Transformer head
        transformer_head = TransformerTemporalHead(
            feature_dim=C,
            num_layers=2,
            num_heads=8
        ).to(device)
        transformer_out, _ = transformer_head(features)

        # Mamba head
        mamba_head = VisionMambaHead(feature_dim=C, num_layers=2).to(device)
        mamba_out, _ = mamba_head(features)

        # Both should produce same output shape
        assert transformer_out.shape == mamba_out.shape == (B, C, H, W)

    def test_quality_aware_temporal_processing(self):
        """Test quality-aware temporal processing across all modules."""
        B, T, C, H, W = 2, 10, 256, 16, 16
        features = torch.randn(B, T, C, H, W)

        # Simulate quality degradation in middle frames
        quality_scores = torch.ones(B, T)
        quality_scores[:, 4:7] = 0.1  # Low quality frames

        # Test with aggregator
        aggregator = TemporalAggregator(
            feature_dim=C,
            gate_threshold=0.3,
            use_hard_gating=True
        )
        aggregator.reset_state()

        outputs = []
        for t in range(T):
            output, state = aggregator(features[:, t], quality_scores[:, t])
            outputs.append(output)

            # Low quality frames should be gated
            if t >= 4 and t < 7:
                should_update = state['should_update']
                assert not should_update.any()

        assert len(outputs) == T


# ============================================================================
# Test ARTrack Components
# ============================================================================

class TestAutoregressiveBlock:
    """Test AutoregressiveBlock for hybrid attention."""

    def test_initialization(self):
        """Test module initialization."""
        block = AutoregressiveBlock(d_model=256, num_heads=8)
        assert hasattr(block, 'self_attn')
        assert hasattr(block, 'cross_attn')

    def test_forward_without_memory(self):
        """Test forward without memory (self-attention only)."""
        B, L, D = 2, 64, 256
        query = torch.randn(B, L, D)

        block = AutoregressiveBlock(d_model=D, num_heads=8)
        output = block(query, memory=None)

        assert output.shape == (B, L, D)

    def test_forward_with_memory(self):
        """Test forward with memory (self + cross attention)."""
        B, L_q, L_m, D = 2, 64, 32, 256
        query = torch.randn(B, L_q, D)
        memory = torch.randn(B, L_m, D)

        block = AutoregressiveBlock(d_model=D, num_heads=8, use_cross_attention=True)
        output = block(query, memory)

        assert output.shape == (B, L_q, D)

    def test_without_cross_attention(self):
        """Test block without cross-attention."""
        B, L, D = 2, 64, 256
        query = torch.randn(B, L, D)
        memory = torch.randn(B, 32, D)

        block = AutoregressiveBlock(d_model=D, num_heads=8, use_cross_attention=False)
        output = block(query, memory)

        # Should work even with memory provided (just ignores it)
        assert output.shape == (B, L, D)


class TestTemporalMemoryBankARTrack:
    """Test TemporalMemoryBank for ARTrack."""

    def test_initialization(self):
        """Test module initialization."""
        memory = TemporalMemoryBank(feature_dim=256, memory_size=20)
        assert memory.feature_dim == 256
        assert memory.memory_size == 20

    def test_add_and_get(self):
        """Test adding and retrieving features."""
        memory = TemporalMemoryBank(feature_dim=256, memory_size=10)

        # Add features
        features = torch.randn(2, 256)
        memory.add(features)

        # Get memory
        retrieved = memory.get(batch_size=2)
        assert retrieved is not None
        assert retrieved.shape[0] == 2
        assert retrieved.shape[2] == 256

    def test_quality_filtering(self):
        """Test quality-based filtering."""
        memory = TemporalMemoryBank(
            feature_dim=256,
            memory_size=10,
            use_quality_filter=True,
            quality_threshold=0.5
        )

        # Add high quality features
        features_high = torch.randn(2, 256)
        quality_high = torch.tensor([0.9, 0.8])
        memory.add(features_high, quality_high)

        count_after_high = memory.memory_count.item()

        # Add low quality features (should be filtered)
        features_low = torch.randn(2, 256)
        quality_low = torch.tensor([0.2, 0.1])
        memory.add(features_low, quality_low)

        # Count should remain same (low quality filtered)
        assert memory.memory_count.item() == count_after_high

    def test_fifo_behavior(self):
        """Test FIFO sliding window."""
        memory = TemporalMemoryBank(feature_dim=256, memory_size=5)

        # Add more than memory_size
        for _ in range(10):
            features = torch.randn(1, 256)
            memory.add(features)

        # Should only keep last 5
        assert memory.memory_count.item() == 5

    def test_reset(self):
        """Test memory reset."""
        memory = TemporalMemoryBank(feature_dim=256, memory_size=10)

        features = torch.randn(2, 256)
        memory.add(features)
        assert memory.memory_count.item() > 0

        memory.reset()
        assert memory.memory_count.item() == 0


class TestARTrackTemporalHead:
    """Test ARTrackTemporalHead."""

    def test_initialization(self):
        """Test module initialization."""
        head = ARTrackTemporalHead(
            feature_dim=256,
            num_layers=3,
            num_heads=8
        )
        assert head.feature_dim == 256
        assert head.num_layers == 3

    def test_forward_single_frame(self):
        """Test forward with single frame."""
        B, C, H, W = 2, 256, 16, 16
        features = torch.randn(B, C, H, W)

        head = ARTrackTemporalHead(feature_dim=C, num_layers=2, num_heads=8)
        output, state = head(features)

        assert output.shape == (B, C, H, W)
        assert 'frame_count' in state
        assert 'memory_size' in state

    def test_forward_temporal_sequence(self):
        """Test forward with temporal sequence."""
        B, T, C, H, W = 2, 8, 256, 16, 16
        features = torch.randn(B, T, C, H, W)

        head = ARTrackTemporalHead(feature_dim=C, num_layers=2, num_heads=8)
        output, state = head(features)

        assert output.shape == (B, C, H, W)
        assert state['frame_count'] == T

    def test_memory_accumulation(self):
        """Test that memory accumulates over frames."""
        B, C, H, W = 2, 256, 16, 16

        head = ARTrackTemporalHead(
            feature_dim=C,
            num_layers=2,
            num_heads=8,
            memory_size=10
        )
        head.reset_state()

        # Process multiple frames
        for i in range(5):
            features = torch.randn(B, C, H, W)
            output, state = head(features)

            # Memory should grow
            assert state['memory_size'] == min(i + 1, 10) * B

    def test_quality_aware_memory(self):
        """Test quality-aware memory updates."""
        B, C, H, W = 2, 256, 16, 16

        head = ARTrackTemporalHead(
            feature_dim=C,
            num_layers=2,
            num_heads=8,
            use_quality_filter=True,
            quality_threshold=0.5
        )
        head.reset_state()

        # High quality frame
        features1 = torch.randn(B, C, H, W)
        quality1 = torch.tensor([0.9, 0.8])
        output1, state1 = head(features1, quality1)

        memory_after_high = state1['memory_size']

        # Low quality frame (should not be added to memory)
        features2 = torch.randn(B, C, H, W)
        quality2 = torch.tensor([0.2, 0.1])
        output2, state2 = head(features2, quality2)

        # Memory size should remain same
        assert state2['memory_size'] == memory_after_high

    def test_gradient_flow(self):
        """Test gradient flow through ARTrack head."""
        B, T, C, H, W = 2, 4, 256, 16, 16
        features = torch.randn(B, T, C, H, W, requires_grad=True)

        head = ARTrackTemporalHead(feature_dim=C, num_layers=2, num_heads=8)
        output, _ = head(features)

        loss = output.mean()
        loss.backward()

        assert features.grad is not None
        assert not torch.isnan(features.grad).any()

    def test_reset_state(self):
        """Test state reset functionality."""
        B, C, H, W = 2, 256, 16, 16

        head = ARTrackTemporalHead(feature_dim=C, num_layers=2, num_heads=8)

        # Process some frames
        for _ in range(5):
            features = torch.randn(B, C, H, W)
            head(features)

        # Reset
        head.reset_state()

        # Memory should be empty
        features = torch.randn(B, C, H, W)
        output, state = head(features)
        assert state['memory_size'] == B  # Only current frame


class TestARTrackTemporalHeadLite:
    """Test lightweight ARTrack head."""

    def test_initialization(self):
        """Test lightweight variant initialization."""
        head = ARTrackTemporalHeadLite(feature_dim=256)
        assert head.num_layers == 2  # Reduced
        assert head.num_heads == 4   # Reduced
        assert head.memory_bank.memory_size == 10  # Reduced

    def test_forward(self):
        """Test forward pass."""
        B, T, C, H, W = 2, 8, 256, 16, 16
        features = torch.randn(B, T, C, H, W)

        head = ARTrackTemporalHeadLite(feature_dim=C)
        output, state = head(features)

        assert output.shape == (B, C, H, W)
