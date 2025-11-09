"""
Temporal Modeling Module for AquaSOT-3D

This package implements temporal modeling heads for robust single object tracking
across video sequences. It provides multiple temporal aggregation strategies including
state space models (Mamba), transformers, and EMA-based smoothing.

Components:
- mamba_head: Vision-Mamba SSM blocks for linear-complexity temporal modeling
- temporal_aggregator: EMA smoothing with visibility gating for short-term aggregation
- transformer_head: Transformer-based temporal modeling as fallback/baseline
- artrack_head: ARTrack-style autoregressive temporal modeling for ablation comparison

Usage:
    from aquasot3d.models.temporal import (
        VisionMambaHead,
        TemporalAggregator,
        TransformerTemporalHead,
        build_vision_mamba_head
    )

    # Build Vision-Mamba head
    mamba_head = build_vision_mamba_head(
        feature_dim=256,
        num_layers=4,
        use_mamba2=False
    )

    # Build temporal aggregator
    aggregator = TemporalAggregator(
        feature_dim=256,
        momentum=0.1,
        adaptive_momentum=True
    )

    # Build Transformer head (baseline)
    transformer_head = TransformerTemporalHead(
        feature_dim=256,
        num_layers=4,
        num_heads=8
    )

    # Build ARTrack head (autoregressive, optional)
    artrack_head = ARTrackTemporalHead(
        feature_dim=256,
        num_layers=3,
        num_heads=8,
        memory_size=20
    )
"""

# Vision-Mamba Head
from aquasot3d.models.temporal.mamba_head import (
    SpatialPositionalEncoding,
    OcclusionAwareGating,
    VisionMambaBlock,
    VisionMambaHead,
    build_vision_mamba_head,
    build_vision_mamba_head_lite
)

# Temporal Aggregator
from aquasot3d.models.temporal.temporal_aggregator import (
    EMASmoothing,
    VisibilityGating,
    FeatureMemoryBank,
    TemporalAggregator,
    build_temporal_aggregator,
    build_temporal_aggregator_with_memory
)

# Transformer Head
from aquasot3d.models.temporal.transformer_head import (
    TemporalPositionalEncoding,
    SpatialPositionalEncoding2D,
    TransformerBlock,
    TransformerTemporalHead,
    TransformerTemporalHeadLite,
    build_transformer_head,
    build_transformer_head_lite,
    build_causal_transformer_head
)

# ARTrack Head
from aquasot3d.models.temporal.artrack_head import (
    AutoregressiveBlock,
    TemporalMemoryBank,
    ARTrackTemporalHead,
    ARTrackTemporalHeadLite,
    build_artrack_head,
    build_artrack_head_lite,
    build_artrack_head_without_memory
)

__all__ = [
    # Vision-Mamba Head
    'SpatialPositionalEncoding',
    'OcclusionAwareGating',
    'VisionMambaBlock',
    'VisionMambaHead',
    'build_vision_mamba_head',
    'build_vision_mamba_head_lite',

    # Temporal Aggregator
    'EMASmoothing',
    'VisibilityGating',
    'FeatureMemoryBank',
    'TemporalAggregator',
    'build_temporal_aggregator',
    'build_temporal_aggregator_with_memory',

    # Transformer Head
    'TemporalPositionalEncoding',
    'SpatialPositionalEncoding2D',
    'TransformerBlock',
    'TransformerTemporalHead',
    'TransformerTemporalHeadLite',
    'build_transformer_head',
    'build_transformer_head_lite',
    'build_causal_transformer_head',

    # ARTrack Head
    'AutoregressiveBlock',
    'TemporalMemoryBank',
    'ARTrackTemporalHead',
    'ARTrackTemporalHeadLite',
    'build_artrack_head',
    'build_artrack_head_lite',
    'build_artrack_head_without_memory'
]
