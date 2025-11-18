"""
Temporal Aggregator for AquaSOT-3D

This module implements short-term temporal aggregation using Exponential Moving
Average (EMA) smoothing with visibility-based gating. It provides robust feature
smoothing across frames while handling occlusions and quality variations.

Components:
- EMASmoothing: Exponential moving average with adaptive momentum
- VisibilityGating: Quality-aware gating for temporal aggregation
- FeatureMemoryBank: Optional long-term feature memory (FIFO buffer)
- TemporalAggregator: Complete temporal aggregation module

Key Features:
- Adaptive EMA momentum based on feature similarity
- Visibility/quality-aware gating to skip low-quality frames
- Optional feature memory bank for long-term context
- Efficient online processing for real-time tracking

Reference:
    AquaSOT-3D Technical Specification Section 5.2: Temporal Aggregator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from collections import deque

from aquasot3d.core.base_module import BaseTemporalHead
from aquasot3d.core.registry import TEMPORAL_HEAD_REGISTRY


class EMASmoothing(nn.Module):
    """
    Exponential Moving Average smoothing for temporal features.

    EMA updates features as: feat_new = alpha * feat_current + (1 - alpha) * feat_prev
    where alpha (momentum) can be fixed or adaptive based on feature similarity.

    Args:
        feature_dim: Feature channel dimension
        momentum: Base EMA momentum (default: 0.1, higher = more current frame weight)
        adaptive: Whether to use adaptive momentum based on similarity
        min_momentum: Minimum momentum for adaptive mode
        max_momentum: Maximum momentum for adaptive mode
    """

    def __init__(
        self,
        feature_dim: int = 256,
        momentum: float = 0.1,
        adaptive: bool = True,
        min_momentum: float = 0.01,
        max_momentum: float = 0.5
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.base_momentum = momentum
        self.adaptive = adaptive
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum

        # Learnable momentum adjustment
        if adaptive:
            self.momentum_adapter = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature_dim * 2, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

    def compute_adaptive_momentum(
        self,
        feat_current: torch.Tensor,
        feat_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive momentum based on feature similarity.

        If features are similar (continuous motion), use lower momentum (more smoothing).
        If features are dissimilar (sudden change), use higher momentum (follow current).

        Args:
            feat_current: Current frame features [B, C, H, W]
            feat_prev: Previous frame features [B, C, H, W]

        Returns:
            Adaptive momentum values [B]
        """
        # Concatenate current and previous features
        feat_concat = torch.cat([feat_current, feat_prev], dim=1)  # [B, 2C, H, W]

        # Predict momentum adjustment factor [0, 1]
        momentum_factor = self.momentum_adapter(feat_concat)  # [B, 1]
        momentum_factor = momentum_factor.squeeze(-1)  # [B]

        # Scale to [min_momentum, max_momentum]
        momentum = (
            self.min_momentum +
            momentum_factor * (self.max_momentum - self.min_momentum)
        )

        return momentum

    def forward(
        self,
        feat_current: torch.Tensor,
        feat_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply EMA smoothing to features.

        Args:
            feat_current: Current frame features [B, C, H, W]
            feat_prev: Previous smoothed features [B, C, H, W]
                      If None, return feat_current directly (first frame)

        Returns:
            Smoothed features [B, C, H, W]
        """
        # First frame: no previous features
        if feat_prev is None:
            return feat_current

        # CRITICAL FIX (Audit 2025111802 Finding 1): Shape validation
        # Validation stage may have different batch size or spatial dimensions than training.
        # If shapes don't match, treat as first frame and reset previous state.
        if feat_current.shape != feat_prev.shape:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"EMA shape mismatch: current {feat_current.shape} vs prev {feat_prev.shape}. "
                "Resetting temporal state (likely transition from training to validation)."
            )
            return feat_current

        # Compute momentum
        if self.adaptive:
            momentum = self.compute_adaptive_momentum(feat_current, feat_prev)
            # Expand momentum to [B, 1, 1, 1] for broadcasting
            momentum = momentum.reshape(-1, 1, 1, 1)
        else:
            momentum = self.base_momentum

        # EMA update
        feat_smoothed = momentum * feat_current + (1 - momentum) * feat_prev

        return feat_smoothed


class VisibilityGating(nn.Module):
    """
    Visibility-based gating for temporal aggregation.

    Determines whether to update temporal features based on visibility/quality scores.
    Low-quality frames (occlusion, blur, noise) are gated to prevent corruption.

    Args:
        feature_dim: Feature channel dimension
        gate_threshold: Quality threshold below which to skip frame (default: 0.3)
        use_hard_gating: If True, use binary gates; if False, use soft weights
    """

    def __init__(
        self,
        feature_dim: int = 256,
        gate_threshold: float = 0.3,
        use_hard_gating: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.gate_threshold = gate_threshold
        self.use_hard_gating = use_hard_gating

        # Quality score encoder (when quality not provided)
        self.quality_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        features: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute visibility gates for features.

        Args:
            features: Input features [B, C, H, W]
            quality_scores: Optional quality scores [B]

        Returns:
            Tuple of:
                - gate_weights: Visibility gates [B]
                - should_update: Boolean mask for whether to update [B]
        """
        B = features.size(0)

        # Predict quality if not provided
        if quality_scores is None:
            quality_scores = self.quality_predictor(features).squeeze(-1)  # [B]

        # Compute gate weights
        if self.use_hard_gating:
            # Binary gating: 1 if quality >= threshold, 0 otherwise
            should_update = quality_scores >= self.gate_threshold
            gate_weights = should_update.float()
        else:
            # Soft gating: use quality scores directly
            gate_weights = quality_scores
            should_update = quality_scores >= self.gate_threshold

        return gate_weights, should_update


class FeatureMemoryBank(nn.Module):
    """
    Feature memory bank for long-term temporal context.

    Maintains a FIFO buffer of recent features for contextual aggregation.
    This is optional and can be used for re-detection after long occlusions.

    Args:
        feature_dim: Feature channel dimension
        spatial_size: Spatial size of features (H, W)
        memory_size: Number of frames to store (default: 10)
        use_attention: Whether to use attention for memory aggregation
    """

    def __init__(
        self,
        feature_dim: int = 256,
        spatial_size: Tuple[int, int] = (16, 16),
        memory_size: int = 10,
        use_attention: bool = False
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.spatial_size = spatial_size
        self.memory_size = memory_size
        self.use_attention = use_attention

        # Initialize memory buffer (will be populated during forward)
        self.register_buffer(
            'memory_buffer',
            torch.zeros(0, feature_dim, *spatial_size)
        )
        self.register_buffer('memory_count', torch.tensor(0))

        # Optional attention for memory aggregation
        if use_attention:
            self.memory_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            )
            self.memory_proj = nn.Linear(feature_dim, feature_dim)

    def push(self, features: torch.Tensor) -> None:
        """
        Add new features to memory bank.

        Args:
            features: Features to add [B, C, H, W]
        """
        B, C, H, W = features.shape

        # Ensure spatial size matches
        if (H, W) != self.spatial_size:
            features = F.interpolate(
                features,
                size=self.spatial_size,
                mode='bilinear',
                align_corners=False
            )

        # Initialize buffer if empty
        if self.memory_buffer.size(0) == 0:
            self.memory_buffer = features
            self.memory_count = torch.tensor(B)
        else:
            # Append to buffer
            self.memory_buffer = torch.cat([self.memory_buffer, features], dim=0)
            self.memory_count = self.memory_count + B

            # Keep only last memory_size frames (FIFO)
            if self.memory_buffer.size(0) > self.memory_size:
                self.memory_buffer = self.memory_buffer[-self.memory_size:]
                self.memory_count = torch.tensor(self.memory_size)

    def query(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Query memory bank with current features.

        Args:
            query_features: Query features [B, C, H, W]

        Returns:
            Aggregated memory features [B, C, H, W]
        """
        if self.memory_buffer.size(0) == 0:
            # No memory yet, return query features
            return query_features

        B, C, H, W = query_features.shape

        # Resize query if needed
        if (H, W) != self.spatial_size:
            query_resized = F.interpolate(
                query_features,
                size=self.spatial_size,
                mode='bilinear',
                align_corners=False
            )
        else:
            query_resized = query_features

        if self.use_attention:
            # Attention-based memory aggregation
            # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
            query_flat = query_resized.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            memory_flat = self.memory_buffer.flatten(2).permute(0, 2, 1)  # [M, H*W, C]

            # Expand memory for all batch items
            M = memory_flat.size(0)
            memory_expanded = memory_flat.unsqueeze(0).expand(B, M, -1, -1)  # [B, M, H*W, C]
            memory_expanded = memory_expanded.reshape(B, M * H * W, C)

            # Apply attention: query attends to memory
            attn_out, _ = self.memory_attention(
                query_flat,  # [B, H*W, C]
                memory_expanded,  # [B, M*H*W, C]
                memory_expanded
            )

            # Project and reshape
            attn_out = self.memory_proj(attn_out)  # [B, H*W, C]
            output = attn_out.permute(0, 2, 1).reshape(B, C, *self.spatial_size)
        else:
            # Simple average pooling over memory
            memory_avg = self.memory_buffer.mean(dim=0, keepdim=True)  # [1, C, H, W]
            output = memory_avg.expand(B, -1, -1, -1)

        # Resize back to original size if needed
        if (H, W) != self.spatial_size:
            output = F.interpolate(
                output,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        return output

    def reset(self) -> None:
        """
        Reset memory bank (for new sequence).
        """
        self.memory_buffer = torch.zeros(
            0, self.feature_dim, *self.spatial_size,
            device=self.memory_buffer.device
        )
        self.memory_count = torch.tensor(0)


@TEMPORAL_HEAD_REGISTRY.register("temporal_aggregator")
class TemporalAggregator(BaseTemporalHead):
    """
    Temporal aggregator with EMA smoothing and visibility gating.

    This module provides short-term temporal feature aggregation suitable for
    online tracking. It combines EMA smoothing with visibility-based gating
    to handle occlusions and quality variations.

    Args:
        feature_dim: Feature channel dimension (default: 256)
        momentum: Base EMA momentum (default: 0.1)
        adaptive_momentum: Whether to use adaptive momentum (default: True)
        gate_threshold: Visibility gate threshold (default: 0.3)
        use_hard_gating: Whether to use hard binary gating (default: True)
        use_memory_bank: Whether to use feature memory bank (default: False)
        memory_size: Size of memory bank if enabled (default: 10)
        spatial_size: Spatial size for memory bank (default: (16, 16))
        config: Optional configuration dictionary
    """

    def __init__(
        self,
        feature_dim: int = 256,
        momentum: float = 0.1,
        adaptive_momentum: bool = True,
        gate_threshold: float = 0.3,
        use_hard_gating: bool = True,
        use_memory_bank: bool = False,
        memory_size: int = 10,
        spatial_size: Tuple[int, int] = (16, 16),
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)

        self.feature_dim = feature_dim
        self.use_memory_bank = use_memory_bank

        # EMA smoothing
        self.ema = EMASmoothing(
            feature_dim=feature_dim,
            momentum=momentum,
            adaptive=adaptive_momentum
        )

        # Visibility gating
        self.visibility_gate = VisibilityGating(
            feature_dim=feature_dim,
            gate_threshold=gate_threshold,
            use_hard_gating=use_hard_gating
        )

        # Optional memory bank
        if use_memory_bank:
            self.memory_bank = FeatureMemoryBank(
                feature_dim=feature_dim,
                spatial_size=spatial_size,
                memory_size=memory_size,
                use_attention=True
            )

        # Temporal state
        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset temporal state for new sequence.
        """
        self._prev_features = None
        self._frame_count = 0
        if self.use_memory_bank:
            self.memory_bank.reset()

    def reset(self) -> None:
        """
        Alias for reset_state() to match tracker's introspection interface.

        CRITICAL FIX (Audit 2025111802 Finding 2):
        Tracker's reset_temporal_state() looks for reset() method via hasattr check.
        This alias ensures proper state clearing during training->validation transition.
        """
        self.reset_state()

    def forward(
        self,
        features: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None,
        temporal_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Aggregate features with EMA smoothing and visibility gating.

        Args:
            features: Input features [B, C, H, W]
            quality_scores: Optional quality/visibility scores [B]
            temporal_state: Optional previous temporal state dict with:
                - 'prev_features': Previous smoothed features
                - 'frame_count': Frame counter

        Returns:
            Tuple of:
                - aggregated_features: Temporally aggregated features [B, C, H, W]
                - new_temporal_state: Updated temporal state dict
        """
        # Load temporal state if provided
        # CRITICAL FIX (Audit 2025111801 Finding 1): Training mode should not use cross-batch state
        # Each training batch should be independent per mini-batch SGD principle.
        # Cross-batch state causes shape mismatch when batch size varies (e.g., last batch).
        # Reference: CVPR 2024 tracking methods (ARTrackV2, OneTracker) don't use cross-batch aggregation.
        if temporal_state is not None:
            prev_features = temporal_state.get('prev_features', None)
            frame_count = temporal_state.get('frame_count', 0)
        elif not self.training:
            # Inference mode: use internal state for online tracking (frame-to-frame smoothing)
            prev_features = self._prev_features
            frame_count = self._frame_count
        else:
            # Training mode: no cross-batch state (each batch is independent temporal sequence)
            prev_features = None
            frame_count = 0

        # Compute visibility gates
        gate_weights, should_update = self.visibility_gate(features, quality_scores)

        # Apply EMA smoothing
        if prev_features is not None and should_update.any():
            # Update only frames that pass visibility gate
            features_smoothed = self.ema(features, prev_features)

            # Apply gating: use smoothed for high-quality, keep previous for low-quality
            gate_weights_expanded = gate_weights.reshape(-1, 1, 1, 1)
            aggregated_features = (
                gate_weights_expanded * features_smoothed +
                (1 - gate_weights_expanded) * prev_features
            )
        else:
            # First frame or all frames gated: use current features
            aggregated_features = features

        # Update memory bank if enabled
        if self.use_memory_bank and should_update.any():
            # Only add high-quality frames to memory
            high_quality_features = features[should_update]
            if high_quality_features.size(0) > 0:
                self.memory_bank.push(high_quality_features)

        # Query memory bank for additional context
        if self.use_memory_bank:
            memory_features = self.memory_bank.query(aggregated_features)
            # Blend with memory (weighted by quality)
            memory_weight = 0.2  # Fixed weight for memory contribution
            aggregated_features = (
                (1 - memory_weight) * aggregated_features +
                memory_weight * memory_features
            )

        # Update temporal state
        self._prev_features = aggregated_features.detach()
        self._frame_count = frame_count + 1

        new_temporal_state = {
            'prev_features': self._prev_features,
            'frame_count': self._frame_count,
            'gate_weights': gate_weights,
            'should_update': should_update
        }

        return aggregated_features, new_temporal_state


# ============================================================================
# Factory Functions
# ============================================================================

def build_temporal_aggregator(
    feature_dim: int = 256,
    momentum: float = 0.1,
    adaptive: bool = True,
    **kwargs
) -> TemporalAggregator:
    """
    Factory function to build TemporalAggregator.

    Args:
        feature_dim: Feature channel dimension
        momentum: Base EMA momentum
        adaptive: Whether to use adaptive momentum
        **kwargs: Additional arguments passed to TemporalAggregator

    Returns:
        TemporalAggregator instance
    """
    return TemporalAggregator(
        feature_dim=feature_dim,
        momentum=momentum,
        adaptive_momentum=adaptive,
        **kwargs
    )


def build_temporal_aggregator_with_memory(
    feature_dim: int = 256,
    momentum: float = 0.1,
    memory_size: int = 10,
    **kwargs
) -> TemporalAggregator:
    """
    Factory function to build TemporalAggregator with memory bank.

    Args:
        feature_dim: Feature channel dimension
        momentum: Base EMA momentum
        memory_size: Size of feature memory bank
        **kwargs: Additional arguments passed to TemporalAggregator

    Returns:
        TemporalAggregator instance with memory bank
    """
    return TemporalAggregator(
        feature_dim=feature_dim,
        momentum=momentum,
        use_memory_bank=True,
        memory_size=memory_size,
        **kwargs
    )
