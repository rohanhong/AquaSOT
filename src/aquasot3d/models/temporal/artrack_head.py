"""
ARTrack-style Temporal Head for AquaSOT-3D

This module implements an autoregressive temporal modeling head inspired by ARTrackV2.
It uses a hybrid attention mechanism (self-attention + cross-attention) for temporal
feature aggregation in video tracking.

Components:
- AutoregressiveBlock: Single autoregressive transformer block
- TemporalMemoryBank: Maintains historical features for cross-attention
- ARTrackTemporalHead: Complete autoregressive temporal head

Key Features:
- Autoregressive temporal modeling with causal masking
- Hybrid attention: self-attention for current frame, cross-attention to history
- Efficient memory management with sliding window
- Quality-aware feature selection for memory updates
- Suitable for online tracking scenarios

Reference:
    ARTrackV2: Prompting Autoregressive Tracker Where to Look and How to Describe (CVPR 2024)
    AquaSOT-3D Technical Specification Section 5.3: ARTrack Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from collections import deque

from aquasot3d.core.base_module import BaseTemporalHead
from aquasot3d.core.registry import TEMPORAL_HEAD_REGISTRY


class AutoregressiveBlock(nn.Module):
    """
    Autoregressive transformer block with hybrid attention.

    Combines self-attention (for current frame features) with cross-attention
    (to historical features) for temporal modeling.

    Args:
        d_model: Feature dimension
        num_heads: Number of attention heads
        dim_feedforward: Feedforward network dimension
        dropout: Dropout rate
        use_cross_attention: Whether to use cross-attention to memory
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_cross_attention: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_cross_attention = use_cross_attention

        # Self-attention for current features
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention to historical features
        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        query: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through autoregressive block.

        Args:
            query: Current frame features [B, L_q, D]
            memory: Historical features [B, L_m, D] (optional)
            query_mask: Attention mask for query [L_q, L_q]
            memory_mask: Key padding mask for memory [B, L_m]

        Returns:
            Output features [B, L_q, D]
        """
        # Self-attention on current features
        x = query
        x2, _ = self.self_attn(
            x, x, x,
            attn_mask=query_mask,
            need_weights=False
        )
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        # Cross-attention to historical features (if available)
        if self.use_cross_attention and memory is not None:
            x2, _ = self.cross_attn(
                x, memory, memory,
                key_padding_mask=memory_mask,
                need_weights=False
            )
            x = x + self.dropout2(x2)
            x = self.norm2(x)

        # Feedforward network
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)
        x = self.norm3(x)

        return x


class TemporalMemoryBank(nn.Module):
    """
    Memory bank for storing and retrieving historical features.

    Maintains a sliding window of recent features for cross-attention.
    Supports quality-aware feature selection.

    Args:
        feature_dim: Feature channel dimension
        memory_size: Maximum number of frames to store
        use_quality_filter: Whether to filter low-quality frames
        quality_threshold: Minimum quality score to add to memory
    """

    def __init__(
        self,
        feature_dim: int = 256,
        memory_size: int = 20,
        use_quality_filter: bool = True,
        quality_threshold: float = 0.3
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.use_quality_filter = use_quality_filter
        self.quality_threshold = quality_threshold

        # Memory buffer (will be populated during forward)
        self.register_buffer(
            'memory_features',
            torch.zeros(0, feature_dim)
        )
        self.register_buffer('memory_count', torch.tensor(0))

        # Quality predictor (when quality not provided)
        self.quality_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def add(
        self,
        features: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None
    ) -> None:
        """
        Add new features to memory bank.

        Args:
            features: Features to add [B, D] or [B, L, D]
            quality_scores: Optional quality scores [B]
        """
        # Flatten spatial dimensions if needed
        if features.ndim == 3:
            B, L, D = features.shape
            features = features.mean(dim=1)  # Average pooling over sequence
        else:
            B, D = features.shape

        # Predict quality if not provided
        if quality_scores is None:
            quality_scores = self.quality_predictor(features).squeeze(-1)  # [B]

        # Filter by quality
        if self.use_quality_filter:
            valid_mask = quality_scores >= self.quality_threshold
            features = features[valid_mask]

            if features.size(0) == 0:
                return  # No valid features to add

        # Detach features before adding to memory
        features = features.detach()

        # Initialize or append to memory
        if self.memory_features.size(0) == 0:
            self.memory_features = features
        else:
            self.memory_features = torch.cat([self.memory_features, features], dim=0)

        # Maintain sliding window (FIFO)
        if self.memory_features.size(0) > self.memory_size:
            self.memory_features = self.memory_features[-self.memory_size:]

        self.memory_count = torch.tensor(self.memory_features.size(0))

    def get(self, batch_size: int = 1) -> Optional[torch.Tensor]:
        """
        Retrieve memory features.

        Args:
            batch_size: Batch size to expand memory for

        Returns:
            Memory features [B, M, D] or None if empty
        """
        if self.memory_features.size(0) == 0:
            return None

        # Expand memory for batch
        memory = self.memory_features.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [B, M, D]

        return memory

    def reset(self) -> None:
        """
        Reset memory bank.
        """
        self.memory_features = torch.zeros(
            0, self.feature_dim,
            device=self.memory_features.device
        )
        self.memory_count = torch.tensor(0)


@TEMPORAL_HEAD_REGISTRY.register("artrack_head")
class ARTrackTemporalHead(BaseTemporalHead):
    """
    ARTrack-style autoregressive temporal head.

    This head uses autoregressive modeling with hybrid attention for temporal
    feature aggregation. It maintains a memory bank of historical features and
    uses cross-attention to incorporate temporal context.

    Args:
        feature_dim: Feature channel dimension (default: 256)
        num_layers: Number of autoregressive blocks (default: 3)
        num_heads: Number of attention heads (default: 8)
        dim_feedforward: Feedforward network dimension (default: 1024)
        memory_size: Size of temporal memory bank (default: 20)
        dropout: Dropout rate (default: 0.1)
        use_cross_attention: Whether to use cross-attention (default: True)
        use_quality_filter: Whether to filter low-quality frames (default: True)
        quality_threshold: Quality threshold for memory (default: 0.3)
        config: Optional configuration dictionary
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        memory_size: int = 20,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
        use_quality_filter: bool = True,
        quality_threshold: float = 0.3,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_cross_attention = use_cross_attention

        # Autoregressive blocks
        self.ar_blocks = nn.ModuleList([
            AutoregressiveBlock(
                d_model=feature_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                use_cross_attention=use_cross_attention
            )
            for _ in range(num_layers)
        ])

        # Temporal memory bank
        self.memory_bank = TemporalMemoryBank(
            feature_dim=feature_dim,
            memory_size=memory_size,
            use_quality_filter=use_quality_filter,
            quality_threshold=quality_threshold
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(feature_dim)

        # Temporal state
        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset temporal state for new sequence.
        """
        self.memory_bank.reset()
        self._frame_count = 0

    def forward(
        self,
        features: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None,
        temporal_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process features through ARTrack temporal head.

        Args:
            features: Input features
                - For single frame: [B, C, H, W]
                - For temporal sequence: [B, T, C, H, W]
            quality_scores: Optional quality/visibility scores
                - For single frame: [B]
                - For temporal sequence: [B, T]
            temporal_state: Previous temporal state (unused in current implementation)

        Returns:
            Tuple of:
                - aggregated_features: Temporally aggregated features [B, C, H, W]
                - new_temporal_state: Updated temporal state dict
        """
        # Determine input format
        if features.ndim == 4:
            # Single frame input [B, C, H, W]
            B, C, H, W = features.shape
            T = 1
            features = features.unsqueeze(1)  # [B, 1, C, H, W]
            if quality_scores is not None:
                quality_scores = quality_scores.unsqueeze(1)  # [B, 1]
        elif features.ndim == 5:
            # Temporal sequence input [B, T, C, H, W]
            B, T, C, H, W = features.shape
        else:
            raise ValueError(
                f"Expected features to be 4D [B, C, H, W] or 5D [B, T, C, H, W], "
                f"got shape {features.shape}"
            )

        # Reshape to sequence format: [B, T, C, H, W] -> [B, T*H*W, C]
        features_seq = features.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)

        # Get memory from temporal memory bank
        memory = self.memory_bank.get(batch_size=B)  # [B, M, C] or None

        # Process through autoregressive blocks
        x = features_seq
        for block in self.ar_blocks:
            x = block(
                query=x,
                memory=memory,
                query_mask=None,
                memory_mask=None
            )

        # Output normalization
        x = self.output_norm(x)

        # Reshape back to spatial: [B, T*H*W, C] -> [B, T, H, W, C] -> [B, T, C, H, W]
        x = x.view(B, T, H, W, C).permute(0, 1, 4, 2, 3)

        # Aggregate temporal dimension (use last frame)
        aggregated_features = x[:, -1]  # [B, C, H, W]

        # Update memory bank with current features
        # Use global average pooling to get frame-level features
        for t in range(T):
            frame_feat = features[:, t]  # [B, C, H, W]
            frame_feat_pooled = F.adaptive_avg_pool2d(frame_feat, 1).squeeze(-1).squeeze(-1)  # [B, C]

            # Get quality score for this frame
            if quality_scores is not None:
                qual_t = quality_scores[:, t]
            else:
                qual_t = None

            self.memory_bank.add(frame_feat_pooled, qual_t)

        # Update frame count
        self._frame_count += T

        # Prepare temporal state
        new_temporal_state = {
            'frame_count': self._frame_count,
            'memory_size': self.memory_bank.memory_count.item()
        }

        return aggregated_features, new_temporal_state


@TEMPORAL_HEAD_REGISTRY.register("artrack_head_lite")
class ARTrackTemporalHeadLite(ARTrackTemporalHead):
    """
    Lightweight ARTrack temporal head with reduced parameters.

    This variant uses fewer layers and smaller memory for faster inference
    while maintaining reasonable performance.

    Args:
        feature_dim: Feature channel dimension (default: 256)
        num_layers: Number of autoregressive blocks (default: 2)
        num_heads: Number of attention heads (default: 4)
        memory_size: Size of temporal memory bank (default: 10)
        **kwargs: Additional arguments passed to ARTrackTemporalHead
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        memory_size: int = 10,
        **kwargs
    ):
        super().__init__(
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=512,  # Reduced from 1024
            memory_size=memory_size,
            **kwargs
        )


# ============================================================================
# Factory Functions
# ============================================================================

def build_artrack_head(
    feature_dim: int = 256,
    num_layers: int = 3,
    num_heads: int = 8,
    memory_size: int = 20,
    **kwargs
) -> ARTrackTemporalHead:
    """
    Factory function to build ARTrackTemporalHead.

    Args:
        feature_dim: Feature channel dimension
        num_layers: Number of autoregressive blocks
        num_heads: Number of attention heads
        memory_size: Size of temporal memory bank
        **kwargs: Additional arguments passed to ARTrackTemporalHead

    Returns:
        ARTrackTemporalHead instance
    """
    return ARTrackTemporalHead(
        feature_dim=feature_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        memory_size=memory_size,
        **kwargs
    )


def build_artrack_head_lite(
    feature_dim: int = 256,
    **kwargs
) -> ARTrackTemporalHeadLite:
    """
    Factory function to build lightweight ARTrackTemporalHeadLite.

    Args:
        feature_dim: Feature channel dimension
        **kwargs: Additional arguments passed to ARTrackTemporalHeadLite

    Returns:
        ARTrackTemporalHeadLite instance
    """
    return ARTrackTemporalHeadLite(
        feature_dim=feature_dim,
        **kwargs
    )


def build_artrack_head_without_memory(
    feature_dim: int = 256,
    num_layers: int = 3,
    **kwargs
) -> ARTrackTemporalHead:
    """
    Factory function to build ARTrack head without cross-attention to memory.

    This variant uses only self-attention, making it similar to a standard
    Transformer encoder but with autoregressive processing.

    Args:
        feature_dim: Feature channel dimension
        num_layers: Number of autoregressive blocks
        **kwargs: Additional arguments passed to ARTrackTemporalHead

    Returns:
        ARTrackTemporalHead instance without cross-attention
    """
    return ARTrackTemporalHead(
        feature_dim=feature_dim,
        num_layers=num_layers,
        use_cross_attention=False,
        **kwargs
    )
