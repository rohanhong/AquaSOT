"""
Transformer Temporal Head for AquaSOT-3D

This module implements a standard Transformer-based temporal modeling head as a
fallback/comparison option to Vision-Mamba. It uses multi-head self-attention
for temporal aggregation across video sequences.

Components:
- PositionalEncoding: Learnable temporal positional encoding
- TransformerBlock: Standard Transformer encoder block
- TransformerTemporalHead: Complete Transformer-based temporal head

Key Features:
- Standard multi-head self-attention mechanism
- Learnable temporal positional encoding
- Layer normalization and residual connections
- Efficient implementation using PyTorch native Transformer
- Suitable for ablation comparison with Vision-Mamba

Note:
    This is provided as a baseline for ablation studies. Vision-Mamba is
    recommended for better efficiency on long sequences (O(N) vs O(N^2)).

Reference:
    AquaSOT-3D Technical Specification Section 5: Temporal Modeling
    Attention Is All You Need (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

from aquasot3d.core.base_module import BaseTemporalHead
from aquasot3d.core.registry import TEMPORAL_HEAD_REGISTRY


class TemporalPositionalEncoding(nn.Module):
    """
    Learnable temporal positional encoding for video sequences.

    Encodes temporal position information to help the model understand
    frame ordering and temporal relationships.

    Args:
        d_model: Feature dimension
        max_sequence_length: Maximum sequence length to support
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 256,
        max_sequence_length: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_sequence_length, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add temporal positional encoding.

        Args:
            x: Input sequence [B, T, D]

        Returns:
            Sequence with positional encoding [B, T, D]
        """
        B, T, D = x.shape

        # Add positional encoding
        pos_enc = self.pos_embedding[:, :T, :]  # [1, T, D]
        x = x + pos_enc

        return self.dropout(x)


class SpatialPositionalEncoding2D(nn.Module):
    """
    2D spatial positional encoding for feature maps.

    Args:
        d_model: Feature dimension
        max_height: Maximum feature map height
        max_width: Maximum feature map width
    """

    def __init__(
        self,
        d_model: int = 256,
        max_height: int = 32,
        max_width: int = 32
    ):
        super().__init__()
        self.d_model = d_model

        # Learnable 2D position embeddings
        self.pos_embed_h = nn.Parameter(
            torch.randn(max_height, d_model // 2) * 0.02
        )
        self.pos_embed_w = nn.Parameter(
            torch.randn(max_width, d_model // 2) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add 2D spatial positional encoding.

        Args:
            x: Feature map [B, C, H, W]

        Returns:
            Feature map with positional encoding [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Get position embeddings
        pos_h = self.pos_embed_h[:H, :].unsqueeze(1).expand(H, W, -1)
        pos_w = self.pos_embed_w[:W, :].unsqueeze(0).expand(H, W, -1)

        # Concatenate and reshape to [C, H, W]
        pos_embed = torch.cat([pos_h, pos_w], dim=-1).permute(2, 0, 1)

        # Add to features
        x = x + pos_embed.unsqueeze(0)

        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block.

    Args:
        d_model: Feature dimension
        num_heads: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        activation: Activation function (default: relu)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Transformer block.

        Args:
            x: Input sequence [B, T, D]
            attn_mask: Attention mask [T, T] (optional)
            key_padding_mask: Key padding mask [B, T] (optional)

        Returns:
            Output sequence [B, T, D]
        """
        # Self-attention with residual connection
        x2, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        # Feedforward with residual connection
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        return x


@TEMPORAL_HEAD_REGISTRY.register("transformer_head")
class TransformerTemporalHead(BaseTemporalHead):
    """
    Transformer-based temporal head for single object tracking.

    This head uses standard Transformer encoder blocks for temporal aggregation.
    It serves as a baseline for comparison with more efficient alternatives like
    Vision-Mamba.

    Note: Transformer has O(N^2) complexity in sequence length, making it less
    efficient than Mamba's O(N) for long sequences. However, it is well-established
    and can serve as a strong baseline.

    Args:
        feature_dim: Feature channel dimension (default: 256)
        num_layers: Number of Transformer blocks (default: 4)
        num_heads: Number of attention heads (default: 8)
        dim_feedforward: Feedforward network dimension (default: 1024)
        dropout: Dropout rate (default: 0.1)
        max_sequence_length: Maximum sequence length (default: 100)
        use_causal_mask: Whether to use causal attention mask (default: False)
        activation: Activation function (default: relu)
        spatial_encoding: Whether to add spatial positional encoding (default: True)
        config: Optional configuration dictionary
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_sequence_length: int = 100,
        use_causal_mask: bool = False,
        activation: str = "relu",
        spatial_encoding: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_causal_mask = use_causal_mask
        self.spatial_encoding = spatial_encoding

        # Spatial positional encoding
        if spatial_encoding:
            self.spatial_pos_enc = SpatialPositionalEncoding2D(
                d_model=feature_dim,
                max_height=32,
                max_width=32
            )

        # Temporal positional encoding
        self.temporal_pos_enc = TemporalPositionalEncoding(
            d_model=feature_dim,
            max_sequence_length=max_sequence_length,
            dropout=dropout
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=feature_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])

        # Output normalization
        self.output_norm = nn.LayerNorm(feature_dim)

        # Temporal state
        self.reset_state()

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate causal attention mask (upper triangular).

        Args:
            sz: Sequence length

        Returns:
            Causal mask [sz, sz]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def reset_state(self) -> None:
        """
        Reset temporal state for new sequence.
        """
        self._temporal_state = None

    def forward(
        self,
        features: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None,
        temporal_state: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any]:
        """
        Process temporal features through Transformer head.

        Args:
            features: Input features
                - For single frame: [B, C, H, W]
                - For temporal sequence: [B, T, C, H, W]
            quality_scores: Optional quality scores (unused in this implementation)
            temporal_state: Previous temporal state (unused in current implementation)

        Returns:
            Tuple of:
                - aggregated_features: Temporally aggregated features [B, C, H, W]
                - new_temporal_state: Updated temporal state (None)
        """
        # Determine input format
        if features.ndim == 4:
            # Single frame input [B, C, H, W]
            B, C, H, W = features.shape
            T = 1
            features = features.unsqueeze(1)  # [B, 1, C, H, W]
        elif features.ndim == 5:
            # Temporal sequence input [B, T, C, H, W]
            B, T, C, H, W = features.shape
        else:
            raise ValueError(
                f"Expected features to be 4D [B, C, H, W] or 5D [B, T, C, H, W], "
                f"got shape {features.shape}"
            )

        # Apply spatial positional encoding
        if self.spatial_encoding:
            features_with_pos = []
            for t in range(T):
                feat_t = self.spatial_pos_enc(features[:, t])
                features_with_pos.append(feat_t)
            features = torch.stack(features_with_pos, dim=1)

        # Reshape to sequence format: [B, T, C, H, W] -> [B, T*H*W, C]
        features_seq = features.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)

        # Add temporal positional encoding
        # For temporal position, we tile the position for each spatial location
        temporal_pos_features = []
        for t in range(T):
            # Extract features for this timestep [B, H*W, C]
            feat_t = features_seq[:, t * H * W:(t + 1) * H * W, :]
            # Create temporal position encoding for this timestep
            pos_t = self.temporal_pos_enc.pos_embedding[:, t:t + 1, :].expand(B, H * W, C)
            # Add positional encoding
            feat_t_with_pos = feat_t + pos_t
            temporal_pos_features.append(feat_t_with_pos)
        features_seq = torch.cat(temporal_pos_features, dim=1)  # [B, T*H*W, C]

        # Prepare attention mask
        if self.use_causal_mask and T > 1:
            # Causal mask for temporal dimension
            # Each spatial token at time t can only attend to t' <= t
            # Generate mask dynamically for current sequence length
            seq_len = T * H * W
            attn_mask = self._generate_square_subsequent_mask(seq_len)
            attn_mask = attn_mask.to(features.device)
        else:
            attn_mask = None

        # Process through Transformer blocks
        for block in self.transformer_blocks:
            features_seq = block(features_seq, attn_mask=attn_mask)

        # Apply output normalization
        features_seq = self.output_norm(features_seq)

        # Reshape back to spatial: [B, T*H*W, C] -> [B, T, H, W, C] -> [B, T, C, H, W]
        features_out = features_seq.view(B, T, H, W, C).permute(0, 1, 4, 2, 3)

        # Aggregate temporal dimension (use last frame)
        # For more sophisticated aggregation, can use attention pooling
        aggregated_features = features_out[:, -1]  # [B, C, H, W]

        return aggregated_features, None


@TEMPORAL_HEAD_REGISTRY.register("transformer_head_lite")
class TransformerTemporalHeadLite(TransformerTemporalHead):
    """
    Lightweight Transformer temporal head with reduced parameters.

    This variant uses fewer layers and smaller feedforward dimension for
    faster inference while maintaining reasonable performance.

    Args:
        feature_dim: Feature channel dimension (default: 256)
        num_layers: Number of Transformer blocks (default: 2)
        num_heads: Number of attention heads (default: 4)
        dim_feedforward: Feedforward network dimension (default: 512)
        **kwargs: Additional arguments passed to TransformerTemporalHead
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        **kwargs
    ):
        super().__init__(
            feature_dim=feature_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            **kwargs
        )


# ============================================================================
# Factory Functions
# ============================================================================

def build_transformer_head(
    feature_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    **kwargs
) -> TransformerTemporalHead:
    """
    Factory function to build TransformerTemporalHead.

    Args:
        feature_dim: Feature channel dimension
        num_layers: Number of Transformer blocks
        num_heads: Number of attention heads
        **kwargs: Additional arguments passed to TransformerTemporalHead

    Returns:
        TransformerTemporalHead instance
    """
    return TransformerTemporalHead(
        feature_dim=feature_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        **kwargs
    )


def build_transformer_head_lite(
    feature_dim: int = 256,
    **kwargs
) -> TransformerTemporalHeadLite:
    """
    Factory function to build lightweight TransformerTemporalHeadLite.

    Args:
        feature_dim: Feature channel dimension
        **kwargs: Additional arguments passed to TransformerTemporalHeadLite

    Returns:
        TransformerTemporalHeadLite instance
    """
    return TransformerTemporalHeadLite(
        feature_dim=feature_dim,
        **kwargs
    )


def build_causal_transformer_head(
    feature_dim: int = 256,
    num_layers: int = 4,
    **kwargs
) -> TransformerTemporalHead:
    """
    Factory function to build causal TransformerTemporalHead.

    Causal masking ensures each frame can only attend to past frames,
    suitable for online/streaming scenarios.

    Args:
        feature_dim: Feature channel dimension
        num_layers: Number of Transformer blocks
        **kwargs: Additional arguments passed to TransformerTemporalHead

    Returns:
        TransformerTemporalHead instance with causal masking
    """
    return TransformerTemporalHead(
        feature_dim=feature_dim,
        num_layers=num_layers,
        use_causal_mask=True,
        **kwargs
    )
