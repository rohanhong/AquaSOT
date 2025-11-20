"""
Vision-Mamba Temporal Head for AquaSOT-3D

This module implements a Vision-Mamba based temporal modeling head for robust
single object tracking across video sequences. It uses Mamba SSM (State Space Model)
blocks for linear-complexity temporal aggregation with long-range dependency capture.

ARCHITECTURE CHANGE (Audit 2025112006):
Implemented true dual-mode operation for frame-level temporal tracking:
- Training mode: Processes sequences [B, T, C, H, W] and returns all temporal features
- Inference mode: Maintains temporal state and returns single-frame features

Components:
- VisionMambaBlock: Single Mamba SSM block with spatial encoding
- OcclusionAwareGating: Dynamic gating based on visibility/quality scores
- VisionMambaHead: Complete temporal head with bidirectional processing

Key Features:
- Linear O(N) complexity for long sequences (vs quadratic for Transformers)
- Selective state space mechanism for adaptive temporal modeling
- Occlusion-aware gating to handle missing/low-quality frames
- Bidirectional temporal processing for context from both past and future
- Dual-mode operation: batch training vs streaming inference
- Compatible with both Mamba and Mamba-2 architectures

Reference:
    AquaSOT-3D Technical Specification Section 5: Temporal Modeling
    Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2024)
    Audit 2025112006: Template-to-Temporal Architecture Migration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import warnings

from aquasot3d.core.base_module import BaseTemporalHead
from aquasot3d.core.registry import TEMPORAL_HEAD_REGISTRY

# Try to import mamba_ssm, with fallback warning if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    warnings.warn(
        "mamba_ssm not installed. Install with: pip install mamba-ssm. "
        "VisionMambaHead will not be functional without it.",
        ImportWarning
    )


class SpatialPositionalEncoding(nn.Module):
    """
    Add spatial positional information to feature maps before temporal processing.

    For spatial features (H, W), this encodes position information to help
    Mamba blocks understand spatial structure when flattened to sequences.

    Args:
        feature_dim: Feature channel dimension
        max_spatial_size: Maximum spatial dimension (height or width)
    """

    def __init__(self, feature_dim: int = 256, max_spatial_size: int = 32):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_spatial_size = max_spatial_size

        # Learnable 2D position embeddings
        self.pos_embed_h = nn.Parameter(
            torch.randn(max_spatial_size, feature_dim // 2) * 0.02
        )
        self.pos_embed_w = nn.Parameter(
            torch.randn(max_spatial_size, feature_dim // 2) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to spatial features.

        Args:
            x: Features of shape [B, C, H, W]

        Returns:
            Features with positional encoding [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Get position embeddings for current spatial size
        pos_h = self.pos_embed_h[:H, :]  # [H, C//2]
        pos_w = self.pos_embed_w[:W, :]  # [W, C//2]

        # Expand to grid
        pos_h = pos_h.unsqueeze(1).expand(H, W, -1)  # [H, W, C//2]
        pos_w = pos_w.unsqueeze(0).expand(H, W, -1)  # [H, W, C//2]

        # Concatenate and permute to [C, H, W]
        pos_embed = torch.cat([pos_h, pos_w], dim=-1).permute(2, 0, 1)  # [C, H, W]

        # Add to features
        x = x + pos_embed.unsqueeze(0)  # Broadcast over batch

        return x


class OcclusionAwareGating(nn.Module):
    """
    Occlusion-aware gating mechanism for temporal aggregation.

    During tracking, targets may be occluded or frames may have poor quality
    (e.g., turbidity, noise in underwater scenarios). This module learns to
    gate features based on quality/visibility scores.

    Args:
        feature_dim: Feature channel dimension
        use_learned_threshold: Whether to learn gating threshold
    """

    def __init__(
        self,
        feature_dim: int = 256,
        use_learned_threshold: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_learned_threshold = use_learned_threshold

        # Quality score encoder
        self.quality_encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Feature-based visibility predictor (when quality score unavailable)
        self.visibility_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Learnable gating threshold
        if use_learned_threshold:
            self.gate_threshold = nn.Parameter(torch.tensor(0.3))
        else:
            self.register_buffer('gate_threshold', torch.tensor(0.3))

    def forward(
        self,
        features: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gating weights based on quality/visibility.

        Args:
            features: Input features [B, C, H, W] or [B, T, C, H, W]
            quality_scores: Optional quality scores [B] or [B, T]

        Returns:
            Tuple of:
                - gated_features: Features with gating applied
                - gate_weights: Computed gate weights [B] or [B, T]
        """
        # Determine if temporal dimension exists
        if features.ndim == 5:  # [B, T, C, H, W]
            B, T, C, H, W = features.shape
            has_temporal = True
            # Reshape for processing
            features_flat = features.reshape(B * T, C, H, W)
        else:  # [B, C, H, W]
            B, C, H, W = features.shape
            has_temporal = False
            features_flat = features

        # Compute visibility/quality scores
        if quality_scores is not None:
            # Use provided quality scores
            if has_temporal:
                quality_scores_flat = quality_scores.reshape(B * T, 1)
            else:
                quality_scores_flat = quality_scores.reshape(B, 1)
            gate_weights = self.quality_encoder(quality_scores_flat).squeeze(-1)
        else:
            # Predict visibility from features
            gate_weights = self.visibility_predictor(features_flat).squeeze(-1)

        # Apply threshold-based gating
        gate_weights = torch.where(
            gate_weights > self.gate_threshold,
            gate_weights,
            torch.zeros_like(gate_weights)
        )

        # Reshape gate weights and apply to features
        if has_temporal:
            gate_weights = gate_weights.reshape(B, T)
            gate_weights_expanded = gate_weights.reshape(B, T, 1, 1, 1)
            gated_features = features * gate_weights_expanded
        else:
            gate_weights_expanded = gate_weights.reshape(B, 1, 1, 1)
            gated_features = features * gate_weights_expanded

        return gated_features, gate_weights


class VisionMambaBlock(nn.Module):
    """
    Single Vision-Mamba block for temporal feature processing.

    This wraps a Mamba SSM block with spatial encoding and normalization
    suitable for vision tasks.

    Args:
        d_model: Model dimension (feature channel size)
        d_state: SSM state expansion factor (typically 16)
        d_conv: Local convolution width (typically 4)
        expand: Block expansion factor (typically 2)
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is required for VisionMambaBlock. "
                "Install with: pip install mamba-ssm"
            )

        self.d_model = d_model

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

        # Mamba SSM block
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process sequence through Mamba block.

        Args:
            x: Input sequence [B, L, D] where L is sequence length

        Returns:
            Output sequence [B, L, D]
        """
        # Pre-normalization
        x_norm = self.norm(x)

        # Mamba SSM
        x_mamba = self.mamba(x_norm)

        # Residual connection and dropout
        x = x + self.dropout(x_mamba)

        return x


@TEMPORAL_HEAD_REGISTRY.register("vision_mamba_head")
class VisionMambaHead(BaseTemporalHead):
    """
    Vision-Mamba temporal head for robust single object tracking.

    This head processes temporal sequences of features using Mamba SSM blocks,
    providing linear-complexity temporal modeling with long-range dependencies.
    Features bidirectional processing and occlusion-aware gating.

    Args:
        feature_dim: Feature channel dimension (default: 256)
        num_layers: Number of Mamba blocks (default: 4)
        d_state: SSM state expansion factor (default: 16)
        d_conv: Local convolution width (default: 4)
        expand: Block expansion factor (default: 2)
        bidirectional: Whether to process sequences bidirectionally (default: True)
        use_occlusion_gating: Whether to use occlusion-aware gating (default: True)
        dropout: Dropout rate (default: 0.1)
        max_spatial_size: Maximum spatial dimension for positional encoding (default: 32)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        use_occlusion_gating: bool = True,
        dropout: float = 0.1,
        max_spatial_size: int = 32,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is required for VisionMambaHead. "
                "Install with: pip install mamba-ssm"
            )

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_occlusion_gating = use_occlusion_gating

        # CRITICAL FIX (Audit 2025111703 Finding 2): Temporal window for state persistence
        # Maintain sliding window of recent frames for streaming inference
        self.temporal_window_size = 5  # Keep last N frames for temporal context

        # Spatial positional encoding
        self.spatial_pos_encoding = SpatialPositionalEncoding(
            feature_dim=feature_dim,
            max_spatial_size=max_spatial_size
        )

        # Occlusion-aware gating
        if use_occlusion_gating:
            self.occlusion_gate = OcclusionAwareGating(
                feature_dim=feature_dim,
                use_learned_threshold=True
            )

        # Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            VisionMambaBlock(
                d_model=feature_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Bidirectional processing: separate blocks for backward pass
        if bidirectional:
            self.mamba_blocks_backward = nn.ModuleList([
                VisionMambaBlock(
                    d_model=feature_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ])

            # Fusion layer for bidirectional outputs
            self.bidirectional_fusion = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim)
            )

        # Final output projection
        self.output_norm = nn.LayerNorm(feature_dim)

        # Temporal state (for incremental processing)
        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset temporal state for new sequence.

        This is called when starting tracking on a new video sequence.

        CRITICAL FIX (Audit 2025111703 Finding 2):
        Clear frame buffer to prevent state leakage across sequences.
        """
        self._temporal_state = None
        self._frame_buffer = []  # List of (features, quality_scores) tuples

    def forward(
        self,
        features: torch.Tensor,
        quality_scores: Optional[torch.Tensor] = None,
        temporal_state: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Any]:
        """
        Process temporal features through Vision-Mamba head with dual-mode operation.

        ARCHITECTURE CHANGE (Audit 2025112006): Dual-mode output
        - Training mode (self.training=True): Returns all temporal features [B, T, C, H, W]
        - Inference mode (self.training=False): Returns single frame [B, C, H, W]

        Args:
            features: Input features
                - For single frame: [B, C, H, W]
                - For temporal sequence: [B, T, C, H, W]
            quality_scores: Optional quality/visibility scores
                - For single frame: [B]
                - For temporal sequence: [B, T]
            temporal_state: Previous temporal state (frame buffer from last call)
                Only used in inference mode for streaming tracking

        Returns:
            Tuple of:
                - aggregated_features: Temporally processed features
                    * Training: [B, T, C, H, W] (all frames for multi-frame supervision)
                    * Inference: [B, C, H, W] (last frame only for prediction)
                - new_temporal_state: Updated temporal state (frame buffer for next call)

        CRITICAL FIXES:
        - Audit 2025111703 Finding 2: Maintains temporal state for streaming inference
        - Audit 2025112006: Dual-mode output for proper training/inference separation
        """
        # Determine input format
        if features.ndim == 4:
            # Single frame input [B, C, H, W]
            B, C, H, W = features.shape
            T = 1
            features = features.unsqueeze(1)  # [B, 1, C, H, W]
            if quality_scores is not None:
                quality_scores = quality_scores.unsqueeze(1)  # [B, 1]
            is_single_frame = True
        elif features.ndim == 5:
            # Temporal sequence input [B, T, C, H, W]
            B, T, C, H, W = features.shape
            is_single_frame = False
        else:
            raise ValueError(
                f"Expected features to be 4D [B, C, H, W] or 5D [B, T, C, H, W], "
                f"got shape {features.shape}"
            )

        # CRITICAL FIX (Audit 2025111703 Finding 2): Use temporal_state for streaming
        # Only use temporal buffer in inference mode to avoid batch size mismatch in training
        if is_single_frame and temporal_state is not None and not self.training:
            # Streaming inference: append current frame to history
            frame_buffer = temporal_state  # List of (feat, qual) tuples

            # Check batch size consistency: if buffer has frames with different batch size, reset
            if len(frame_buffer) > 0:
                buffer_batch_size = frame_buffer[0][0].shape[0]
                if buffer_batch_size != B:
                    # Batch size mismatch: clear buffer and start fresh
                    frame_buffer = []

            # Add current frame
            current_feat = features[:, 0]  # [B, C, H, W]
            current_qual = quality_scores[:, 0] if quality_scores is not None else None
            frame_buffer.append((current_feat.detach().cpu(), current_qual.detach().cpu() if current_qual is not None else None))

            # Maintain window size (FIFO)
            if len(frame_buffer) > self.temporal_window_size:
                frame_buffer.pop(0)

            # Reconstruct sequence from buffer
            buffer_features = []
            buffer_quality = []
            for feat, qual in frame_buffer:
                buffer_features.append(feat.to(features.device))
                if qual is not None:
                    buffer_quality.append(qual.to(features.device))

            features = torch.stack(buffer_features, dim=1)  # [B, T_buffer, C, H, W]
            if len(buffer_quality) > 0:
                quality_scores = torch.stack(buffer_quality, dim=1)  # [B, T_buffer]
            else:
                quality_scores = None

            T = len(frame_buffer)
        else:
            # Training or first frame: use provided sequence as-is
            frame_buffer = []
            for t in range(T):
                feat = features[:, t]
                qual = quality_scores[:, t] if quality_scores is not None else None
                frame_buffer.append((feat.detach().cpu(), qual.detach().cpu() if qual is not None else None))

        # Apply spatial positional encoding
        features_with_pos = []
        for t in range(T):
            feat_t = self.spatial_pos_encoding(features[:, t])  # [B, C, H, W]
            features_with_pos.append(feat_t)
        features = torch.stack(features_with_pos, dim=1)  # [B, T, C, H, W]

        # Apply occlusion-aware gating if enabled
        if self.use_occlusion_gating:
            features, gate_weights = self.occlusion_gate(features, quality_scores)

        # Reshape features for sequence processing: [B, T, C, H, W] -> [B, T*H*W, C]
        features_seq = features.permute(0, 1, 3, 4, 2).reshape(B, T * H * W, C)

        # Forward pass through Mamba blocks
        features_forward = features_seq
        for block in self.mamba_blocks:
            features_forward = block(features_forward)

        # Bidirectional processing if enabled
        if self.bidirectional:
            # Reverse temporal dimension
            features_reversed = features_seq.reshape(B, T, H * W, C)
            features_reversed = torch.flip(features_reversed, dims=[1])
            features_reversed = features_reversed.reshape(B, T * H * W, C)

            # Backward pass
            features_backward = features_reversed
            for block in self.mamba_blocks_backward:
                features_backward = block(features_backward)

            # Reverse back to original temporal order
            features_backward = features_backward.reshape(B, T, H * W, C)
            features_backward = torch.flip(features_backward, dims=[1])
            features_backward = features_backward.reshape(B, T * H * W, C)

            # Concatenate forward and backward features
            features_bidir = torch.cat([features_forward, features_backward], dim=-1)

            # Fuse bidirectional features
            features_out = self.bidirectional_fusion(features_bidir)
        else:
            features_out = features_forward

        # Apply output normalization
        features_out = self.output_norm(features_out)

        # Reshape back to spatial: [B, T*H*W, C] -> [B, T, H, W, C] -> [B, T, C, H, W]
        features_out = features_out.reshape(B, T, H, W, C).permute(0, 1, 4, 2, 3)

        # ARCHITECTURE CHANGE (Audit 2025112006): Dual-mode output
        # Training mode: return all frames [B, T, C, H, W] for multi-frame supervision
        # Inference mode: return last frame [B, C, H, W] for single-frame prediction
        if self.training:
            # Training: return all temporal frames for loss computation on each frame
            aggregated_features = features_out  # [B, T, C, H, W]
        else:
            # Inference: aggregate to single frame (use last frame by default)
            # For more sophisticated aggregation, can use attention or pooling
            aggregated_features = features_out[:, -1]  # [B, C, H, W]

        # CRITICAL FIX (Audit 2025111703 Finding 2): Return frame_buffer as temporal_state
        # This enables streaming inference with temporal context across frames
        return aggregated_features, frame_buffer


# ============================================================================
# Factory Functions
# ============================================================================

def build_vision_mamba_head(
    feature_dim: int = 256,
    num_layers: int = 4,
    bidirectional: bool = True,
    **kwargs
) -> VisionMambaHead:
    """
    Factory function to build VisionMambaHead.

    Args:
        feature_dim: Feature channel dimension
        num_layers: Number of Mamba blocks
        bidirectional: Whether to use bidirectional processing
        **kwargs: Additional arguments passed to VisionMambaHead

    Returns:
        VisionMambaHead instance
    """
    return VisionMambaHead(
        feature_dim=feature_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
        **kwargs
    )


def build_vision_mamba_head_lite(
    feature_dim: int = 256,
    num_layers: int = 2,
    **kwargs
) -> VisionMambaHead:
    """
    Factory function to build lightweight VisionMambaHead (fewer layers).

    Args:
        feature_dim: Feature channel dimension
        num_layers: Number of Mamba blocks (default: 2 for lite version)
        **kwargs: Additional arguments passed to VisionMambaHead

    Returns:
        Lightweight VisionMambaHead instance
    """
    return VisionMambaHead(
        feature_dim=feature_dim,
        num_layers=num_layers,
        bidirectional=False,  # Lite version uses unidirectional processing
        **kwargs
    )
