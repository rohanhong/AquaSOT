"""
Adaptive Weighting for Multi-Modal Fusion

Dynamically adjusts fusion weights based on RGB and Sonar quality scores.
Includes temporal smoothing and threshold-based gating.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from aquasot3d.core import BaseModule


class AdaptiveWeighting(BaseModule):
    """
    Adaptive weighting for multi-modal fusion.

    Dynamically computes fusion weights (w_rgb, w_sonar) based on quality scores,
    with optional temporal smoothing and threshold-based gating.

    Args:
        weighting_method: Method for computing weights ('softmax', 'sigmoid', 'learned')
        temporal_smoothing: Whether to apply EMA temporal smoothing
        ema_alpha: EMA smoothing factor (0 < alpha <= 1)
        quality_threshold: Threshold below which modality is disabled
        use_gating: Whether to apply threshold-based gating
        learnable_fusion: Whether to use learnable fusion weights

    Example:
        >>> weighting = AdaptiveWeighting(weighting_method='softmax', temporal_smoothing=True)
        >>> rgb_feat = torch.randn(4, 256, 14, 14)
        >>> sonar_feat = torch.randn(4, 256, 14, 14)
        >>> q_rgb = torch.tensor([[0.8], [0.9], [0.3], [0.7]])
        >>> q_sonar = torch.tensor([[0.6], [0.4], [0.85], [0.5]])
        >>> fused, weights = weighting(rgb_feat, sonar_feat, q_rgb, q_sonar)
        >>> print(fused.shape, weights['w_rgb'].shape)
    """

    def __init__(
        self,
        weighting_method: str = 'softmax',
        temporal_smoothing: bool = True,
        ema_alpha: float = 0.9,
        quality_threshold: float = 0.3,
        use_gating: bool = True,
        learnable_fusion: bool = False,
        **kwargs
    ):
        super().__init__()

        self.weighting_method = weighting_method
        self.temporal_smoothing = temporal_smoothing
        self.ema_alpha = ema_alpha
        self.quality_threshold = quality_threshold
        self.use_gating = use_gating
        self.learnable_fusion = learnable_fusion

        # EMA buffers for temporal smoothing
        if temporal_smoothing:
            self.register_buffer('ema_q_rgb', None)
            self.register_buffer('ema_q_sonar', None)

        # Learnable fusion network
        if learnable_fusion:
            # Takes quality scores + features, outputs weights
            self.fusion_net = nn.Sequential(
                nn.Linear(2, 16),  # 2 quality scores
                nn.ReLU(inplace=True),
                nn.Linear(16, 2),
                nn.Softmax(dim=-1)
            )
        else:
            self.fusion_net = None

        self.logger.info(
            f"Initialized AdaptiveWeighting: method={weighting_method}, "
            f"temporal_smoothing={temporal_smoothing}, "
            f"learnable={learnable_fusion}"
        )

    def apply_temporal_smoothing(
        self,
        q_rgb: torch.Tensor,
        q_sonar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply exponential moving average (EMA) temporal smoothing per sample.

        CRITICAL FIX (Audit 2025111409 Finding 3):
        Previous implementation averaged the entire batch into a single scalar
        and broadcast it back to all samples, causing all frames to share identical
        smoothed quality. This defeated quality-aware fusion and mixed states across
        different sequences.

        New implementation maintains per-sample EMA state:
        - Each sample in the batch has independent smoothing: ema = alpha * prev + (1-alpha) * current
        - Initialization uses current quality values (no batch averaging)
        - During evaluation (batch_size=1), this naturally provides per-frame smoothing
        - During training, each sample in the batch maintains independent state

        Args:
            q_rgb: Current RGB quality (B, 1)
            q_sonar: Current Sonar quality (B, 1)

        Returns:
            Smoothed quality scores (B, 1), each sample independently smoothed
        """
        batch_size = q_rgb.shape[0]

        # Initialize buffers on first call
        if self.ema_q_rgb is None:
            # Initialize with current quality values (no averaging)
            # Detach to prevent gradient flow through EMA state
            self.ema_q_rgb = q_rgb.detach().clone()  # (B, 1)
            self.ema_q_sonar = q_sonar.detach().clone()  # (B, 1)
            # Return original values on initialization
            return q_rgb, q_sonar

        # Handle batch size mismatch (can happen during training with varying batch sizes)
        if self.ema_q_rgb.shape[0] != batch_size:
            if batch_size == 1:
                # Evaluation mode (batch_size=1): Use only the first EMA state
                # This allows continuous per-frame smoothing across the sequence
                prev_rgb = self.ema_q_rgb[0:1].clone()
                prev_sonar = self.ema_q_sonar[0:1].clone()
            else:
                # Training mode with varying batch size: Reset to current values
                self.ema_q_rgb = q_rgb.detach().clone()
                self.ema_q_sonar = q_sonar.detach().clone()
                return q_rgb, q_sonar

            # Update buffers to new batch size for evaluation
            self.ema_q_rgb = prev_rgb
            self.ema_q_sonar = prev_sonar

        # Apply per-sample EMA update
        # ema_t = alpha * ema_{t-1} + (1 - alpha) * current_t
        self.ema_q_rgb = (
            self.ema_alpha * self.ema_q_rgb +
            (1 - self.ema_alpha) * q_rgb.detach()
        )
        self.ema_q_sonar = (
            self.ema_alpha * self.ema_q_sonar +
            (1 - self.ema_alpha) * q_sonar.detach()
        )

        # Return smoothed values (each sample has its own smoothed quality)
        q_rgb_smoothed = self.ema_q_rgb.clone()
        q_sonar_smoothed = self.ema_q_sonar.clone()

        return q_rgb_smoothed, q_sonar_smoothed

    def reset_state(self):
        """
        Reset temporal state (EMA buffers).

        Call this method at the start of each new sequence or epoch
        to prevent cross-sequence state leakage.
        """
        if self.temporal_smoothing:
            self.ema_q_rgb = None
            self.ema_q_sonar = None
            self.logger.debug("Reset AdaptiveWeighting temporal state")

    def compute_weights_softmax(
        self,
        q_rgb: torch.Tensor,
        q_sonar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fusion weights using softmax normalization.

        Args:
            q_rgb: RGB quality (B, 1)
            q_sonar: Sonar quality (B, 1)

        Returns:
            (w_rgb, w_sonar) each (B, 1)
        """
        # Stack quality scores
        qualities = torch.cat([q_rgb, q_sonar], dim=1)  # (B, 2)

        # Softmax
        weights = F.softmax(qualities, dim=1)  # (B, 2)

        w_rgb = weights[:, 0:1]  # (B, 1)
        w_sonar = weights[:, 1:2]  # (B, 1)

        return w_rgb, w_sonar

    def compute_weights_sigmoid(
        self,
        q_rgb: torch.Tensor,
        q_sonar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fusion weights using sigmoid.

        Args:
            q_rgb: RGB quality (B, 1)
            q_sonar: Sonar quality (B, 1)

        Returns:
            (w_rgb, w_sonar) each (B, 1)
        """
        # Sigmoid on difference
        diff = q_rgb - q_sonar  # (B, 1)
        w_rgb = torch.sigmoid(diff)  # (B, 1)
        w_sonar = 1.0 - w_rgb

        return w_rgb, w_sonar

    def compute_weights_learned(
        self,
        q_rgb: torch.Tensor,
        q_sonar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute fusion weights using learned network.

        Args:
            q_rgb: RGB quality (B, 1)
            q_sonar: Sonar quality (B, 1)

        Returns:
            (w_rgb, w_sonar) each (B, 1)
        """
        # Concatenate qualities
        qualities = torch.cat([q_rgb, q_sonar], dim=1)  # (B, 2)

        # Learned fusion
        weights = self.fusion_net(qualities)  # (B, 2)

        w_rgb = weights[:, 0:1]
        w_sonar = weights[:, 1:2]

        return w_rgb, w_sonar

    def apply_quality_gating(
        self,
        w_rgb: torch.Tensor,
        w_sonar: torch.Tensor,
        q_rgb: torch.Tensor,
        q_sonar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply threshold-based gating to disable low-quality modalities.

        Args:
            w_rgb: RGB weight (B, 1)
            w_sonar: Sonar weight (B, 1)
            q_rgb: RGB quality (B, 1)
            q_sonar: Sonar quality (B, 1)

        Returns:
            Gated weights (w_rgb, w_sonar)
        """
        # Create gates
        gate_rgb = (q_rgb >= self.quality_threshold).float()  # (B, 1)
        gate_sonar = (q_sonar >= self.quality_threshold).float()  # (B, 1)

        # Apply gates
        w_rgb_gated = w_rgb * gate_rgb
        w_sonar_gated = w_sonar * gate_sonar

        # Renormalize
        total_weight = w_rgb_gated + w_sonar_gated

        # Handle case where both modalities are gated out (total_weight ~= 0)
        # In this case, use equal weights as fallback
        zero_mask = (total_weight < 1e-6).float()
        total_weight = total_weight + zero_mask * 1.0  # Add 1.0 for zero cases
        w_rgb_gated = w_rgb_gated + zero_mask * 0.5
        w_sonar_gated = w_sonar_gated + zero_mask * 0.5

        # Normalize
        w_rgb_gated = w_rgb_gated / (total_weight + 1e-8)
        w_sonar_gated = w_sonar_gated / (total_weight + 1e-8)

        return w_rgb_gated, w_sonar_gated

    def forward(
        self,
        rgb_feat: torch.Tensor,
        sonar_feat: torch.Tensor,
        q_rgb: torch.Tensor,
        q_sonar: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Adaptive fusion of RGB and Sonar features.

        Args:
            rgb_feat: RGB features (B, C, H, W) or (B, C)
            sonar_feat: Sonar features (B, C, H, W) or (B, C)
            q_rgb: RGB quality scores (B, 1)
            q_sonar: Sonar quality scores (B, 1)

        Returns:
            fused_feat: Fused features (same shape as input)
            weights_dict: Dictionary containing:
                - w_rgb: RGB weights (B, 1)
                - w_sonar: Sonar weights (B, 1)
                - q_rgb_smoothed: Smoothed RGB quality (B, 1)
                - q_sonar_smoothed: Smoothed Sonar quality (B, 1)
        """
        # Apply temporal smoothing
        if self.temporal_smoothing:
            q_rgb_smoothed, q_sonar_smoothed = self.apply_temporal_smoothing(
                q_rgb, q_sonar
            )
        else:
            q_rgb_smoothed, q_sonar_smoothed = q_rgb, q_sonar

        # Compute fusion weights
        if self.weighting_method == 'softmax':
            w_rgb, w_sonar = self.compute_weights_softmax(
                q_rgb_smoothed, q_sonar_smoothed
            )
        elif self.weighting_method == 'sigmoid':
            w_rgb, w_sonar = self.compute_weights_sigmoid(
                q_rgb_smoothed, q_sonar_smoothed
            )
        elif self.weighting_method == 'learned':
            if self.learnable_fusion and self.fusion_net is not None:
                w_rgb, w_sonar = self.compute_weights_learned(
                    q_rgb_smoothed, q_sonar_smoothed
                )
            else:
                # Fallback to softmax
                w_rgb, w_sonar = self.compute_weights_softmax(
                    q_rgb_smoothed, q_sonar_smoothed
                )
        else:
            raise ValueError(f"Unknown weighting method: {self.weighting_method}")

        # Apply quality gating
        if self.use_gating:
            w_rgb, w_sonar = self.apply_quality_gating(
                w_rgb, w_sonar, q_rgb_smoothed, q_sonar_smoothed
            )

        # Fuse features
        if rgb_feat.dim() == 4:
            # Spatial features (B, C, H, W)
            w_rgb_spatial = w_rgb.view(-1, 1, 1, 1)
            w_sonar_spatial = w_sonar.view(-1, 1, 1, 1)
            fused_feat = w_rgb_spatial * rgb_feat + w_sonar_spatial * sonar_feat
        elif rgb_feat.dim() == 2:
            # Vector features (B, C)
            fused_feat = w_rgb * rgb_feat + w_sonar * sonar_feat
        else:
            raise ValueError(f"Unsupported feature dimension: {rgb_feat.dim()}")

        # Return fused features and weights
        weights_dict = {
            'w_rgb': w_rgb,
            'w_sonar': w_sonar,
            'q_rgb_smoothed': q_rgb_smoothed,
            'q_sonar_smoothed': q_sonar_smoothed,
        }

        return fused_feat, weights_dict


def create_adaptive_weighting(
    weighting_method: str = 'softmax',
    temporal_smoothing: bool = True,
    **kwargs
) -> AdaptiveWeighting:
    """
    Factory function to create adaptive weighting module.

    Args:
        weighting_method: Weighting method
        temporal_smoothing: Whether to use temporal smoothing
        **kwargs: Additional arguments

    Returns:
        AdaptiveWeighting instance

    Example:
        >>> weighting = create_adaptive_weighting('softmax', temporal_smoothing=True)
    """
    return AdaptiveWeighting(
        weighting_method=weighting_method,
        temporal_smoothing=temporal_smoothing,
        **kwargs
    )
