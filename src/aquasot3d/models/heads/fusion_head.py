"""
Fusion Head for Multi-Modal Feature Integration

This module implements a fusion head that combines RGB and Sonar features with
uncertainty-aware weighting. It predicts quality scores and adaptively weights
modality-specific predictions based on estimated uncertainties.

Architecture:
- Multi-modal feature fusion with learnable attention
- Uncertainty estimation for each modality (quality/confidence scores)
- Adaptive weighting based on uncertainty
- Quality score prediction for final output

Reference: NA-OSTrack (https://github.com/fbasatemur/NA-OSTrack)

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Literal

from aquasot3d.core import BaseHead, PREDICTION_HEAD_REGISTRY


class UncertaintyEstimator(nn.Module):
    """
    Estimates uncertainty/quality for a single modality.

    Uses feature statistics to estimate confidence score, which is inversely
    related to uncertainty. Higher quality features get higher confidence.

    Args:
        in_channels: Input feature dimension
        hidden_dim: Hidden layer dimension
        method: Uncertainty estimation method ('learned' or 'entropy')
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        method: Literal['learned', 'entropy'] = 'learned'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.method = method

        if method == 'learned':
            # Learnable uncertainty estimation
            self.estimator = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Output in [0, 1], higher = more confident
            )
        elif method == 'entropy':
            # Entropy-based uncertainty (lower entropy = higher confidence)
            pass  # Computed in forward
        else:
            raise ValueError(f"Unknown uncertainty estimation method: {method}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate confidence score for features.

        Args:
            features: Input features, shape (B, C) or (B, C, H, W)

        Returns:
            Confidence scores, shape (B,)
        """
        # Handle spatial features
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        elif features.dim() == 3:
            features = features.mean(dim=1)

        if self.method == 'learned':
            confidence = self.estimator(features).squeeze(-1)
        elif self.method == 'entropy':
            # Compute feature entropy as uncertainty proxy
            # Normalize features to probability distribution
            probs = F.softmax(features, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=-1)
            # Normalize entropy to [0, 1] and invert (high entropy = low confidence)
            max_entropy = torch.log(torch.tensor(features.shape[-1], dtype=torch.float32))
            confidence = 1.0 - (entropy / max_entropy)

        return confidence


class AdaptiveFusionGate(nn.Module):
    """
    Adaptive fusion gate that combines multi-modal features based on uncertainty.

    Uses uncertainty estimates to compute adaptive fusion weights, giving more
    importance to more confident modalities.

    Args:
        rgb_channels: RGB feature dimension
        sonar_channels: Sonar feature dimension
        hidden_dim: Hidden layer dimension
        fusion_method: Fusion method ('attention', 'mlp', or 'uncertainty_weighted')
    """

    def __init__(
        self,
        rgb_channels: int,
        sonar_channels: int,
        hidden_dim: int = 256,
        fusion_method: Literal['attention', 'mlp', 'uncertainty_weighted'] = 'attention'
    ):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.sonar_channels = sonar_channels
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method

        if fusion_method == 'attention':
            # Cross-attention based fusion
            self.query_proj = nn.Linear(rgb_channels + sonar_channels, hidden_dim)
            self.key_proj = nn.Linear(rgb_channels + sonar_channels, hidden_dim)
            self.value_proj = nn.Linear(rgb_channels + sonar_channels, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        elif fusion_method == 'mlp':
            # Simple MLP fusion
            self.fusion_mlp = nn.Sequential(
                nn.Linear(rgb_channels + sonar_channels, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

        elif fusion_method == 'uncertainty_weighted':
            # Uncertainty-aware weighted fusion
            self.rgb_proj = nn.Linear(rgb_channels, hidden_dim)
            self.sonar_proj = nn.Linear(sonar_channels, hidden_dim)
            # Weight predictor based on concatenated features + uncertainties
            self.weight_predictor = nn.Sequential(
                nn.Linear(rgb_channels + sonar_channels + 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1)
            )

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        rgb_confidence: Optional[torch.Tensor] = None,
        sonar_confidence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse RGB and Sonar features adaptively.

        Args:
            rgb_features: RGB features, shape (B, C_rgb)
            sonar_features: Sonar features, shape (B, C_sonar)
            rgb_confidence: RGB confidence scores, shape (B,), optional
            sonar_confidence: Sonar confidence scores, shape (B,), optional

        Returns:
            Tuple of (fused_features, fusion_weights)
        """
        batch_size = rgb_features.shape[0]

        # Handle spatial and sequence features
        if rgb_features.dim() == 4:
            rgb_features = F.adaptive_avg_pool2d(rgb_features, (1, 1)).flatten(1)
        elif rgb_features.dim() == 3:
            rgb_features = rgb_features.mean(dim=1)  # Average over sequence

        if sonar_features.dim() == 4:
            sonar_features = F.adaptive_avg_pool2d(sonar_features, (1, 1)).flatten(1)
        elif sonar_features.dim() == 3:
            sonar_features = sonar_features.mean(dim=1)  # Average over sequence

        if self.fusion_method == 'attention':
            # Concatenate features
            combined = torch.cat([rgb_features, sonar_features], dim=-1)

            # Self-attention
            Q = self.query_proj(combined)
            K = self.key_proj(combined)
            V = self.value_proj(combined)

            # Attention weights
            attn_weights = torch.softmax(
                torch.bmm(Q.unsqueeze(1), K.unsqueeze(2)) / (self.hidden_dim ** 0.5),
                dim=-1
            )

            # Weighted values
            attended = torch.bmm(attn_weights, V.unsqueeze(1)).squeeze(1)
            fused = self.out_proj(attended)

            # Dummy fusion weights (not explicitly computed in attention)
            fusion_weights = torch.ones(batch_size, 2, device=rgb_features.device) * 0.5

        elif self.fusion_method == 'mlp':
            # Simple concatenation and MLP
            combined = torch.cat([rgb_features, sonar_features], dim=-1)
            fused = self.fusion_mlp(combined)

            # Dummy fusion weights
            fusion_weights = torch.ones(batch_size, 2, device=rgb_features.device) * 0.5

        elif self.fusion_method == 'uncertainty_weighted':
            # Project to common space
            rgb_proj = self.rgb_proj(rgb_features)
            sonar_proj = self.sonar_proj(sonar_features)

            # Default confidence if not provided
            if rgb_confidence is None:
                rgb_confidence = torch.ones(batch_size, device=rgb_features.device)
            if sonar_confidence is None:
                sonar_confidence = torch.ones(batch_size, device=sonar_features.device)

            # Predict fusion weights based on features and confidence
            weight_input = torch.cat([
                rgb_features,
                sonar_features,
                rgb_confidence.unsqueeze(-1),
                sonar_confidence.unsqueeze(-1)
            ], dim=-1)
            fusion_weights = self.weight_predictor(weight_input)  # (B, 2)

            # Weighted fusion
            fused = (
                fusion_weights[:, 0:1] * rgb_proj +
                fusion_weights[:, 1:2] * sonar_proj
            )

        return fused, fusion_weights


@PREDICTION_HEAD_REGISTRY.register("FusionHead")
class FusionHead(BaseHead):
    """
    Multi-Modal Fusion Head with Uncertainty-Aware Weighting.

    Combines RGB and Sonar features adaptively based on estimated uncertainties,
    and predicts a unified quality score for the fused output.

    Args:
        rgb_channels: RGB feature dimension
        sonar_channels: Sonar feature dimension
        hidden_dim: Hidden layer dimension (default: 256)
        uncertainty_method: Uncertainty estimation method (default: 'learned')
        fusion_method: Fusion method (default: 'uncertainty_weighted')
        predict_uncertainty: Whether to output uncertainty estimates (default: True)

    Returns:
        Dictionary containing:
            - 'fused_features': Fused feature representation, shape (B, hidden_dim)
            - 'fusion_weights': Adaptive fusion weights, shape (B, 2)
            - 'rgb_confidence': RGB confidence score, shape (B,)
            - 'sonar_confidence': Sonar confidence score, shape (B,)
            - 'quality_score': Unified quality prediction, shape (B,)

    Example:
        >>> head = FusionHead(rgb_channels=768, sonar_channels=384)
        >>> rgb_feat = torch.randn(4, 768)
        >>> sonar_feat = torch.randn(4, 384)
        >>> outputs = head(rgb_feat, sonar_feat)
        >>> print(outputs['fused_features'].shape)  # (4, 256)
        >>> print(outputs['fusion_weights'].shape)  # (4, 2)
    """

    def __init__(
        self,
        rgb_channels: int,
        sonar_channels: int,
        hidden_dim: int = 256,
        uncertainty_method: Literal['learned', 'entropy'] = 'learned',
        fusion_method: Literal['attention', 'mlp', 'uncertainty_weighted'] = 'uncertainty_weighted',
        predict_uncertainty: bool = True,
        **kwargs
    ):
        super().__init__()

        self.rgb_channels = rgb_channels
        self.sonar_channels = sonar_channels
        self.hidden_dim = hidden_dim
        self.uncertainty_method = uncertainty_method
        self.fusion_method = fusion_method
        self.predict_uncertainty = predict_uncertainty

        # Uncertainty estimators
        self.rgb_uncertainty_estimator = UncertaintyEstimator(
            in_channels=rgb_channels,
            hidden_dim=hidden_dim // 2,
            method=uncertainty_method
        )
        self.sonar_uncertainty_estimator = UncertaintyEstimator(
            in_channels=sonar_channels,
            hidden_dim=hidden_dim // 2,
            method=uncertainty_method
        )

        # Adaptive fusion gate
        self.fusion_gate = AdaptiveFusionGate(
            rgb_channels=rgb_channels,
            sonar_channels=sonar_channels,
            hidden_dim=hidden_dim,
            fusion_method=fusion_method
        )

        # Quality score predictor (for final fused output)
        self.quality_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.logger.info(
            f"Initialized FusionHead: rgb_channels={rgb_channels}, "
            f"sonar_channels={sonar_channels}, hidden_dim={hidden_dim}, "
            f"uncertainty_method={uncertainty_method}, fusion_method={fusion_method}"
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-modal fusion.

        Args:
            rgb_features: RGB features, shape (B, C_rgb) or (B, C_rgb, H, W)
            sonar_features: Sonar features, shape (B, C_sonar) or (B, C_sonar, H, W)
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Dictionary with fused features and quality predictions
        """
        # Estimate confidence/uncertainty for each modality
        rgb_confidence = self.rgb_uncertainty_estimator(rgb_features)  # (B,)
        sonar_confidence = self.sonar_uncertainty_estimator(sonar_features)  # (B,)

        # Adaptive fusion
        fused_features, fusion_weights = self.fusion_gate(
            rgb_features,
            sonar_features,
            rgb_confidence,
            sonar_confidence
        )  # (B, hidden_dim), (B, 2)

        # Predict quality score
        quality_score = self.quality_predictor(fused_features).squeeze(-1)  # (B,)

        outputs = {
            'fused_features': fused_features,
            'fusion_weights': fusion_weights,
            'quality_score': quality_score,
        }

        if return_uncertainty or self.predict_uncertainty:
            outputs['rgb_confidence'] = rgb_confidence
            outputs['sonar_confidence'] = sonar_confidence
            # Compute overall uncertainty (inverse of confidence)
            overall_uncertainty = 1.0 - (
                fusion_weights[:, 0] * rgb_confidence +
                fusion_weights[:, 1] * sonar_confidence
            )
            outputs['uncertainty'] = overall_uncertainty

        return outputs

    def compute_fusion_loss(
        self,
        rgb_quality: torch.Tensor,
        sonar_quality: torch.Tensor,
        fusion_weights: torch.Tensor,
        target_quality: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute fusion consistency loss.

        Encourages fusion weights to correlate with modality-specific quality scores.

        Args:
            rgb_quality: RGB quality scores, shape (B,)
            sonar_quality: Sonar quality scores, shape (B,)
            fusion_weights: Fusion weights, shape (B, 2)
            target_quality: Target quality (e.g., IoU), shape (B,), optional

        Returns:
            Fusion loss
        """
        # Weighted quality
        weighted_quality = (
            fusion_weights[:, 0] * rgb_quality +
            fusion_weights[:, 1] * sonar_quality
        )

        if target_quality is not None:
            # L1 loss between weighted quality and target
            loss = F.l1_loss(weighted_quality, target_quality)
        else:
            # Encourage high quality when both modalities agree
            agreement = 1.0 - torch.abs(rgb_quality - sonar_quality)
            loss = -agreement.mean()  # Maximize agreement

        return loss


def build_fusion_head(
    rgb_channels: int = 768,
    sonar_channels: int = 384,
    **kwargs
) -> FusionHead:
    """
    Factory function to build fusion head.

    Args:
        rgb_channels: RGB feature dimension
        sonar_channels: Sonar feature dimension
        **kwargs: Additional arguments for FusionHead

    Returns:
        FusionHead instance

    Example:
        >>> head = build_fusion_head(rgb_channels=768, sonar_channels=384, hidden_dim=256)
    """
    return FusionHead(rgb_channels=rgb_channels, sonar_channels=sonar_channels, **kwargs)
