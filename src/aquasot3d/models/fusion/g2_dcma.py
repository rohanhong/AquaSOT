"""
G²-DCMA: Geometry-Gated Deformable Cross-Modal Attention

This is the core fusion module of AquaSOT-3D, integrating three key components:
1. Geometry-Gated Attention: Uses radial and scale priors to modulate cross-attention
2. Deformable Attention: Learns sparse sampling offsets for spatial alignment
3. Dual Consistency: Enforces component and representation consistency

The G²-DCMA module enables calibration-free fusion between spatially unaligned
RGB and Sonar modalities by combining geometric awareness, adaptive sampling,
and consistency constraints.

Architecture:
    RGB features + Sonar features
           ↓
    [Geometric Gating]  (radial + scale priors)
           ↓
    [Deformable Attention]  (learnable offsets)
           ↓
    [Fusion + Consistency]  (component + representation)
           ↓
    Fused features + Consistency losses

Reference:
    AquaSOT-3D Technical Specification Section 3: G²-DCMA Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

from aquasot3d.core.base_module import BaseModule
from aquasot3d.core.registry import FUSION_REGISTRY
from aquasot3d.models.fusion.geometric_gating import GeometryGatedAttention
from aquasot3d.models.fusion.deformable_attention import (
    DeformableCrossAttention,
    MultiScaleDeformableAttention
)
from aquasot3d.models.fusion.consistency_loss import (
    DualConsistencyLoss,
    ProjectionHead
)


@FUSION_REGISTRY.register("g2_dcma")
class G2DCMA(BaseModule):
    """
    Geometry-Gated Deformable Cross-Modal Attention (G²-DCMA).

    Core fusion module that integrates geometric gating, deformable attention,
    and dual consistency for calibration-free RGB-Sonar fusion.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_radial_bins: Number of radial bins for sonar geometric gating
        num_scales: Number of pyramid scales for RGB geometric gating
        num_deformable_points: Number of sampling points per head for deformable attention
        use_multi_scale: Whether to use multi-scale deformable attention
        num_levels: Number of pyramid levels (if multi-scale)
        dropout: Dropout probability
        consistency_config: Configuration for dual consistency loss
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_radial_bins: int = 50,
        num_scales: int = 3,
        num_deformable_points: int = 4,
        use_multi_scale: bool = False,
        num_levels: int = 3,
        dropout: float = 0.1,
        consistency_config: Optional[Dict] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_multi_scale = use_multi_scale

        # Geometric gating module
        self.geometry_gating = GeometryGatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_radial_bins=num_radial_bins,
            num_scales=num_scales,
            dropout=dropout
        )

        # Deformable attention module
        if use_multi_scale:
            self.deformable_attention = MultiScaleDeformableAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_deformable_points,
                dropout=dropout
            )
        else:
            self.deformable_attention = DeformableCrossAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_points=num_deformable_points,
                dropout=dropout
            )

        # Feature projection heads for consistency learning
        self.rgb_projection = ProjectionHead(
            input_dim=embed_dim,
            hidden_dim=512,
            output_dim=128,
            num_layers=2
        )
        self.sonar_projection = ProjectionHead(
            input_dim=embed_dim,
            hidden_dim=512,
            output_dim=128,
            num_layers=2
        )

        # Dual consistency loss
        if consistency_config is None:
            consistency_config = {
                'component_weight': 0.2,
                'representation_weight': 0.1,
                'component_loss_type': 'l1',
                'infonce_temperature': 0.07
            }
        self.consistency_loss = DualConsistencyLoss(**consistency_config)

        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward_single_scale(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        rgb_pyramid: Optional[List[torch.Tensor]] = None,
        return_consistency: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single-scale fusion.

        Args:
            rgb_features: RGB feature maps [B, C, H_rgb, W_rgb]
            sonar_features: Sonar feature maps [B, C, H_sonar, W_sonar]
            rgb_pyramid: Optional multi-scale RGB features for geometric gating
            return_consistency: Whether to compute consistency losses

        Returns:
            dict containing:
                - fused_features: Fused cross-modal features [B, C, H_rgb, W_rgb]
                - geometry_gated: Geometry-gated features
                - deformable_attended: Deformable attention output
                - consistency_losses: Dict of consistency losses (if return_consistency=True)
                - geometric_info: Geometric gating information
                - deformable_info: Deformable attention information
        """
        B, C, H_rgb, W_rgb = rgb_features.shape

        # Step 1: Geometric Gating
        geometry_output = self.geometry_gating(
            rgb_features=rgb_features,
            sonar_features=sonar_features,
            rgb_pyramid=rgb_pyramid
        )
        geometry_gated = geometry_output['fused_features']  # [B, C, H_rgb, W_rgb]

        # Step 2: Deformable Attention
        # Apply deformable attention to geometry-gated features
        deformable_output = self.deformable_attention(
            query_features=rgb_features,  # Use original RGB as queries
            key_features=sonar_features,  # Sonar as keys
            value_features=geometry_gated  # Geometry-gated as values
        )
        deformable_attended = deformable_output['fused_features']  # [B, C, H_rgb, W_rgb]

        # Step 3: Feature Fusion
        # Concatenate and fuse
        concat_features = torch.cat([rgb_features, deformable_attended], dim=1)  # [B, 2C, H_rgb, W_rgb]
        fused_features = self.fusion_layer(concat_features)  # [B, C, H_rgb, W_rgb]

        # Residual connection
        alpha = torch.sigmoid(self.residual_weight)
        fused_features = alpha * fused_features + (1 - alpha) * rgb_features

        output = {
            'fused_features': fused_features,
            'geometry_gated': geometry_gated,
            'deformable_attended': deformable_attended,
            'geometric_info': {
                'gating_weights': geometry_output['gating_weights'],
                'radial_encoding': geometry_output['radial_encoding'],
                'scale_encoding': geometry_output['scale_encoding'],
                'attention_probs': geometry_output['attention_probs']
            },
            'deformable_info': {
                'offsets': deformable_output['offsets'],
                'sampling_locations': deformable_output['sampling_locations'],
                'attention_weights': deformable_output['attention_weights']
            }
        }

        # Step 4: Consistency Losses
        if return_consistency:
            # Project features to embedding space
            rgb_embeddings = self.rgb_projection(rgb_features)  # [B, 128]
            sonar_embeddings = self.sonar_projection(sonar_features)  # [B, 128]

            # For component consistency, we use X coordinates from geometric encodings
            # (In practice, this would come from 3D prediction heads, here we use encodings as proxy)
            pred_rgb_component = geometry_output['scale_encoding']  # [B, 64]
            pred_sonar_component = geometry_output['radial_encoding']  # [B, 64]

            consistency_output = self.consistency_loss(
                pred_rgb_component=pred_rgb_component,
                pred_sonar_component=pred_sonar_component,
                embeddings_rgb=rgb_embeddings,
                embeddings_sonar=sonar_embeddings
            )

            output['consistency_losses'] = consistency_output

        return output

    def forward_multi_scale(
        self,
        rgb_pyramid: List[torch.Tensor],
        sonar_pyramid: List[torch.Tensor],
        return_consistency: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-scale fusion.

        Args:
            rgb_pyramid: List of RGB features at different scales
            sonar_pyramid: List of Sonar features at different scales
            return_consistency: Whether to compute consistency losses

        Returns:
            dict containing multi-scale fusion results
        """
        # Multi-scale deformable attention
        deformable_output = self.deformable_attention(
            query_pyramid=rgb_pyramid,
            key_pyramid=sonar_pyramid,
            value_pyramid=sonar_pyramid
        )

        fused_features = deformable_output['fused_features']
        fused_pyramid = deformable_output['fused_pyramid']

        # Apply fusion layer to final output
        B, C, H, W = fused_features.shape
        concat_features = torch.cat([rgb_pyramid[0], fused_features], dim=1)
        fused_features = self.fusion_layer(concat_features)

        # Residual connection
        alpha = torch.sigmoid(self.residual_weight)
        fused_features = alpha * fused_features + (1 - alpha) * rgb_pyramid[0]

        output = {
            'fused_features': fused_features,
            'fused_pyramid': fused_pyramid,
            'level_weights': deformable_output['level_weights']
        }

        # Consistency losses
        if return_consistency:
            rgb_embeddings = self.rgb_projection(rgb_pyramid[0])
            sonar_embeddings = self.sonar_projection(sonar_pyramid[0])

            # Use level-0 features for component consistency
            pred_rgb_component = F.adaptive_avg_pool2d(rgb_pyramid[0], 1).flatten(1)
            pred_sonar_component = F.adaptive_avg_pool2d(sonar_pyramid[0], 1).flatten(1)

            consistency_output = self.consistency_loss(
                pred_rgb_component=pred_rgb_component,
                pred_sonar_component=pred_sonar_component,
                embeddings_rgb=rgb_embeddings,
                embeddings_sonar=sonar_embeddings
            )

            output['consistency_losses'] = consistency_output

        return output

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        rgb_pyramid: Optional[List[torch.Tensor]] = None,
        sonar_pyramid: Optional[List[torch.Tensor]] = None,
        return_consistency: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for G²-DCMA fusion.

        Args:
            rgb_features: RGB feature maps [B, C, H_rgb, W_rgb]
            sonar_features: Sonar feature maps [B, C, H_sonar, W_sonar]
            rgb_pyramid: Optional list of multi-scale RGB features
            sonar_pyramid: Optional list of multi-scale Sonar features
            return_consistency: Whether to compute consistency losses

        Returns:
            dict containing fusion results and consistency losses
        """
        if self.use_multi_scale and rgb_pyramid is not None and sonar_pyramid is not None:
            return self.forward_multi_scale(rgb_pyramid, sonar_pyramid, return_consistency)
        else:
            return self.forward_single_scale(rgb_features, sonar_features, rgb_pyramid, return_consistency)


@FUSION_REGISTRY.register("g2_dcma_lite")
class G2DCMALite(BaseModule):
    """
    Lightweight variant of G²-DCMA with reduced computational cost.

    Simplifications:
    - Fewer attention heads
    - Fewer deformable sampling points
    - Shared projection heads
    - Single-scale only

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads (reduced)
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Lightweight G²-DCMA with reduced parameters
        self.g2_dcma = G2DCMA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_radial_bins=25,  # Reduced from 50
            num_scales=2,  # Reduced from 3
            num_deformable_points=2,  # Reduced from 4
            use_multi_scale=False,
            dropout=dropout,
            consistency_config={
                'component_weight': 0.1,
                'representation_weight': 0.05,
                'component_loss_type': 'l1',
                'infonce_temperature': 0.1
            }
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        return_consistency: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for lightweight G²-DCMA."""
        return self.g2_dcma(
            rgb_features=rgb_features,
            sonar_features=sonar_features,
            return_consistency=return_consistency
        )


def build_g2_dcma(config: dict) -> G2DCMA:
    """
    Factory function to build G²-DCMA from config.

    Args:
        config: Configuration dictionary

    Returns:
        G2DCMA instance
    """
    consistency_config = config.get('consistency', {})

    return G2DCMA(
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_radial_bins=config.get('num_radial_bins', 50),
        num_scales=config.get('num_scales', 3),
        num_deformable_points=config.get('num_deformable_points', 4),
        use_multi_scale=config.get('use_multi_scale', False),
        num_levels=config.get('num_levels', 3),
        dropout=config.get('dropout', 0.1),
        consistency_config=consistency_config if consistency_config else None
    )


def build_g2_dcma_lite(config: dict) -> G2DCMALite:
    """
    Factory function to build lightweight G²-DCMA from config.

    Args:
        config: Configuration dictionary

    Returns:
        G2DCMALite instance
    """
    return G2DCMALite(
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1)
    )
