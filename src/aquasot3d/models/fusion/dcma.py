"""
DCMA: Deformable Cross-Modal Attention with Confidence Estimation

Core components:
1. Confidence Estimation: Learnable end-to-end confidence scores for each modality
2. Deformable Attention: Learns sparse sampling offsets for spatial alignment
3. Dual Consistency: Enforces component and representation consistency

Architecture:
    RGB features + Sonar features
           |
    [Confidence Estimation] (learnable modality reliability)
           |
    [Confidence Modulation] (modulates attention scores)
           |
    [Deformable Attention]  (learnable offsets, bidirectional)
           |
    [Fusion + Consistency]  (component + representation)
           |
    Fused features + Confidence scores + Consistency losses

Reference:
    - PyTorch MultiheadAttention standard implementation
    - Uncertainty-Aware Attention (NeurIPS)
    - Deformable DETR (ICLR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Union

from aquasot3d.core.base_module import BaseModule
from aquasot3d.core.registry import FUSION_REGISTRY
from aquasot3d.models.fusion.deformable_attention import (
    DeformableCrossAttention,
    MultiScaleDeformableAttention
)
from aquasot3d.models.fusion.consistency_loss import (
    DualConsistencyLoss,
    ProjectionHead
)


class ModalityConfidenceEstimator(BaseModule):
    """
    Learnable confidence estimator for a single modality.

    Confidence score indicates the reliability of the modality for current frame:
    - High confidence (close to 1.0): modality provides reliable information
    - Low confidence (close to 0.0): modality is degraded (blur, noise, occlusion)

    Args:
        in_channels: Input feature dimension
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability for regularization

    Returns:
        Confidence scores in [0, 1], shape (B,)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels

        self.estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate confidence score from features.

        Args:
            features: Input features, shape (B, C, H, W) or (B, N, C)

        Returns:
            confidence: Confidence scores, shape (B,)
        """
        if features.dim() == 3:
            B, N, C = features.shape
            if int((N - 1) ** 0.5) ** 2 == N - 1:
                features = features[:, 1:, :]
                N = N - 1
            H = W = int(N ** 0.5)
            features = features.permute(0, 2, 1).reshape(B, C, H, W)

        actual_dim = features.shape[1]
        if actual_dim != self.in_channels:
            if not hasattr(self, 'input_proj'):
                self.input_proj = nn.Conv2d(
                    actual_dim, self.in_channels, kernel_size=1
                ).to(features.device)
            features = self.input_proj(features)

        confidence = self.estimator(features).squeeze(-1)
        return confidence


@FUSION_REGISTRY.register("dcma")
class DCMA(BaseModule):
    """
    Deformable Cross-Modal Attention (DCMA) with Confidence Estimation.

    Core fusion module that integrates:
    - Confidence estimation: learnable modality reliability scores
    - Deformable attention: adaptive spatial sampling (bidirectional)
    - Dual consistency: component and representation alignment

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_deformable_points: Number of sampling points per head for deformable attention
        use_multi_scale: Whether to use multi-scale deformable attention
        num_levels: Number of pyramid levels (if multi-scale)
        dropout: Dropout probability
        consistency_config: Configuration for dual consistency loss
        confidence_hidden_dim: Hidden dimension for confidence estimators (default: 128)
        use_confidence_modulation: Whether to modulate attention with confidence (default: True)
        use_deformable: Enable/disable deformable attention (for ablation)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_deformable_points: int = 4,
        use_multi_scale: bool = False,
        num_levels: int = 3,
        dropout: float = 0.1,
        consistency_config: Optional[Dict] = None,
        confidence_hidden_dim: int = 128,
        use_confidence_modulation: bool = True,
        use_deformable: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_multi_scale = use_multi_scale
        self.use_confidence_modulation = use_confidence_modulation
        self.use_deformable = use_deformable
        self.dropout = dropout

        # Confidence estimators for each modality
        self.rgb_confidence_estimator = ModalityConfidenceEstimator(
            in_channels=embed_dim,
            hidden_dim=confidence_hidden_dim,
            dropout=dropout
        )
        self.sonar_confidence_estimator = ModalityConfidenceEstimator(
            in_channels=embed_dim,
            hidden_dim=confidence_hidden_dim,
            dropout=dropout
        )

        # Input projection layers (dynamic - created on first forward if needed)
        self.rgb_input_proj = None
        self.sonar_input_proj = None

        # Initial fusion layer: concatenate RGB and Sonar features
        self.initial_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
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
                'infonce_temperature': 0.07,
                'view_consistency_weight': 0.1
            }
        self.view_consistency_weight = consistency_config.pop('view_consistency_weight', 0.1)
        self.consistency_loss = DualConsistencyLoss(**consistency_config)

        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Sonar view fusion layer
        self.sonar_fusion_layer = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.5))

        # Dynamic adapters for dimension matching
        self.concat_adapters = nn.ModuleDict()
        self.rgb_residual_adapters = nn.ModuleDict()

        self.logger.info(
            f"Initialized DCMA: embed_dim={embed_dim}, num_heads={num_heads}, "
            f"use_confidence_modulation={use_confidence_modulation}, "
            f"use_deformable={use_deformable}"
        )

    def forward_single_scale(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        rgb_pyramid: Optional[List[torch.Tensor]] = None,
        return_consistency: bool = True,
        sonar_letterbox_metadata: Union[dict, List[dict], None] = None,
        rgb_letterbox_metadata: Union[dict, List[dict], None] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single-scale fusion with confidence estimation.

        Args:
            rgb_features: RGB feature maps [B, C, H_rgb, W_rgb]
            sonar_features: Sonar feature maps [B, C, H_sonar, W_sonar]
            rgb_pyramid: Optional multi-scale RGB features (unused, kept for API compatibility)
            return_consistency: Whether to compute consistency losses
            sonar_letterbox_metadata: Metadata (unused, kept for API compatibility)
            rgb_letterbox_metadata: Metadata (unused, kept for API compatibility)

        Returns:
            dict containing:
                - fused_features: Fused cross-modal features [B, C, H_rgb, W_rgb]
                - rgb_confidence: RGB modality confidence scores [B,]
                - sonar_confidence: Sonar modality confidence scores [B,]
                - combined_confidence: Geometric mean of confidences [B,]
                - rgb_view_fused: RGB-view fused features
                - sonar_view_fused: Sonar-view fused features
                - consistency_losses: Dict of consistency losses (if return_consistency=True)
                - deformable_info: Deformable attention information
        """
        B, C_rgb, H_rgb, W_rgb = rgb_features.shape
        C_sonar = sonar_features.shape[1]

        # Input projection if channels don't match embed_dim
        if C_rgb != self.embed_dim:
            if self.rgb_input_proj is None:
                self.rgb_input_proj = nn.Conv2d(C_rgb, self.embed_dim, kernel_size=1).to(rgb_features.device)
            rgb_features = self.rgb_input_proj(rgb_features)

        if C_sonar != self.embed_dim:
            if self.sonar_input_proj is None:
                self.sonar_input_proj = nn.Conv2d(C_sonar, self.embed_dim, kernel_size=1).to(sonar_features.device)
            sonar_features = self.sonar_input_proj(sonar_features)

        # Step 1: Confidence estimation
        rgb_confidence = self.rgb_confidence_estimator(rgb_features)
        sonar_confidence = self.sonar_confidence_estimator(sonar_features)
        combined_confidence = (rgb_confidence * sonar_confidence).sqrt()

        # Step 2: Initial fusion - resize sonar to match RGB and concatenate
        if sonar_features.shape[2:] != rgb_features.shape[2:]:
            sonar_resized = F.interpolate(
                sonar_features,
                size=rgb_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            sonar_resized = sonar_features

        # Concatenate and project to embed_dim
        initial_concat = torch.cat([rgb_features, sonar_resized], dim=1)
        initial_fused = self.initial_fusion(initial_concat)

        # Apply confidence modulation
        if self.use_confidence_modulation:
            conf_modulation = combined_confidence.view(B, 1, 1, 1)
            initial_fused = initial_fused * conf_modulation

        # Step 3: Bidirectional Deformable Attention
        if self.use_deformable:
            # Direction 1: RGB -> Sonar (RGB queries attend to Sonar)
            rgb_to_sonar_output = self.deformable_attention(
                query_features=rgb_features,
                key_features=sonar_features,
                value_features=initial_fused
            )
            rgb_attended_sonar = rgb_to_sonar_output['fused_features']

            # Direction 2: Sonar -> RGB (Sonar queries attend to RGB)
            sonar_to_rgb_output = self.deformable_attention(
                query_features=sonar_features,
                key_features=rgb_features,
                value_features=rgb_features
            )
            sonar_attended_rgb = sonar_to_rgb_output['fused_features']
        else:
            # Ablation: Skip deformable attention
            rgb_attended_sonar = initial_fused
            sonar_attended_rgb = sonar_features

        # Resize sonar_attended_rgb to match RGB spatial dimensions
        if sonar_attended_rgb.shape[2:] != rgb_features.shape[2:]:
            sonar_attended_rgb = F.interpolate(
                sonar_attended_rgb,
                size=rgb_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Step 4: Dual-View Fusion
        # RGB-view: RGB + attended Sonar information
        rgb_view_input = torch.cat([rgb_features, rgb_attended_sonar], dim=1)
        concat_dim_rgb = rgb_view_input.shape[1]
        expected_dim = self.embed_dim * 2
        if concat_dim_rgb != expected_dim:
            adapter_key = f'rgb_view_{concat_dim_rgb}_to_{expected_dim}'
            if adapter_key not in self.concat_adapters:
                self.concat_adapters[adapter_key] = nn.Conv2d(
                    concat_dim_rgb, expected_dim, kernel_size=1
                ).to(rgb_view_input.device)
            rgb_view_input = self.concat_adapters[adapter_key](rgb_view_input)
        rgb_view_fused_raw = self.fusion_layer(rgb_view_input)

        # Sonar-view: Sonar + attended RGB information
        sonar_view_input = torch.cat([sonar_resized, sonar_attended_rgb], dim=1)
        concat_dim_sonar = sonar_view_input.shape[1]
        if concat_dim_sonar != expected_dim:
            adapter_key = f'sonar_view_{concat_dim_sonar}_to_{expected_dim}'
            if adapter_key not in self.concat_adapters:
                self.concat_adapters[adapter_key] = nn.Conv2d(
                    concat_dim_sonar, expected_dim, kernel_size=1
                ).to(sonar_view_input.device)
            sonar_view_input = self.concat_adapters[adapter_key](sonar_view_input)
        sonar_view_fused_raw = self.sonar_fusion_layer(sonar_view_input)

        # Resize sonar_view to match RGB if needed
        if sonar_view_fused_raw.shape[2:] != rgb_features.shape[2:]:
            sonar_view_fused_raw = F.interpolate(
                sonar_view_fused_raw,
                size=rgb_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Step 5: Residual connections
        if rgb_features.shape[1] != rgb_view_fused_raw.shape[1]:
            adapter_key = f'rgb_residual_{rgb_features.shape[1]}_to_{rgb_view_fused_raw.shape[1]}'
            if adapter_key not in self.rgb_residual_adapters:
                self.rgb_residual_adapters[adapter_key] = nn.Conv2d(
                    rgb_features.shape[1], rgb_view_fused_raw.shape[1], kernel_size=1
                ).to(rgb_features.device)
            rgb_features_proj = self.rgb_residual_adapters[adapter_key](rgb_features)
        else:
            rgb_features_proj = rgb_features

        alpha = torch.sigmoid(self.residual_weight)
        rgb_view_fused = alpha * rgb_view_fused_raw + (1 - alpha) * rgb_features_proj

        # Sonar-view residual
        if sonar_resized.shape[1] != sonar_view_fused_raw.shape[1]:
            adapter_key = f'sonar_residual_{sonar_resized.shape[1]}_to_{sonar_view_fused_raw.shape[1]}'
            if adapter_key not in self.concat_adapters:
                self.concat_adapters[adapter_key] = nn.Conv2d(
                    sonar_resized.shape[1], sonar_view_fused_raw.shape[1], kernel_size=1
                ).to(sonar_resized.device)
            sonar_features_proj = self.concat_adapters[adapter_key](sonar_resized)
        else:
            sonar_features_proj = sonar_resized

        sonar_view_fused = alpha * sonar_view_fused_raw + (1 - alpha) * sonar_features_proj

        # Step 6: View consistency loss
        view_consistency_loss = F.mse_loss(
            rgb_view_fused, sonar_view_fused
        ) * self.view_consistency_weight

        # Build output
        output = {
            'rgb_view_fused': rgb_view_fused,
            'sonar_view_fused': sonar_view_fused,
            'fused_features': rgb_view_fused,  # Backward compatibility
            'rgb_confidence': rgb_confidence,
            'sonar_confidence': sonar_confidence,
            'combined_confidence': combined_confidence,
            'initial_fused': initial_fused,
            'rgb_attended_sonar': rgb_attended_sonar,
            'sonar_attended_rgb': sonar_attended_rgb,
            'view_consistency_loss': view_consistency_loss,
        }

        # Deformable info
        if self.use_deformable:
            output['deformable_info'] = {
                'rgb_to_sonar_offsets': rgb_to_sonar_output['offsets'],
                'sonar_to_rgb_offsets': sonar_to_rgb_output['offsets'],
                'rgb_sampling_locations': rgb_to_sonar_output['sampling_locations'],
                'sonar_sampling_locations': sonar_to_rgb_output['sampling_locations'],
                'rgb_attention_weights': rgb_to_sonar_output['attention_weights'],
                'sonar_attention_weights': sonar_to_rgb_output['attention_weights']
            }
        else:
            output['deformable_info'] = {
                'rgb_to_sonar_offsets': torch.zeros(B, 1, device=rgb_features.device),
                'sonar_to_rgb_offsets': torch.zeros(B, 1, device=rgb_features.device),
                'rgb_sampling_locations': torch.zeros(B, 1, device=rgb_features.device),
                'sonar_sampling_locations': torch.zeros(B, 1, device=rgb_features.device),
                'rgb_attention_weights': torch.zeros(B, 1, device=rgb_features.device),
                'sonar_attention_weights': torch.zeros(B, 1, device=rgb_features.device),
            }

        # Step 7: Consistency losses
        if return_consistency:
            rgb_embeddings = self.rgb_projection(rgb_features)
            sonar_embeddings = self.sonar_projection(sonar_features)

            # Use pooled features as component proxy
            pred_rgb_component = F.adaptive_avg_pool2d(rgb_features, 1).flatten(1)
            pred_sonar_component = F.adaptive_avg_pool2d(sonar_features, 1).flatten(1)

            # Project to consistent dimension (64) for consistency loss
            if not hasattr(self, 'rgb_component_proj'):
                self.rgb_component_proj = nn.Linear(
                    pred_rgb_component.shape[1], 64
                ).to(pred_rgb_component.device)
            if not hasattr(self, 'sonar_component_proj'):
                self.sonar_component_proj = nn.Linear(
                    pred_sonar_component.shape[1], 64
                ).to(pred_sonar_component.device)

            pred_rgb_component = self.rgb_component_proj(pred_rgb_component)
            pred_sonar_component = self.sonar_component_proj(pred_sonar_component)

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
        deformable_output = self.deformable_attention(
            query_pyramid=rgb_pyramid,
            key_pyramid=sonar_pyramid,
            value_pyramid=sonar_pyramid
        )

        fused_features = deformable_output['fused_features']
        fused_pyramid = deformable_output['fused_pyramid']

        B, C, H, W = fused_features.shape
        concat_features = torch.cat([rgb_pyramid[0], fused_features], dim=1)
        fused_features = self.fusion_layer(concat_features)

        alpha = torch.sigmoid(self.residual_weight)
        fused_features = alpha * fused_features + (1 - alpha) * rgb_pyramid[0]

        output = {
            'fused_features': fused_features,
            'fused_pyramid': fused_pyramid,
            'level_weights': deformable_output['level_weights']
        }

        if return_consistency:
            rgb_embeddings = self.rgb_projection(rgb_pyramid[0])
            sonar_embeddings = self.sonar_projection(sonar_pyramid[0])

            pred_rgb_component = F.adaptive_avg_pool2d(rgb_pyramid[0], 1).flatten(1)
            pred_sonar_component = F.adaptive_avg_pool2d(sonar_pyramid[0], 1).flatten(1)

            # Project to consistent dimension
            if not hasattr(self, 'rgb_component_proj'):
                self.rgb_component_proj = nn.Linear(
                    pred_rgb_component.shape[1], 64
                ).to(pred_rgb_component.device)
            if not hasattr(self, 'sonar_component_proj'):
                self.sonar_component_proj = nn.Linear(
                    pred_sonar_component.shape[1], 64
                ).to(pred_sonar_component.device)

            pred_rgb_component = self.rgb_component_proj(pred_rgb_component)
            pred_sonar_component = self.sonar_component_proj(pred_sonar_component)

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
        return_consistency: bool = True,
        sonar_letterbox_metadata: Union[dict, List[dict], None] = None,
        rgb_letterbox_metadata: Union[dict, List[dict], None] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for DCMA fusion.

        Args:
            rgb_features: RGB feature maps [B, C, H_rgb, W_rgb]
            sonar_features: Sonar feature maps [B, C, H_sonar, W_sonar]
            rgb_pyramid: Optional list of multi-scale RGB features
            sonar_pyramid: Optional list of multi-scale Sonar features
            return_consistency: Whether to compute consistency losses
            sonar_letterbox_metadata: Unused, kept for API compatibility
            rgb_letterbox_metadata: Unused, kept for API compatibility

        Returns:
            dict containing fusion results and consistency losses
        """
        if self.use_multi_scale and rgb_pyramid is not None and sonar_pyramid is not None:
            return self.forward_multi_scale(rgb_pyramid, sonar_pyramid, return_consistency)
        else:
            return self.forward_single_scale(
                rgb_features,
                sonar_features,
                rgb_pyramid,
                return_consistency,
                sonar_letterbox_metadata,
                rgb_letterbox_metadata
            )


@FUSION_REGISTRY.register("dcma_lite")
class DCMALite(BaseModule):
    """
    Lightweight variant of DCMA with reduced computational cost.

    Simplifications:
    - Fewer attention heads
    - Fewer deformable sampling points
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

        self.dcma = DCMA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_deformable_points=2,
            use_multi_scale=False,
            dropout=dropout,
            consistency_config={
                'component_weight': 0.1,
                'representation_weight': 0.05,
                'component_loss_type': 'l1',
                'infonce_temperature': 0.1,
                'view_consistency_weight': 0.1
            }
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        return_consistency: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for lightweight DCMA."""
        return self.dcma(
            rgb_features=rgb_features,
            sonar_features=sonar_features,
            return_consistency=return_consistency
        )


def build_dcma(config: dict) -> DCMA:
    """
    Factory function to build DCMA from config.

    Args:
        config: Configuration dictionary

    Returns:
        DCMA instance
    """
    consistency_config = config.get('consistency', {})

    return DCMA(
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 8),
        num_deformable_points=config.get('num_deformable_points', 4),
        use_multi_scale=config.get('use_multi_scale', False),
        num_levels=config.get('num_levels', 3),
        dropout=config.get('dropout', 0.1),
        consistency_config=consistency_config if consistency_config else None,
        confidence_hidden_dim=config.get('confidence_hidden_dim', 128),
        use_confidence_modulation=config.get('use_confidence_modulation', True),
        use_deformable=config.get('use_deformable', True)
    )


def build_dcma_lite(config: dict) -> DCMALite:
    """
    Factory function to build lightweight DCMA from config.

    Args:
        config: Configuration dictionary

    Returns:
        DCMALite instance
    """
    return DCMALite(
        embed_dim=config.get('embed_dim', 256),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1)
    )


# ==============================================================================
# Conditional Routing Variant
# ==============================================================================


@FUSION_REGISTRY.register("conditional_dcma")
class ConditionalDCMA(BaseModule):
    """
    Conditional DCMA with single-modality bypass paths.

    Routes samples to appropriate fusion paths based on modality availability:
    - Dual-modality: Use full DCMA fusion
    - Single-RGB: Use RGB bypass branch
    - Single-Sonar: Use Sonar bypass branch
    - Zero-modality: Return minimal/zero features

    Args:
        embed_dim: Embedding dimension for DCMA
        num_heads: Number of attention heads
        num_deformable_points: Number of sampling points per head
        use_multi_scale: Whether to use multi-scale deformable attention
        num_levels: Number of pyramid levels (if multi-scale)
        dropout: Dropout probability
        consistency_config: Configuration for dual consistency loss
        rgb_channels: RGB feature channels (for bypass branch)
        sonar_channels: Sonar feature channels (for bypass branch)
        confidence_threshold: Threshold for automatic validity inference

    Returns:
        dict containing:
            - fused_features: Fused/bypassed features [B, embed_dim, H, W]
            - routing_info: Information about which path was taken
            - consistency_losses: Consistency losses (only for dual-modality)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_deformable_points: int = 4,
        use_multi_scale: bool = False,
        num_levels: int = 3,
        dropout: float = 0.1,
        consistency_config: Optional[Dict] = None,
        rgb_channels: int = 384,
        sonar_channels: int = 384,
        confidence_threshold: float = 0.3,
        **kwargs
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.rgb_channels = rgb_channels
        self.sonar_channels = sonar_channels
        self.confidence_threshold = confidence_threshold

        # Core DCMA for dual-modality fusion
        self.dcma = DCMA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_deformable_points=num_deformable_points,
            use_multi_scale=use_multi_scale,
            num_levels=num_levels,
            dropout=dropout,
            consistency_config=consistency_config
        )

        # Lightweight confidence estimators for routing decisions
        self.routing_rgb_confidence = ModalityConfidenceEstimator(
            in_channels=rgb_channels,
            hidden_dim=128,
            dropout=dropout
        )
        self.routing_sonar_confidence = ModalityConfidenceEstimator(
            in_channels=sonar_channels,
            hidden_dim=128,
            dropout=dropout
        )

        # Single-modality bypass branches
        self.rgb_bypass = nn.Sequential(
            nn.Conv2d(rgb_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.sonar_bypass = nn.Sequential(
            nn.Conv2d(sonar_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        self.routing_confidence = nn.Parameter(torch.tensor(1.0))

        self.logger.info(
            f"Initialized ConditionalDCMA: embed_dim={embed_dim}, "
            f"rgb_channels={rgb_channels}, sonar_channels={sonar_channels}, "
            f"num_heads={num_heads}, confidence_threshold={confidence_threshold}"
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        is_rgb_valid: Optional[torch.Tensor] = None,
        is_sonar_valid: Optional[torch.Tensor] = None,
        rgb_pyramid: Optional[List[torch.Tensor]] = None,
        return_consistency: bool = True,
        sonar_letterbox_metadata: Union[dict, List[dict], None] = None,
        rgb_letterbox_metadata: Union[dict, List[dict], None] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Conditional fusion with modality-aware routing.

        ARCHITECTURE UPDATE (Audit 2025121001 Issue 3):
        Geometry gating has been removed. Letterbox metadata parameters are kept
        for API compatibility but are no longer used. Fusion routing now relies on:
        1. Head validity outputs (rgb_validity/sonar_validity) - preferred
        2. Backbone confidence estimators - fallback when validity not passed

        Args:
            rgb_features: RGB feature maps [B, C_rgb, H_rgb, W_rgb]
            sonar_features: Sonar feature maps [B, C_sonar, H_sonar, W_sonar]
            is_rgb_valid: [B] boolean tensor, True if RGB has valid target (optional)
                If None, inferred from backbone confidence estimators.
            is_sonar_valid: [B] boolean tensor, True if Sonar has valid target (optional)
                If None, inferred from backbone confidence estimators.
            rgb_pyramid: Optional multi-scale RGB features for deformable attention
            return_consistency: Whether to compute consistency losses
            sonar_letterbox_metadata: DEPRECATED - kept for API compatibility only.
                Previously used for geometry gating (now removed).
            rgb_letterbox_metadata: DEPRECATED - kept for API compatibility only.
                Previously used for geometry gating (now removed).

        Returns:
            dict containing:
                - fused_features: [B, embed_dim, H, W] fused feature maps
                - rgb_view_fused: [B, embed_dim, H, W] RGB-view fused features
                - sonar_view_fused: [B, embed_dim, H, W] Sonar-view fused features
                - routing_info: dict with routing statistics
                - consistency_losses: dict with consistency loss values (if return_consistency)
                - inferred_is_rgb_valid: [B] inferred RGB validity
                - inferred_is_sonar_valid: [B] inferred Sonar validity
        """
        B, C_rgb, H_rgb, W_rgb = rgb_features.shape
        B_s, C_sonar, H_sonar, W_sonar = sonar_features.shape

        assert B == B_s, f"Batch size mismatch: RGB={B}, Sonar={B_s}"

        # CRITICAL FIX (Audit 2025121001 Issue 4): Always compute routing confidence
        # During training with GT is_valid: provides supervision signal for confidence estimators
        # During inference without is_valid: enables automatic modality routing
        rgb_conf_scores = self.routing_rgb_confidence(rgb_features)
        sonar_conf_scores = self.routing_sonar_confidence(sonar_features)

        # Store original GT for loss computation (before potentially being overwritten)
        gt_rgb_valid = is_rgb_valid
        gt_sonar_valid = is_sonar_valid

        # Use GT validity if provided, otherwise infer from confidence scores
        if is_rgb_valid is None:
            is_rgb_valid = rgb_conf_scores > self.confidence_threshold
        if is_sonar_valid is None:
            is_sonar_valid = sonar_conf_scores > self.confidence_threshold

        # CRITICAL FIX (Audit 2025121001 Issue 4): Compute routing validity loss
        # BCE loss between predicted confidence and GT validity, enables learned routing
        routing_validity_loss = torch.tensor(0.0, device=rgb_features.device)
        if gt_rgb_valid is not None and gt_sonar_valid is not None:
            gt_rgb_float = gt_rgb_valid.float() if gt_rgb_valid.dtype == torch.bool else gt_rgb_valid
            gt_sonar_float = gt_sonar_valid.float() if gt_sonar_valid.dtype == torch.bool else gt_sonar_valid
            # Ensure confidence scores are squeezed to match GT shape
            rgb_conf_squeezed = rgb_conf_scores.squeeze(-1) if rgb_conf_scores.dim() > 1 else rgb_conf_scores
            sonar_conf_squeezed = sonar_conf_scores.squeeze(-1) if sonar_conf_scores.dim() > 1 else sonar_conf_scores
            rgb_bce = F.binary_cross_entropy(rgb_conf_squeezed, gt_rgb_float, reduction='mean')
            sonar_bce = F.binary_cross_entropy(sonar_conf_squeezed, gt_sonar_float, reduction='mean')
            routing_validity_loss = (rgb_bce + sonar_bce) / 2

        # Convert to bool if needed
        if is_rgb_valid.dtype != torch.bool:
            is_rgb_valid = is_rgb_valid.bool()
        if is_sonar_valid.dtype != torch.bool:
            is_sonar_valid = is_sonar_valid.bool()

        # Detect modality availability
        zero_modality_mask = (~is_rgb_valid) & (~is_sonar_valid)
        single_rgb_mask = is_rgb_valid & (~is_sonar_valid)
        single_sonar_mask = (~is_rgb_valid) & is_sonar_valid
        dual_modality_mask = is_rgb_valid & is_sonar_valid

        # Initialize output tensors
        fused_features = torch.zeros(B, self.embed_dim, H_rgb, W_rgb, device=rgb_features.device, dtype=rgb_features.dtype)
        rgb_view_features = torch.zeros(B, self.embed_dim, H_rgb, W_rgb, device=rgb_features.device, dtype=rgb_features.dtype)
        sonar_view_features = torch.zeros(B, self.embed_dim, H_rgb, W_rgb, device=rgb_features.device, dtype=rgb_features.dtype)

        routing_info = {
            'num_dual': dual_modality_mask.sum().item(),
            'num_single_rgb': single_rgb_mask.sum().item(),
            'num_single_sonar': single_sonar_mask.sum().item(),
            'num_zero': zero_modality_mask.sum().item(),
            'routing_confidence': torch.sigmoid(self.routing_confidence).item()
        }

        consistency_losses = {
            'component_consistency': torch.tensor(0.0, device=rgb_features.device),
            'representation_consistency': torch.tensor(0.0, device=rgb_features.device)
        }
        view_consistency_loss_accum = torch.tensor(0.0, device=rgb_features.device)

        # Route 1: Dual-modality -> Full DCMA fusion
        if dual_modality_mask.any():
            dual_idx = dual_modality_mask.nonzero(as_tuple=False).squeeze(-1)
            rgb_dual = rgb_features[dual_idx]
            sonar_dual = sonar_features[dual_idx]

            dcma_output = self.dcma.forward_single_scale(
                rgb_features=rgb_dual,
                sonar_features=sonar_dual,
                rgb_pyramid=[p[dual_idx] for p in rgb_pyramid] if rgb_pyramid else None,
                return_consistency=return_consistency
            )

            fused_features[dual_idx] = dcma_output['fused_features']
            rgb_view_features[dual_idx] = dcma_output['rgb_view_fused']
            sonar_view_features[dual_idx] = dcma_output['sonar_view_fused']

            if return_consistency and 'consistency_losses' in dcma_output:
                weight = dual_idx.shape[0] / B
                consistency_losses['component_consistency'] = dcma_output['consistency_losses'].get('component_loss', torch.tensor(0.0, device=rgb_features.device)) * weight
                consistency_losses['representation_consistency'] = dcma_output['consistency_losses'].get('infonce_loss', torch.tensor(0.0, device=rgb_features.device)) * weight

            if 'view_consistency_loss' in dcma_output:
                weight = dual_idx.shape[0] / B
                view_consistency_loss_accum += dcma_output['view_consistency_loss'] * weight

        # Route 2: Single-RGB -> RGB bypass
        if single_rgb_mask.any():
            rgb_idx = single_rgb_mask.nonzero(as_tuple=False).squeeze(-1)
            rgb_single = rgb_features[rgb_idx]
            rgb_bypassed = self.rgb_bypass(rgb_single)

            fused_features[rgb_idx] = rgb_bypassed
            rgb_view_features[rgb_idx] = rgb_bypassed
            sonar_view_features[rgb_idx] = torch.zeros_like(rgb_bypassed)

        # Route 3: Single-Sonar -> Sonar bypass
        if single_sonar_mask.any():
            sonar_idx = single_sonar_mask.nonzero(as_tuple=False).squeeze(-1)
            sonar_single = sonar_features[sonar_idx]
            sonar_bypassed = self.sonar_bypass(sonar_single)

            if sonar_bypassed.shape[2:] != (H_rgb, W_rgb):
                sonar_bypassed = F.interpolate(
                    sonar_bypassed,
                    size=(H_rgb, W_rgb),
                    mode='bilinear',
                    align_corners=False
                )

            fused_features[sonar_idx] = sonar_bypassed
            rgb_view_features[sonar_idx] = torch.zeros_like(sonar_bypassed)
            sonar_view_features[sonar_idx] = sonar_bypassed

        # Route 4: Zero-modality -> zeros (already initialized)

        if self.logger.isEnabledFor(10):
            self.logger.debug(
                f"ConditionalDCMA routing: "
                f"dual={routing_info['num_dual']}, "
                f"RGB-only={routing_info['num_single_rgb']}, "
                f"Sonar-only={routing_info['num_single_sonar']}, "
                f"zero={routing_info['num_zero']}"
            )

        output = {
            'fused_features': fused_features,
            'rgb_view_fused': rgb_view_features,
            'sonar_view_fused': sonar_view_features,
            'routing_info': routing_info,
            'consistency_losses': consistency_losses if return_consistency else None,
            'view_consistency_loss': view_consistency_loss_accum,
            'inferred_is_rgb_valid': is_rgb_valid,
            'inferred_is_sonar_valid': is_sonar_valid,
            # CRITICAL FIX (Audit 2025121001 Issue 4): Routing validity supervision loss
            'routing_validity_loss': routing_validity_loss,
            # Also output confidence scores for downstream use (e.g., output selection)
            'routing_rgb_confidence': rgb_conf_scores,
            'routing_sonar_confidence': sonar_conf_scores
        }

        return output


def build_conditional_dcma(
    rgb_channels: int = 384,
    sonar_channels: int = 384,
    embed_dim: int = 256,
    **kwargs
) -> ConditionalDCMA:
    """
    Factory function to build conditional DCMA.

    Args:
        rgb_channels: RGB feature dimension
        sonar_channels: Sonar feature dimension
        embed_dim: Embedding dimension
        **kwargs: Additional arguments for ConditionalDCMA

    Returns:
        ConditionalDCMA instance
    """
    return ConditionalDCMA(
        rgb_channels=rgb_channels,
        sonar_channels=sonar_channels,
        embed_dim=embed_dim,
        **kwargs
    )


# ==============================================================================
# Backward Compatibility Aliases (deprecated, will be removed in future)
# ==============================================================================

# Backward compatibility: keep old names as aliases
G2DCMA = DCMA
G2DCMALite = DCMALite
ConditionalG2DCMA = ConditionalDCMA
build_g2_dcma = build_dcma
build_g2_dcma_lite = build_dcma_lite
build_conditional_g2_dcma = build_conditional_dcma


# =============================================================================
# IdentityFusion for Single-Modality Ablation Experiments
# =============================================================================
# CRITICAL FIX (Audit 2025121001 Issue 8): IdentityFusion for single-modality ablation
# This is a pass-through fusion module that directly returns the input features
# without cross-modal fusion, for ablation experiments where only one modality is used.

@FUSION_REGISTRY.register("identity_fusion")
class IdentityFusion(BaseModule):
    """
    Identity fusion for single-modality ablation experiments.

    This module is used when only one modality (RGB or Sonar) is available.
    It simply projects the available modality features to the fusion dimension
    without performing any cross-modal fusion.

    Args:
        rgb_channels: Number of RGB input channels
        sonar_channels: Number of Sonar input channels
        embed_dim: Output embedding dimension
        active_modality: Which modality to use ('rgb' or 'sonar')

    Example:
        >>> fusion = IdentityFusion(768, 768, 256, active_modality='rgb')
        >>> outputs = fusion(rgb_features, sonar_features)
        >>> # outputs['fused_features'] == projected rgb_features
    """

    def __init__(
        self,
        rgb_channels: int = 768,
        sonar_channels: int = 768,
        embed_dim: int = 256,
        active_modality: str = 'rgb'  # 'rgb' or 'sonar'
    ):
        super().__init__()
        self.rgb_channels = rgb_channels
        self.sonar_channels = sonar_channels
        self.embed_dim = embed_dim
        self.active_modality = active_modality

        # Projection layers
        if active_modality == 'rgb':
            self.projection = nn.Conv2d(rgb_channels, embed_dim, kernel_size=1)
        elif active_modality == 'sonar':
            self.projection = nn.Conv2d(sonar_channels, embed_dim, kernel_size=1)
        else:
            raise ValueError(f"Unknown active_modality: {active_modality}. Must be 'rgb' or 'sonar'")

        self.logger.info(f"Initialized IdentityFusion (active_modality={active_modality})")

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        is_rgb_valid: Optional[torch.Tensor] = None,
        is_sonar_valid: Optional[torch.Tensor] = None,
        rgb_pyramid: Optional[List[torch.Tensor]] = None,
        return_consistency: bool = True,
        sonar_letterbox_metadata: Union[dict, List[dict], None] = None,
        rgb_letterbox_metadata: Union[dict, List[dict], None] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Identity forward pass - returns projected single-modality features.

        Args:
            rgb_features: RGB features [B, C_rgb, H, W]
            sonar_features: Sonar features [B, C_sonar, H, W]
            is_rgb_valid: Ignored (kept for API compatibility)
            is_sonar_valid: Ignored (kept for API compatibility)
            rgb_pyramid: Ignored (kept for API compatibility)
            return_consistency: Ignored (no consistency in single modality)
            sonar_letterbox_metadata: Ignored (kept for API compatibility)
            rgb_letterbox_metadata: Ignored (kept for API compatibility)

        Returns:
            dict containing:
                - fused_features: [B, embed_dim, H, W] projected features
                - rgb_view_fused: Same as fused_features (for compatibility)
                - sonar_view_fused: Same as fused_features (for compatibility)
                - routing_info: dict with modality usage statistics
                - consistency_losses: Empty dict (no cross-modal consistency)
        """
        B = rgb_features.shape[0]

        # Select and project the active modality
        if self.active_modality == 'rgb':
            source_features = rgb_features
            is_valid = is_rgb_valid if is_rgb_valid is not None else torch.ones(B, dtype=torch.bool, device=rgb_features.device)
        else:
            source_features = sonar_features
            # Handle potential size mismatch (sonar may have different resolution)
            target_h, target_w = rgb_features.shape[2:]
            if source_features.shape[2:] != (target_h, target_w):
                source_features = F.interpolate(
                    source_features, size=(target_h, target_w),
                    mode='bilinear', align_corners=False
                )
            is_valid = is_sonar_valid if is_sonar_valid is not None else torch.ones(B, dtype=torch.bool, device=sonar_features.device)

        # Project to embedding dimension
        fused_features = self.projection(source_features)

        # Routing info for compatibility
        routing_info = {
            'num_dual': 0,
            'num_single_rgb': B if self.active_modality == 'rgb' else 0,
            'num_single_sonar': B if self.active_modality == 'sonar' else 0,
            'num_zero': 0,
            'routing_confidence': 1.0,
            'active_modality': self.active_modality
        }

        return {
            'fused_features': fused_features,
            'rgb_view_fused': fused_features,  # Same for compatibility
            'sonar_view_fused': fused_features,  # Same for compatibility
            'routing_info': routing_info,
            'consistency_losses': {},  # No consistency losses in single-modality
            'inferred_is_rgb_valid': torch.ones(B, dtype=torch.bool, device=rgb_features.device) if self.active_modality == 'rgb' else torch.zeros(B, dtype=torch.bool, device=rgb_features.device),
            'inferred_is_sonar_valid': torch.ones(B, dtype=torch.bool, device=sonar_features.device) if self.active_modality == 'sonar' else torch.zeros(B, dtype=torch.bool, device=sonar_features.device)
        }


def build_identity_fusion(
    rgb_channels: int = 768,
    sonar_channels: int = 768,
    embed_dim: int = 256,
    active_modality: str = 'rgb',
    **kwargs
) -> IdentityFusion:
    """
    Build IdentityFusion from configuration.

    Args:
        rgb_channels: RGB input channels
        sonar_channels: Sonar input channels
        embed_dim: Output embedding dimension
        active_modality: Which modality to use ('rgb' or 'sonar')
        **kwargs: Ignored (for API compatibility)

    Returns:
        IdentityFusion instance
    """
    return IdentityFusion(
        rgb_channels=rgb_channels,
        sonar_channels=sonar_channels,
        embed_dim=embed_dim,
        active_modality=active_modality
    )
