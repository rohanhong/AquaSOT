"""
G²-DCMA: Geometry-Gated Deformable Cross-Modal Attention with Confidence Estimation

MAJOR ARCHITECTURE UPDATE (Audit 2025112301):
This module now integrates learnable confidence estimation to replace hand-crafted
physical quality estimation, following mainstream deep learning paradigms.

Core components:
1. Geometry-Gated Attention: Uses radial and scale priors to modulate cross-attention
2. Confidence Estimation: Learnable end-to-end confidence scores for each modality
3. Deformable Attention: Learns sparse sampling offsets for spatial alignment
4. Dual Consistency: Enforces component and representation consistency

Architecture:
    RGB features + Sonar features
           ↓
    [Confidence Estimation] (learnable, replaces physical quality)
           ↓
    [Geometric Gating]  (radial + scale priors)
           ↓
    [Confidence Modulation] (modulates attention scores)
           ↓
    [Deformable Attention]  (learnable offsets)
           ↓
    [Fusion + Consistency]  (component + representation)
           ↓
    Fused features + Confidence scores + Consistency losses

Reference:
    - AquaSOT-3D Audit 2025112301: Confidence-Aware Fusion Architecture
    - PyTorch MultiheadAttention standard implementation
    - Uncertainty-Aware Attention (NeurIPS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Union

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


class ModalityConfidenceEstimator(BaseModule):
    """
    Learnable confidence estimator for a single modality.

    ARCHITECTURE UPDATE (Audit 2025112301):
    Replaces hand-crafted physical quality estimation (Laplacian variance, SNR)
    with end-to-end learnable confidence scores, following PyTorch standard practices.

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


@FUSION_REGISTRY.register("g2_dcma")
class G2DCMA(BaseModule):
    """
    Geometry-Gated Deformable Cross-Modal Attention (G²-DCMA) with Confidence Estimation.

    ARCHITECTURE UPDATE (Audit 2025112301):
    Integrates learnable confidence estimation to replace physical quality estimation.

    Core fusion module that integrates:
    - Confidence estimation (NEW): learnable modality reliability scores
    - Geometric gating: spatial prior-based attention modulation
    - Deformable attention: adaptive spatial sampling
    - Dual consistency: component and representation alignment

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
        confidence_hidden_dim: Hidden dimension for confidence estimators (default: 128)
        use_confidence_modulation: Whether to modulate attention with confidence (default: True)
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
        consistency_config: Optional[Dict] = None,
        confidence_hidden_dim: int = 128,
        use_confidence_modulation: bool = True,
        use_geometry_gating: bool = True,
        use_deformable: bool = True
    ):
        """
        CRITICAL FIX (Audit 2025112502 Finding 5): Added ablation switches
        - use_geometry_gating: Enable/disable geometric gating module
        - use_deformable: Enable/disable deformable attention
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_multi_scale = use_multi_scale
        self.use_confidence_modulation = use_confidence_modulation
        # CRITICAL FIX (Audit 2025112502 Finding 5): Store ablation flags
        self.use_geometry_gating = use_geometry_gating
        self.use_deformable = use_deformable
        # CRITICAL FIX (Audit 2025112502 Final): Store dropout for sonar_fusion_layer
        self.dropout = dropout

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

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.residual_weight = nn.Parameter(torch.tensor(0.5))

        self.concat_adapters = nn.ModuleDict()
        self.rgb_residual_adapters = nn.ModuleDict()

        self.logger.info(
            f"Initialized G2DCMA with confidence estimation: "
            f"embed_dim={embed_dim}, num_heads={num_heads}, "
            f"use_confidence_modulation={use_confidence_modulation}"
        )

    def forward_single_scale(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        rgb_pyramid: Optional[List[torch.Tensor]] = None,
        return_consistency: bool = True,
        sonar_letterbox_metadata: Union[dict, List[dict], None] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single-scale fusion with confidence estimation.

        ARCHITECTURE UPDATE (Audit 2025112301):
        Now includes learnable confidence estimation and modulation.

        CRITICAL FIX (Audit 2025112403 Finding 3): Added sonar_letterbox_metadata parameter
        to enable letterbox-aware geometric gating.

        ENHANCED FIX (Audit 2025112403 Follow-up): Per-sample metadata support
        Now accepts list of dicts for heterogeneous batches.

        Args:
            rgb_features: RGB feature maps [B, C, H_rgb, W_rgb]
            sonar_features: Sonar feature maps [B, C, H_sonar, W_sonar]
            rgb_pyramid: Optional multi-scale RGB features for geometric gating
            return_consistency: Whether to compute consistency losses
            sonar_letterbox_metadata: Either single dict or list of dicts for per-sample metadata

        Returns:
            dict containing:
                - fused_features: Fused cross-modal features [B, C, H_rgb, W_rgb]
                - rgb_confidence: RGB modality confidence scores [B,]
                - sonar_confidence: Sonar modality confidence scores [B,]
                - combined_confidence: Geometric mean of confidences [B,]
                - geometry_gated: Geometry-gated features
                - deformable_attended: Deformable attention output
                - consistency_losses: Dict of consistency losses (if return_consistency=True)
                - geometric_info: Geometric gating information
                - deformable_info: Deformable attention information
        """
        B, C, H_rgb, W_rgb = rgb_features.shape

        rgb_confidence = self.rgb_confidence_estimator(rgb_features)
        sonar_confidence = self.sonar_confidence_estimator(sonar_features)
        combined_confidence = (rgb_confidence * sonar_confidence).sqrt()

        # CRITICAL FIX (Audit 2025112502 Revision): Make ablation switches actually work
        # use_geometry_gating and use_deformable must control execution, not just be stored
        if self.use_geometry_gating:
            # CRITICAL FIX (Audit 2025112403 Finding 3): Pass letterbox metadata for radial binning
            geometry_output = self.geometry_gating(
                rgb_features=rgb_features,
                sonar_features=sonar_features,
                rgb_pyramid=rgb_pyramid,
                sonar_letterbox_metadata=sonar_letterbox_metadata
            )
            geometry_gated = geometry_output['fused_features']

            if self.use_confidence_modulation:
                conf_modulation = combined_confidence.view(B, 1, 1, 1)
                geometry_gated = geometry_gated * conf_modulation
        else:
            # CRITICAL FIX (Audit 2025112503 Finding 2): Safe ablation branch
            # When geometry gating is disabled, construct a placeholder geometry_output
            # to prevent UnboundLocalError when accessing geometry_output later

            # Ablation: Skip geometry gating, use simple concatenation
            # Resize sonar to match RGB if needed
            if sonar_features.shape[2:] != rgb_features.shape[2:]:
                sonar_resized = F.interpolate(
                    sonar_features,
                    size=rgb_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                sonar_resized = sonar_features
            # Simple concatenation as baseline (no geometric gating)
            geometry_gated = torch.cat([rgb_features, sonar_resized], dim=1)
            # Project back to embed_dim
            if not hasattr(self, 'geometry_bypass_proj'):
                self.geometry_bypass_proj = nn.Conv2d(
                    geometry_gated.shape[1], self.embed_dim, kernel_size=1
                ).to(geometry_gated.device)
            geometry_gated = self.geometry_bypass_proj(geometry_gated)

            # CRITICAL FIX (Audit 2025112503 Finding 2): Construct placeholder geometry_output
            # This prevents crashes when returning geometric_info later
            geometry_output = {
                'fused_features': geometry_gated,
                'gating_weights': torch.zeros(B, device=rgb_features.device),
                'radial_encoding': torch.zeros(B, 64, device=rgb_features.device),
                'scale_encoding': torch.zeros(B, 64, device=rgb_features.device),
                'attention_probs': torch.zeros(B, 1, device=rgb_features.device),
                '_ablation_mode': 'no_gating'  # Flag for debugging
            }

        # ARCHITECTURE UPDATE (Audit 2025112313 - Dual-View Fusion):
        # Step 2: Bidirectional Deformable Attention for Dual-View Fusion
        # Learn implicit RGB-Sonar correspondence without calibration

        if self.use_deformable:
            # Direction 1: RGB→Sonar (RGB queries attend to Sonar)
            rgb_to_sonar_output = self.deformable_attention(
                query_features=rgb_features,  # RGB queries
                key_features=sonar_features,  # Sonar keys (learn RGB→Sonar mapping)
                value_features=geometry_gated  # Geometry-gated values
            )
            rgb_attended_sonar = rgb_to_sonar_output['fused_features']  # [B, C, H_rgb, W_rgb]

            # Direction 2: Sonar→RGB (Sonar queries attend to RGB)
            sonar_to_rgb_output = self.deformable_attention(
                query_features=sonar_features,  # Sonar queries
                key_features=rgb_features,      # RGB keys (learn Sonar→RGB mapping)
                value_features=rgb_features     # RGB values
            )
            sonar_attended_rgb = sonar_to_rgb_output['fused_features']  # [B, C, H_sonar, W_sonar]
        else:
            # Ablation: Skip deformable attention, use direct features
            rgb_attended_sonar = geometry_gated  # Use geometry-gated features directly
            # For sonar view, use raw sonar features
            sonar_attended_rgb = sonar_features

        # Resize sonar_attended_rgb to match RGB spatial dimensions
        if sonar_attended_rgb.shape[2:] != rgb_features.shape[2:]:
            sonar_attended_rgb = F.interpolate(
                sonar_attended_rgb,
                size=rgb_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Step 3: Dual-View Fusion
        # RGB-view fusion: RGB perspective with complementary Sonar information
        rgb_view_input = torch.cat([rgb_features, rgb_attended_sonar], dim=1)

        # Dimension adaptation for RGB view
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

        # Sonar-view fusion: Sonar perspective with complementary RGB information
        sonar_view_input = torch.cat([sonar_features, sonar_attended_rgb], dim=1)

        # Dimension adaptation for Sonar view
        concat_dim_sonar = sonar_view_input.shape[1]
        if concat_dim_sonar != expected_dim:
            adapter_key = f'sonar_view_{concat_dim_sonar}_to_{expected_dim}'
            if adapter_key not in self.concat_adapters:
                self.concat_adapters[adapter_key] = nn.Conv2d(
                    concat_dim_sonar, expected_dim, kernel_size=1
                ).to(sonar_view_input.device)
            sonar_view_input = self.concat_adapters[adapter_key](sonar_view_input)

        # Create sonar fusion layer if not exists
        if not hasattr(self, 'sonar_fusion_layer'):
            self.sonar_fusion_layer = nn.Sequential(
                nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1),
                nn.BatchNorm2d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout if hasattr(self, 'dropout') else 0.1)
            ).to(sonar_view_input.device)

        sonar_view_fused_raw = self.sonar_fusion_layer(sonar_view_input)

        # Resize sonar_view to match RGB spatial dimensions for consistency
        if sonar_view_fused_raw.shape[2:] != rgb_features.shape[2:]:
            sonar_view_fused_raw = F.interpolate(
                sonar_view_fused_raw,
                size=rgb_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # Step 4: Residual Connections with Dual-View Specific Adaptations
        # RGB-view residual
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

        # Sonar-view residual (use Sonar features as residual connection)
        if sonar_features.shape[1] != sonar_view_fused_raw.shape[1]:
            adapter_key = f'sonar_residual_{sonar_features.shape[1]}_to_{sonar_view_fused_raw.shape[1]}'
            if adapter_key not in self.concat_adapters:
                self.concat_adapters[adapter_key] = nn.Conv2d(
                    sonar_features.shape[1], sonar_view_fused_raw.shape[1], kernel_size=1
                ).to(sonar_features.device)
            sonar_features_proj = self.concat_adapters[adapter_key](sonar_features)
        else:
            sonar_features_proj = sonar_features

        # Resize sonar_features_proj if needed
        if sonar_features_proj.shape[2:] != sonar_view_fused_raw.shape[2:]:
            sonar_features_proj = F.interpolate(
                sonar_features_proj,
                size=sonar_view_fused_raw.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        sonar_view_fused = alpha * sonar_view_fused_raw + (1 - alpha) * sonar_features_proj

        # Step 5: Cross-View Consistency (for regularization)
        # Ensure both views can reconstruct each other
        view_consistency_loss = F.mse_loss(
            rgb_view_fused, sonar_view_fused
        ) * 0.1  # Small weight for consistency regularization

        output = {
            'rgb_view_fused': rgb_view_fused,  # [B, C, H_rgb, W_rgb] - For RGB BoxHead
            'sonar_view_fused': sonar_view_fused,  # [B, C, H_rgb, W_rgb] - For Sonar BoxHead
            'fused_features': rgb_view_fused,  # Backward compatibility (use RGB view as default)
            'rgb_confidence': rgb_confidence,
            'sonar_confidence': sonar_confidence,
            'combined_confidence': combined_confidence,
            'geometry_gated': geometry_gated,
            'rgb_attended_sonar': rgb_attended_sonar,  # For analysis
            'sonar_attended_rgb': sonar_attended_rgb,  # For analysis
            'view_consistency_loss': view_consistency_loss,  # Cross-view regularization
            'geometric_info': {
                'gating_weights': geometry_output['gating_weights'],
                'radial_encoding': geometry_output['radial_encoding'],
                'scale_encoding': geometry_output['scale_encoding'],
                'attention_probs': geometry_output['attention_probs']
            }
        }

        # CRITICAL FIX (Audit 2025112507 Finding 1): Conditional deformable_info construction
        # When use_deformable=False (ablation mode), rgb_to_sonar_output and sonar_to_rgb_output
        # are not defined, causing NameError. Provide placeholder tensors for ablation experiments.
        if self.use_deformable:
            output['deformable_info'] = {
                'rgb_to_sonar_offsets': rgb_to_sonar_output['offsets'],  # RGB→Sonar mapping
                'sonar_to_rgb_offsets': sonar_to_rgb_output['offsets'],  # Sonar→RGB mapping
                'rgb_sampling_locations': rgb_to_sonar_output['sampling_locations'],
                'sonar_sampling_locations': sonar_to_rgb_output['sampling_locations'],
                'rgb_attention_weights': rgb_to_sonar_output['attention_weights'],
                'sonar_attention_weights': sonar_to_rgb_output['attention_weights']
            }
        else:
            # Ablation mode: provide zero placeholders for downstream code compatibility
            output['deformable_info'] = {
                'rgb_to_sonar_offsets': torch.zeros(B, 1, device=rgb_features.device),
                'sonar_to_rgb_offsets': torch.zeros(B, 1, device=rgb_features.device),
                'rgb_sampling_locations': torch.zeros(B, 1, device=rgb_features.device),
                'sonar_sampling_locations': torch.zeros(B, 1, device=rgb_features.device),
                'rgb_attention_weights': torch.zeros(B, 1, device=rgb_features.device),
                'sonar_attention_weights': torch.zeros(B, 1, device=rgb_features.device),
                '_ablation_mode': 'no_deformable'  # Flag for debugging
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
        return_consistency: bool = True,
        sonar_letterbox_metadata: dict = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for G²-DCMA fusion.

        CRITICAL FIX (Audit 2025112403 Finding 3): Added sonar_letterbox_metadata parameter

        Args:
            rgb_features: RGB feature maps [B, C, H_rgb, W_rgb]
            sonar_features: Sonar feature maps [B, C, H_sonar, W_sonar]
            rgb_pyramid: Optional list of multi-scale RGB features
            sonar_pyramid: Optional list of multi-scale Sonar features
            return_consistency: Whether to compute consistency losses
            sonar_letterbox_metadata: Letterbox metadata for sonar images

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
                sonar_letterbox_metadata
            )


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
    Factory function to build G²-DCMA with confidence estimation from config.

    ARCHITECTURE UPDATE (Audit 2025112301):
    Now supports confidence estimation parameters.

    CRITICAL FIX (Audit 2025112502 Finding 5): Added ablation switches
    - use_geometry_gating: Enable/disable geometric gating module
    - use_deformable: Enable/disable deformable attention
    These switches enable paper ablation experiments (Table 3).

    Args:
        config: Configuration dictionary with optional keys:
            - use_geometry_gating (bool): Enable geometric gating (default: True)
            - use_deformable (bool): Use deformable attention (default: True)
            - use_confidence_modulation (bool): Enable confidence modulation (default: True)

    Returns:
        G2DCMA instance with confidence estimation
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
        consistency_config=consistency_config if consistency_config else None,
        confidence_hidden_dim=config.get('confidence_hidden_dim', 128),
        use_confidence_modulation=config.get('use_confidence_modulation', True),
        # CRITICAL FIX (Audit 2025112502 Finding 5): Ablation switches
        use_geometry_gating=config.get('use_geometry_gating', True),
        use_deformable=config.get('use_deformable', True)
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


# ==============================================================================
# SECTION 4: Conditional Routing Variant (Audit 2025112101 Phase 2.1)
# ==============================================================================
# FILE CONSOLIDATION (Audit 2025112309):
# ConditionalG2DCMA was previously in conditional_g2_dcma.py.
# Consolidated here to improve code organization and reduce file redundancy.
# ==============================================================================


@FUSION_REGISTRY.register("conditional_g2_dcma")
class ConditionalG2DCMA(BaseModule):
    """
    Conditional G²-DCMA with single-modality bypass paths.

    CRITICAL FIX (Audit 2025112101 Phase 2.1):
    Extends G²-DCMA to handle single/zero-modality scenarios by adding bypass paths:
    - Dual-modality: Use full G²-DCMA fusion
    - Single-RGB: Use RGB bypass branch
    - Single-Sonar: Use Sonar bypass branch
    - Zero-modality: Return minimal/zero features

    CONTINUOUS FLOW DESIGN (Audit 2025112307):
    In continuous flow design, ConditionalG2DCMA's role changes from "conditional skip"
    to "adaptive fusion strategy":
    - All features flow continuously through the network
    - Routing based on has_object flags (not conditional skip)
    - Handles batch-level heterogeneity (different samples, different validity)
    - Provides computational efficiency (avoid expensive fusion for single-modality)

    Args:
        embed_dim: Embedding dimension for G²-DCMA
        num_heads: Number of attention heads
        num_radial_bins: Number of radial bins for sonar geometric gating
        num_scales: Number of pyramid scales for RGB geometric gating
        num_deformable_points: Number of sampling points per head
        use_multi_scale: Whether to use multi-scale deformable attention
        num_levels: Number of pyramid levels (if multi-scale)
        dropout: Dropout probability
        consistency_config: Configuration for dual consistency loss
        rgb_channels: RGB feature channels (for bypass branch)
        sonar_channels: Sonar feature channels (for bypass branch)

    Returns:
        dict containing:
            - fused_features: Fused/bypassed features [B, embed_dim, H, W]
            - routing_info: Information about which path was taken
            - consistency_losses: Consistency losses (only for dual-modality)
            - (other G²-DCMA outputs when applicable)
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
        consistency_config: Optional[Dict] = None,
        rgb_channels: int = 384,
        sonar_channels: int = 384,
        **kwargs
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.rgb_channels = rgb_channels
        self.sonar_channels = sonar_channels

        # Core G²-DCMA for dual-modality fusion
        self.g2_dcma = G2DCMA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_radial_bins=num_radial_bins,
            num_scales=num_scales,
            num_deformable_points=num_deformable_points,
            use_multi_scale=use_multi_scale,
            num_levels=num_levels,
            dropout=dropout,
            consistency_config=consistency_config
        )

        # Single-modality bypass branches
        # These branches process single-modality features to produce features
        # compatible with downstream heads

        # RGB bypass: RGB features → embed_dim
        self.rgb_bypass = nn.Sequential(
            nn.Conv2d(rgb_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Sonar bypass: Sonar features → embed_dim
        # Note: Sonar features may have different spatial dimensions,
        # so we use adaptive pooling + upsampling if needed
        self.sonar_bypass = nn.Sequential(
            nn.Conv2d(sonar_channels, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Learnable routing confidence
        # Used to gradually transition from bypass to full fusion during training
        self.routing_confidence = nn.Parameter(torch.tensor(1.0))

        self.logger.info(
            f"Initialized ConditionalG2DCMA: embed_dim={embed_dim}, "
            f"rgb_channels={rgb_channels}, sonar_channels={sonar_channels}, "
            f"num_heads={num_heads}"
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        is_rgb_valid: Optional[torch.Tensor] = None,
        is_sonar_valid: Optional[torch.Tensor] = None,
        rgb_pyramid: Optional[List[torch.Tensor]] = None,
        return_consistency: bool = True,
        sonar_letterbox_metadata: Union[dict, List[dict], None] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Conditional fusion with modality-aware routing.

        CRITICAL FIX (Audit 2025112101 Phase 2.1):
        Routes samples to appropriate fusion paths based on modality availability:
        - Dual-modality: Full G²-DCMA fusion
        - Single-RGB: RGB bypass
        - Single-Sonar: Sonar bypass
        - Zero-modality: Minimal features (zeros)

        CRITICAL FIX (Audit 2025112403 Finding 3): Added sonar_letterbox_metadata parameter

        ENHANCED FIX (Audit 2025112403 Follow-up): Per-sample metadata support
        Now accepts and passes through list of dicts for heterogeneous batches.

        Args:
            rgb_features: RGB feature maps [B, C_rgb, H_rgb, W_rgb]
            sonar_features: Sonar feature maps [B, C_sonar, H_sonar, W_sonar]
            is_rgb_valid: [B] boolean tensor, True if RGB has valid target (optional)
            is_sonar_valid: [B] boolean tensor, True if Sonar has valid target (optional)
            rgb_pyramid: Optional multi-scale RGB features for geometric gating
            return_consistency: Whether to compute consistency losses
            sonar_letterbox_metadata: Either single dict or list of dicts for per-sample metadata

        Returns:
            dict containing:
                - fused_features: Fused/bypassed features [B, embed_dim, H_rgb, W_rgb]
                - routing_info: Dict with routing statistics
                - consistency_losses: Dict of consistency losses (only for dual-modality)
                - (other outputs from G²-DCMA when applicable)
        """
        B, C_rgb, H_rgb, W_rgb = rgb_features.shape
        B_s, C_sonar, H_sonar, W_sonar = sonar_features.shape

        assert B == B_s, f"Batch size mismatch: RGB={B}, Sonar={B_s}"

        # CRITICAL FIX (Audit 2025112502 Revision): Convert float32 validity masks to bool
        # Trainer/collate_fn produces float32 tensors (0.0/1.0), but bitwise ops require bool
        if is_rgb_valid is None:
            is_rgb_valid = torch.ones(B, dtype=torch.bool, device=rgb_features.device)
        else:
            # Convert to bool if float (from Trainer's float32 validity masks)
            if is_rgb_valid.dtype != torch.bool:
                is_rgb_valid = is_rgb_valid.bool()

        if is_sonar_valid is None:
            is_sonar_valid = torch.ones(B, dtype=torch.bool, device=sonar_features.device)
        else:
            # Convert to bool if float (from Trainer's float32 validity masks)
            if is_sonar_valid.dtype != torch.bool:
                is_sonar_valid = is_sonar_valid.bool()

        # Detect modality availability for each sample
        zero_modality_mask = (~is_rgb_valid) & (~is_sonar_valid)  # [B]
        single_rgb_mask = is_rgb_valid & (~is_sonar_valid)  # [B]
        single_sonar_mask = (~is_rgb_valid) & is_sonar_valid  # [B]
        dual_modality_mask = is_rgb_valid & is_sonar_valid  # [B]

        # Initialize output tensors
        fused_features = torch.zeros(B, self.embed_dim, H_rgb, W_rgb, device=rgb_features.device, dtype=rgb_features.dtype)
        # CRITICAL FIX (Audit 2025112503 Finding 1): Initialize dual-view output tensors
        # RGB and Sonar BoxHeads require separate view-specific features
        rgb_view_features = torch.zeros(B, self.embed_dim, H_rgb, W_rgb, device=rgb_features.device, dtype=rgb_features.dtype)
        sonar_view_features = torch.zeros(B, self.embed_dim, H_rgb, W_rgb, device=rgb_features.device, dtype=rgb_features.dtype)

        # Routing info for debugging and analysis
        routing_info = {
            'num_dual': dual_modality_mask.sum().item(),
            'num_single_rgb': single_rgb_mask.sum().item(),
            'num_single_sonar': single_sonar_mask.sum().item(),
            'num_zero': zero_modality_mask.sum().item(),
            'routing_confidence': torch.sigmoid(self.routing_confidence).item()
        }

        # Initialize consistency losses (will only be populated for dual-modality)
        consistency_losses = {
            'component_consistency': torch.tensor(0.0, device=rgb_features.device),
            'representation_consistency': torch.tensor(0.0, device=rgb_features.device)
        }
        # CRITICAL FIX (Audit 2025112503 Remaining Issue 1): Initialize view_consistency_loss
        view_consistency_loss_accum = torch.tensor(0.0, device=rgb_features.device)

        # Route 1: Dual-modality → Full G²-DCMA fusion
        if dual_modality_mask.any():
            dual_idx = dual_modality_mask.nonzero(as_tuple=False).squeeze(-1)  # [N_dual]
            rgb_dual = rgb_features[dual_idx]  # [N_dual, C_rgb, H_rgb, W_rgb]
            sonar_dual = sonar_features[dual_idx]  # [N_dual, C_sonar, H_sonar, W_sonar]

            # CRITICAL FIX (Audit 2025112403 Finding 3): Extract metadata for dual-modality samples
            # ENHANCED FIX (Audit 2025112403 Follow-up): Per-sample metadata support
            sonar_metadata_dual = None
            if sonar_letterbox_metadata is not None:
                if isinstance(sonar_letterbox_metadata, dict):
                    # Single metadata dict (same for all samples in batch)
                    sonar_metadata_dual = sonar_letterbox_metadata
                elif isinstance(sonar_letterbox_metadata, list):
                    # Extract metadata subset for dual-modality samples
                    sonar_metadata_dual = [sonar_letterbox_metadata[i] for i in dual_idx.tolist()]

            # Apply full G²-DCMA
            g2_output = self.g2_dcma.forward_single_scale(
                rgb_features=rgb_dual,
                sonar_features=sonar_dual,
                rgb_pyramid=[p[dual_idx] for p in rgb_pyramid] if rgb_pyramid else None,
                return_consistency=return_consistency,
                sonar_letterbox_metadata=sonar_metadata_dual
            )

            # Place fused features back to output tensor
            fused_features[dual_idx] = g2_output['fused_features']

            # CRITICAL FIX (Audit 2025112503 Finding 1): Propagate dual-view features
            # RGB BoxHead needs RGB-view fused features, Sonar BoxHead needs Sonar-view fused features
            rgb_view_features[dual_idx] = g2_output['rgb_view_fused']
            sonar_view_features[dual_idx] = g2_output['sonar_view_fused']

            # Accumulate consistency losses (weighted by number of dual-modality samples)
            if return_consistency and 'consistency_losses' in g2_output:
                weight = dual_idx.shape[0] / B
                # CRITICAL FIX: Use correct keys from DualConsistencyLoss
                # DualConsistencyLoss returns 'component_loss' and 'infonce_loss'
                consistency_losses['component_consistency'] = g2_output['consistency_losses'].get('component_loss', torch.tensor(0.0, device=rgb_features.device)) * weight
                consistency_losses['representation_consistency'] = g2_output['consistency_losses'].get('infonce_loss', torch.tensor(0.0, device=rgb_features.device)) * weight

            # CRITICAL FIX (Audit 2025112503 Remaining Issue 1): Propagate view_consistency_loss
            # G2DCMA returns view_consistency_loss for cross-view regularization
            if 'view_consistency_loss' in g2_output:
                weight = dual_idx.shape[0] / B
                view_consistency_loss_accum += g2_output['view_consistency_loss'] * weight

        # Route 2: Single-RGB → RGB bypass
        if single_rgb_mask.any():
            rgb_idx = single_rgb_mask.nonzero(as_tuple=False).squeeze(-1)  # [N_rgb]
            rgb_single = rgb_features[rgb_idx]  # [N_rgb, C_rgb, H_rgb, W_rgb]

            # Apply RGB bypass
            rgb_bypassed = self.rgb_bypass(rgb_single)  # [N_rgb, embed_dim, H_rgb, W_rgb]

            # Place bypassed features back to output tensor
            fused_features[rgb_idx] = rgb_bypassed

            # CRITICAL FIX (Audit 2025112503 Finding 1): Construct dual-view for single-RGB
            # RGB-view: use bypassed RGB features
            # Sonar-view: zero (no sonar information available)
            rgb_view_features[rgb_idx] = rgb_bypassed
            sonar_view_features[rgb_idx] = torch.zeros_like(rgb_bypassed)

        # Route 3: Single-Sonar → Sonar bypass
        if single_sonar_mask.any():
            sonar_idx = single_sonar_mask.nonzero(as_tuple=False).squeeze(-1)  # [N_sonar]
            sonar_single = sonar_features[sonar_idx]  # [N_sonar, C_sonar, H_sonar, W_sonar]

            # Apply Sonar bypass
            sonar_bypassed = self.sonar_bypass(sonar_single)  # [N_sonar, embed_dim, H_sonar, W_sonar]

            # Resize to RGB spatial dimensions if needed
            if sonar_bypassed.shape[2:] != (H_rgb, W_rgb):
                sonar_bypassed = F.interpolate(
                    sonar_bypassed,
                    size=(H_rgb, W_rgb),
                    mode='bilinear',
                    align_corners=False
                )  # [N_sonar, embed_dim, H_rgb, W_rgb]

            # Place bypassed features back to output tensor
            fused_features[sonar_idx] = sonar_bypassed

            # CRITICAL FIX (Audit 2025112503 Finding 1): Construct dual-view for single-Sonar
            # RGB-view: zero (no RGB information available)
            # Sonar-view: use bypassed Sonar features (already resized to RGB dimensions)
            rgb_view_features[sonar_idx] = torch.zeros_like(sonar_bypassed)
            sonar_view_features[sonar_idx] = sonar_bypassed

        # Route 4: Zero-modality → Minimal features (already initialized to zeros)
        # No additional processing needed for zero-modality samples

        # Log routing distribution if in debug mode
        if self.logger.isEnabledFor(10):  # DEBUG level
            self.logger.debug(
                f"ConditionalG2DCMA routing: "
                f"dual={routing_info['num_dual']}, "
                f"RGB-only={routing_info['num_single_rgb']}, "
                f"Sonar-only={routing_info['num_single_sonar']}, "
                f"zero={routing_info['num_zero']}"
            )

        output = {
            'fused_features': fused_features,
            # CRITICAL FIX (Audit 2025112503 Finding 1): Add dual-view outputs
            # These are essential for RGB and Sonar BoxHeads to use their respective view features
            'rgb_view_fused': rgb_view_features,    # For RGB BoxHead
            'sonar_view_fused': sonar_view_features,  # For Sonar BoxHead
            'routing_info': routing_info,
            'consistency_losses': consistency_losses if return_consistency else None,
            # CRITICAL FIX (Audit 2025112503 Remaining Issue 1): Propagate view_consistency_loss
            # This enables training.loss_weights.view_consistency to take effect
            'view_consistency_loss': view_consistency_loss_accum
        }

        return output


def build_conditional_g2_dcma(
    rgb_channels: int = 384,
    sonar_channels: int = 384,
    embed_dim: int = 256,
    **kwargs
) -> ConditionalG2DCMA:
    """
    Factory function to build conditional G²-DCMA.

    Args:
        rgb_channels: RGB feature dimension
        sonar_channels: Sonar feature dimension
        embed_dim: Embedding dimension
        **kwargs: Additional arguments for ConditionalG2DCMA

    Returns:
        ConditionalG2DCMA instance

    Example:
        >>> fusion = build_conditional_g2_dcma(
        ...     rgb_channels=384,
        ...     sonar_channels=384,
        ...     embed_dim=256
        ... )
    """
    return ConditionalG2DCMA(
        rgb_channels=rgb_channels,
        sonar_channels=sonar_channels,
        embed_dim=embed_dim,
        **kwargs
    )
