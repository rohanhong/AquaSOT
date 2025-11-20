"""
Conditional G²-DCMA: Modality-Aware Fusion with Single-Modality Bypass

CRITICAL FIX (Audit 2025112101 Phase 2.1):
This module extends G²-DCMA with conditional fusion paths to handle
single-modality and zero-modality scenarios gracefully.

Key improvements:
1. Single-modality bypass: When only one modality is valid, directly use that modality's features
2. Modality-aware routing: Dynamically select fusion path based on modality availability
3. Zero-modality handling: Return minimal features when both modalities are invalid

Architecture:
    is_rgb_valid, is_sonar_valid
           ↓
    [Modality Router]
           ↓
    ┌──────────┴──────────┐
    │  Dual-Modality      │  Single-Modality    │  Zero-Modality
    │  (G²-DCMA)          │  (Bypass)           │  (Minimal)
    │  - Geometric Gating │  - RGB Branch       │  - Zero Features
    │  - Deformable Attn  │  - Sonar Branch     │
    │  - Consistency Loss │                     │
    └────────┬────────────┴─────────────────────┘
             ↓
    Fused features + Routing info

Author: AquaSOT-3D Team
Date: 2025-11-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

from aquasot3d.core.base_module import BaseModule
from aquasot3d.core.registry import FUSION_REGISTRY
from aquasot3d.models.fusion.g2_dcma import G2DCMA


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
        return_consistency: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Conditional fusion with modality-aware routing.

        CRITICAL FIX (Audit 2025112101 Phase 2.1):
        Routes samples to appropriate fusion paths based on modality availability:
        - Dual-modality: Full G²-DCMA fusion
        - Single-RGB: RGB bypass
        - Single-Sonar: Sonar bypass
        - Zero-modality: Minimal features (zeros)

        Args:
            rgb_features: RGB feature maps [B, C_rgb, H_rgb, W_rgb]
            sonar_features: Sonar feature maps [B, C_sonar, H_sonar, W_sonar]
            is_rgb_valid: [B] boolean tensor, True if RGB has valid target (optional)
            is_sonar_valid: [B] boolean tensor, True if Sonar has valid target (optional)
            rgb_pyramid: Optional multi-scale RGB features for geometric gating
            return_consistency: Whether to compute consistency losses

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

        # If no validity information provided, assume all samples are dual-modality
        if is_rgb_valid is None:
            is_rgb_valid = torch.ones(B, dtype=torch.bool, device=rgb_features.device)
        if is_sonar_valid is None:
            is_sonar_valid = torch.ones(B, dtype=torch.bool, device=sonar_features.device)

        # Detect modality availability for each sample
        zero_modality_mask = (~is_rgb_valid) & (~is_sonar_valid)  # [B]
        single_rgb_mask = is_rgb_valid & (~is_sonar_valid)  # [B]
        single_sonar_mask = (~is_rgb_valid) & is_sonar_valid  # [B]
        dual_modality_mask = is_rgb_valid & is_sonar_valid  # [B]

        # Initialize output tensors
        fused_features = torch.zeros(B, self.embed_dim, H_rgb, W_rgb, device=rgb_features.device, dtype=rgb_features.dtype)

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

        # Route 1: Dual-modality → Full G²-DCMA fusion
        if dual_modality_mask.any():
            dual_idx = dual_modality_mask.nonzero(as_tuple=False).squeeze(-1)  # [N_dual]
            rgb_dual = rgb_features[dual_idx]  # [N_dual, C_rgb, H_rgb, W_rgb]
            sonar_dual = sonar_features[dual_idx]  # [N_dual, C_sonar, H_sonar, W_sonar]

            # Apply full G²-DCMA
            g2_output = self.g2_dcma.forward_single_scale(
                rgb_features=rgb_dual,
                sonar_features=sonar_dual,
                rgb_pyramid=[p[dual_idx] for p in rgb_pyramid] if rgb_pyramid else None,
                return_consistency=return_consistency
            )

            # Place fused features back to output tensor
            fused_features[dual_idx] = g2_output['fused_features']

            # Accumulate consistency losses (weighted by number of dual-modality samples)
            if return_consistency and 'consistency_losses' in g2_output:
                weight = dual_idx.shape[0] / B
                # CRITICAL FIX: Use correct keys from DualConsistencyLoss
                # DualConsistencyLoss returns 'component_loss' and 'infonce_loss'
                consistency_losses['component_consistency'] = g2_output['consistency_losses'].get('component_loss', torch.tensor(0.0, device=rgb_features.device)) * weight
                consistency_losses['representation_consistency'] = g2_output['consistency_losses'].get('infonce_loss', torch.tensor(0.0, device=rgb_features.device)) * weight

        # Route 2: Single-RGB → RGB bypass
        if single_rgb_mask.any():
            rgb_idx = single_rgb_mask.nonzero(as_tuple=False).squeeze(-1)  # [N_rgb]
            rgb_single = rgb_features[rgb_idx]  # [N_rgb, C_rgb, H_rgb, W_rgb]

            # Apply RGB bypass
            rgb_bypassed = self.rgb_bypass(rgb_single)  # [N_rgb, embed_dim, H_rgb, W_rgb]

            # Place bypassed features back to output tensor
            fused_features[rgb_idx] = rgb_bypassed

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
            'routing_info': routing_info,
            'consistency_losses': consistency_losses if return_consistency else None
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
