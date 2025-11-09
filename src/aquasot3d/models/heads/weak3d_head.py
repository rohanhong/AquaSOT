"""
Weak 3D Center Head for Calibration-Free 3D Estimation

This module implements a weak 3D center estimation head that predicts (X, Y, Z)
coordinates without requiring extrinsic calibration between RGB and Sonar modalities.

Architecture:
- Z (Depth): Estimated from sonar radial activation via learnable weighted integral
- X (Horizontal): Dual-branch regression (RGB + Sonar) with consistency loss
- Y (Vertical): RGB-dominant regression (better vertical resolution)
- Coordinate System: Fusion Frame (X→right, Y→down, Z→forward)

Key Innovation: Uses sonar radial geometry as depth proxy and enforces cross-modal
consistency without requiring calibration parameters.

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from aquasot3d.core import BaseHead, PREDICTION_HEAD_REGISTRY


def extract_radial_activation(
    sonar_img: torch.Tensor,
    sonar_feat_map: torch.Tensor,
    num_bins: int = 50
) -> torch.Tensor:
    """
    Extract radial activation profile from sonar feature map.

    This function computes a 1D radial profile from 2D sonar features by
    averaging activations at similar distances from the sonar sensor origin.
    The profile serves as a depth proxy for weak 3D estimation.

    Args:
        sonar_img: Original sonar image, shape (B, 1, H_orig, W_orig)
                   e.g., (B, 1, 376, 682) for standard FLS
        sonar_feat_map: Sonar feature map, shape (B, C, H', W')
                        Typically downsampled from original
        num_bins: Number of radial bins (default: 50)

    Returns:
        Radial activation profile, shape (B, num_bins)

    Example:
        >>> sonar_img = torch.randn(4, 1, 376, 682)
        >>> sonar_feat = torch.randn(4, 256, 14, 14)
        >>> profile = extract_radial_activation(sonar_img, sonar_feat, num_bins=50)
        >>> print(profile.shape)  # (4, 50)
    """
    B, C, H, W = sonar_feat_map.shape

    # Calculate downsampling scale
    H_orig, W_orig = sonar_img.shape[-2:]
    scale_h = H / H_orig
    scale_w = W / W_orig

    # Sensor position in feature map coordinates (scaled from original)
    # Standard FLS: sensor at bottom center (cx=W_orig//2, cy=H_orig-1)
    cx = (W_orig // 2) * scale_w
    cy = (H_orig - 1) * scale_h

    # Create radial distance map in feature space
    y_coords = torch.arange(H, device=sonar_img.device, dtype=torch.float32)
    x_coords = torch.arange(W, device=sonar_img.device, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Compute radial distance from sensor origin
    r = torch.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)  # (H, W)

    # Normalize radial distance to bin indices [0, num_bins)
    r_max = r.max()
    r_normalized = (r / (r_max + 1e-7)) * (num_bins - 1)
    r_bins = r_normalized.long().clamp(0, num_bins - 1)  # (H, W)

    # Compute average feature activation per radial bin
    radial_profile = []
    for b in range(B):
        profile = []
        for bin_idx in range(num_bins):
            mask = (r_bins == bin_idx)  # (H, W)
            if mask.sum() > 0:
                # Mean activation across all channels in this bin
                bin_activation = sonar_feat_map[b, :, mask].mean()
            else:
                # Empty bin (shouldn't happen with proper discretization)
                bin_activation = torch.tensor(0.0, device=sonar_img.device)
            profile.append(bin_activation)
        radial_profile.append(torch.stack(profile))

    return torch.stack(radial_profile)  # (B, num_bins)


@PREDICTION_HEAD_REGISTRY.register("Weak3DHead")
class Weak3DHead(BaseHead):
    """
    Weak 3D Center Estimation Head.

    Predicts 3D center (X, Y, Z) in Fusion Frame without extrinsic calibration.
    Uses sonar radial geometry for depth and dual-branch consistency for horizontal.

    Coordinate System (Fusion Frame):
        - X: Horizontal (right positive)
        - Y: Vertical (down positive)
        - Z: Depth/Forward (forward positive)

    Args:
        rgb_feat_dim: RGB feature dimension
        sonar_feat_dim: Sonar feature dimension
        num_radial_bins: Number of radial bins for depth estimation (default: 50)
        hidden_dim: Hidden layer dimension (default: 256)
        use_residual_z: Use residual connection for Z estimation (default: True)
        consistency_weight: Weight for X consistency loss (default: 0.2)

    Returns:
        Dictionary containing:
            - 'center_3d': 3D center coordinates (X, Y, Z), shape (B, 3)
            - 'X_rgb': X from RGB branch, shape (B, 1)
            - 'X_sonar': X from Sonar branch, shape (B, 1)
            - 'X_weights': Fusion weights for X, shape (B, 2)
            - 'Y': Y coordinate, shape (B, 1)
            - 'Z': Z coordinate, shape (B, 1)
            - 'Z_weighted': Weighted integral component, shape (B, 1)
            - 'consistency_loss': X consistency loss (if enabled)

    Example:
        >>> head = Weak3DHead(rgb_feat_dim=768, sonar_feat_dim=384)
        >>> rgb_feat = torch.randn(4, 768)
        >>> sonar_feat = torch.randn(4, 384)
        >>> sonar_radial = torch.randn(4, 50)
        >>> outputs = head(rgb_feat, sonar_feat, sonar_radial)
        >>> print(outputs['center_3d'].shape)  # (4, 3)
    """

    def __init__(
        self,
        rgb_feat_dim: int,
        sonar_feat_dim: int,
        num_radial_bins: int = 50,
        hidden_dim: int = 256,
        use_residual_z: bool = True,
        consistency_weight: float = 0.2,
        **kwargs
    ):
        super().__init__()

        self.rgb_feat_dim = rgb_feat_dim
        self.sonar_feat_dim = sonar_feat_dim
        self.num_radial_bins = num_radial_bins
        self.hidden_dim = hidden_dim
        self.use_residual_z = use_residual_z
        self.consistency_weight = consistency_weight

        # Z estimation from sonar radial activation profile
        # Learnable weights for weighted integral
        self.radial_weights = nn.Parameter(torch.randn(num_radial_bins))

        # MLP for refining depth from radial profile
        self.radial_encoder = nn.Sequential(
            nn.Linear(num_radial_bins, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

        # X regression: Dual-branch (RGB + Sonar) with consistency
        self.rgb_x_head = nn.Sequential(
            nn.Linear(rgb_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.sonar_x_head = nn.Sequential(
            nn.Linear(sonar_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        # Adaptive fusion gate for X coordinate
        self.x_fusion_gate = nn.Sequential(
            nn.Linear(rgb_feat_dim + sonar_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

        # Y regression: RGB-dominant (better vertical resolution)
        self.y_head = nn.Sequential(
            nn.Linear(rgb_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

        self.logger.info(
            f"Initialized Weak3DHead: rgb_feat_dim={rgb_feat_dim}, "
            f"sonar_feat_dim={sonar_feat_dim}, num_radial_bins={num_radial_bins}, "
            f"hidden_dim={hidden_dim}, use_residual_z={use_residual_z}"
        )

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        sonar_radial_activation: torch.Tensor,
        compute_consistency_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for weak 3D center estimation.

        Args:
            rgb_features: RGB features, shape (B, C_rgb) or (B, C_rgb, H, W)
            sonar_features: Sonar features, shape (B, C_sonar) or (B, C_sonar, H, W)
            sonar_radial_activation: Radial activation profile, shape (B, num_bins)
            compute_consistency_loss: Whether to compute X consistency loss

        Returns:
            Dictionary with 3D predictions and auxiliary outputs
        """
        # Handle spatial features
        if rgb_features.dim() == 4:
            rgb_features = F.adaptive_avg_pool2d(rgb_features, (1, 1)).flatten(1)
        elif rgb_features.dim() == 3:
            rgb_features = rgb_features.mean(dim=1)

        if sonar_features.dim() == 4:
            sonar_features = F.adaptive_avg_pool2d(sonar_features, (1, 1)).flatten(1)
        elif sonar_features.dim() == 3:
            sonar_features = sonar_features.mean(dim=1)

        batch_size = rgb_features.shape[0]

        # ============================================
        # Z Estimation from Sonar Radial Activation
        # ============================================
        # Weighted integral: learnable attention over radial bins
        weights = F.softmax(self.radial_weights, dim=0)  # (num_bins,)
        Z_weighted = torch.sum(
            sonar_radial_activation * weights.unsqueeze(0),
            dim=-1,
            keepdim=True
        )  # (B, 1)

        # MLP refinement
        Z_mlp = self.radial_encoder(sonar_radial_activation)  # (B, 1)

        # Combine with residual connection
        if self.use_residual_z:
            Z = Z_mlp + Z_weighted
        else:
            Z = Z_mlp

        # ============================================
        # X Estimation: Dual-Branch with Consistency
        # ============================================
        X_rgb = self.rgb_x_head(rgb_features)  # (B, 1)
        X_sonar = self.sonar_x_head(sonar_features)  # (B, 1)

        # Adaptive fusion based on both modalities
        fusion_features = torch.cat([rgb_features, sonar_features], dim=-1)
        fusion_weights = self.x_fusion_gate(fusion_features)  # (B, 2)

        X_fused = (
            fusion_weights[:, 0:1] * X_rgb +
            fusion_weights[:, 1:2] * X_sonar
        )  # (B, 1)

        # Consistency loss: Encourage agreement between RGB and Sonar X predictions
        if compute_consistency_loss:
            consistency_loss = F.l1_loss(X_rgb, X_sonar) * self.consistency_weight
        else:
            consistency_loss = torch.tensor(0.0, device=rgb_features.device)

        # ============================================
        # Y Estimation: RGB-Dominant
        # ============================================
        Y = self.y_head(rgb_features)  # (B, 1)

        # ============================================
        # Assemble 3D Center
        # ============================================
        center_3d = torch.cat([X_fused, Y, Z], dim=-1)  # (B, 3)

        outputs = {
            'center_3d': center_3d,
            'X_rgb': X_rgb,
            'X_sonar': X_sonar,
            'X_fused': X_fused,
            'X_weights': fusion_weights,
            'Y': Y,
            'Z': Z,
            'Z_weighted': Z_weighted,
        }

        if compute_consistency_loss:
            outputs['consistency_loss'] = consistency_loss

        return outputs

    def compute_3d_loss(
        self,
        pred_center: torch.Tensor,
        target_center: torch.Tensor,
        loss_type: str = 'l1'
    ) -> torch.Tensor:
        """
        Compute 3D center prediction loss.

        Args:
            pred_center: Predicted 3D center, shape (B, 3)
            target_center: Target 3D center, shape (B, 3)
            loss_type: Loss type ('l1', 'l2', or 'smooth_l1')

        Returns:
            3D center loss
        """
        if loss_type == 'l1':
            loss = F.l1_loss(pred_center, target_center)
        elif loss_type == 'l2':
            loss = F.mse_loss(pred_center, target_center)
        elif loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred_center, target_center)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return loss


def build_weak3d_head(
    rgb_feat_dim: int = 768,
    sonar_feat_dim: int = 384,
    **kwargs
) -> Weak3DHead:
    """
    Factory function to build Weak 3D head.

    Args:
        rgb_feat_dim: RGB feature dimension
        sonar_feat_dim: Sonar feature dimension
        **kwargs: Additional arguments for Weak3DHead

    Returns:
        Weak3DHead instance

    Example:
        >>> head = build_weak3d_head(rgb_feat_dim=768, sonar_feat_dim=384, num_radial_bins=50)
    """
    return Weak3DHead(rgb_feat_dim=rgb_feat_dim, sonar_feat_dim=sonar_feat_dim, **kwargs)
