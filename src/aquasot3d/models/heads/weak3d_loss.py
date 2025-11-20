"""
3D Coordinate Consistency Loss for Weak3DHead_v2

CRITICAL FIX (Audit 2025112101 Phase 3.3):
This module implements consistency constraints for training single-modality
bypass branches in Weak3DHead_v2.

Training Strategy:
1. In dual-modality scenarios, supervise both primary and bypass branches
2. Enforce cross-modal consistency between RGB and Sonar estimates
3. Use curriculum learning to gradually increase consistency weight

Key Losses:
- Depth Consistency: RGB depth estimator vs Sonar radial depth
- Vertical Consistency: RGB Y vs Sonar Y-branch
- Cross-modal X Consistency: RGB X vs Sonar X

Author: AquaSOT-3D Team
Date: 2025-11-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from aquasot3d.core.base_module import BaseModule


class Weak3DConsistencyLoss(BaseModule):
    """
    3D Coordinate Consistency Loss for Weak3DHead_v2.

    CRITICAL FIX (Audit 2025112101 Phase 3.3):
    Supervises single-modality bypass branches by enforcing consistency
    with primary branches in dual-modality scenarios.

    Training Rationale:
    - RGB depth estimator needs supervision from Sonar (ground truth depth)
    - Sonar Y-branch needs supervision from RGB (ground truth Y)
    - Cross-modal consistency ensures unified 3D representation

    Args:
        depth_consistency_weight: Weight for depth consistency loss
        y_consistency_weight: Weight for Y-coordinate consistency loss
        x_consistency_weight: Weight for X-coordinate consistency loss
        loss_type: Type of distance metric ('l1', 'l2', 'smooth_l1')
        use_stop_gradient: Whether to stop gradient for primary branch
        curriculum_warmup_epochs: Epochs for curriculum warmup (0 = no warmup)

    Loss Components:
        1. Depth Consistency (Z):
           L_depth = |Z_rgb_depth_est - Z_sonar_radial|
           Trains RGB depth estimator using Sonar depth as pseudo-GT

        2. Vertical Consistency (Y):
           L_y = |Y_rgb - Y_sonar_branch|
           Trains Sonar Y-branch using RGB Y as pseudo-GT

        3. Horizontal Consistency (X):
           L_x = |X_rgb - X_sonar|
           Optional: Both RGB and Sonar are good at X
    """

    def __init__(
        self,
        depth_consistency_weight: float = 1.0,
        y_consistency_weight: float = 1.0,
        x_consistency_weight: float = 0.1,
        loss_type: str = 'smooth_l1',
        use_stop_gradient: bool = True,
        curriculum_warmup_epochs: int = 0,
        **kwargs
    ):
        super().__init__()

        self.depth_consistency_weight = depth_consistency_weight
        self.y_consistency_weight = y_consistency_weight
        self.x_consistency_weight = x_consistency_weight
        self.loss_type = loss_type
        self.use_stop_gradient = use_stop_gradient
        self.curriculum_warmup_epochs = curriculum_warmup_epochs

        # Current training epoch (for curriculum)
        self.current_epoch = 0

        # Loss function selection
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction='none', beta=1.0)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum learning."""
        self.current_epoch = epoch

    def get_curriculum_weight(self) -> float:
        """
        Get curriculum learning weight based on current epoch.

        Returns:
            weight: Scaling factor in [0, 1] based on warmup progress
        """
        if self.curriculum_warmup_epochs == 0:
            return 1.0

        if self.current_epoch >= self.curriculum_warmup_epochs:
            return 1.0
        else:
            # Linear warmup
            return self.current_epoch / self.curriculum_warmup_epochs

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        is_rgb_valid: torch.Tensor,
        is_sonar_valid: torch.Tensor,
        weak3d_outputs: Dict[str, torch.Tensor],
        intermediate_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute 3D consistency losses.

        CRITICAL: This loss is only applied to dual-modality samples.
        Single-modality samples use the bypass branches during inference
        but are trained on dual-modality data.

        Args:
            rgb_features: RGB features [B, C_rgb]
            sonar_features: Sonar features [B, C_sonar]
            is_rgb_valid: [B] boolean tensor
            is_sonar_valid: [B] boolean tensor
            weak3d_outputs: Outputs from Weak3DHead_v2.forward()
                - Must contain 'X', 'Y', 'Z' predictions
            intermediate_outputs: Optional intermediate predictions from bypass branches
                - 'X_rgb_bypass': X from RGB-only head
                - 'Y_rgb_bypass': Y from RGB-only head
                - 'Z_rgb_bypass': Z from RGB depth estimator
                - 'X_sonar_bypass': X from Sonar-only head
                - 'Y_sonar_bypass': Y from Sonar Y-branch
                - 'Z_sonar_bypass': Z from Sonar radial

        Returns:
            dict containing:
                - total_loss: Weighted sum of all consistency losses
                - depth_consistency: Depth consistency loss
                - y_consistency: Y-coordinate consistency loss
                - x_consistency: X-coordinate consistency loss
        """
        B = rgb_features.shape[0]
        device = rgb_features.device

        # Identify dual-modality samples
        dual_modality_mask = is_rgb_valid & is_sonar_valid  # [B]

        if not dual_modality_mask.any():
            # No dual-modality samples, return zero losses
            return {
                'total_loss': torch.tensor(0.0, device=device),
                'depth_consistency': torch.tensor(0.0, device=device),
                'y_consistency': torch.tensor(0.0, device=device),
                'x_consistency': torch.tensor(0.0, device=device)
            }

        # Get curriculum weight
        curriculum_weight = self.get_curriculum_weight()

        # Extract primary predictions (from dual-modality routing)
        # These come from the final weak3d_outputs
        X_primary = weak3d_outputs['X']  # [B, 1]
        Y_primary = weak3d_outputs['Y']  # [B, 1]
        Z_primary = weak3d_outputs['Z']  # [B, 1]

        # Initialize losses
        total_loss = torch.tensor(0.0, device=device)
        depth_consistency_loss = torch.tensor(0.0, device=device)
        y_consistency_loss = torch.tensor(0.0, device=device)
        x_consistency_loss = torch.tensor(0.0, device=device)

        # If intermediate outputs not provided, we cannot compute consistency
        if intermediate_outputs is None:
            return {
                'total_loss': total_loss,
                'depth_consistency': depth_consistency_loss,
                'y_consistency': y_consistency_loss,
                'x_consistency': x_consistency_loss
            }

        # ========================================
        # Loss 1: Depth Consistency (Z)
        # ========================================
        # Train RGB depth estimator using Sonar depth as pseudo-GT
        if 'Z_rgb_bypass' in intermediate_outputs and 'Z_sonar_bypass' in intermediate_outputs:
            Z_rgb_est = intermediate_outputs['Z_rgb_bypass']  # [B, 1] from RGB depth estimator
            Z_sonar_gt = intermediate_outputs['Z_sonar_bypass']  # [B, 1] from Sonar radial (reliable)

            if self.use_stop_gradient:
                # Stop gradient for Sonar (teacher), only train RGB estimator
                Z_sonar_gt = Z_sonar_gt.detach()

            # Compute depth consistency for dual-modality samples
            depth_diff = self.loss_fn(Z_rgb_est, Z_sonar_gt)  # [B, 1]
            depth_diff_masked = depth_diff[dual_modality_mask]  # [N_dual, 1]

            if depth_diff_masked.numel() > 0:
                depth_consistency_loss = depth_diff_masked.mean()
                total_loss = total_loss + self.depth_consistency_weight * curriculum_weight * depth_consistency_loss

        # ========================================
        # Loss 2: Vertical Consistency (Y)
        # ========================================
        # Train Sonar Y-branch using RGB Y as pseudo-GT
        if 'Y_rgb_bypass' in intermediate_outputs and 'Y_sonar_bypass' in intermediate_outputs:
            Y_rgb_gt = intermediate_outputs['Y_rgb_bypass']  # [B, 1] from RGB (reliable)
            Y_sonar_est = intermediate_outputs['Y_sonar_bypass']  # [B, 1] from Sonar Y-branch

            if self.use_stop_gradient:
                # Stop gradient for RGB (teacher), only train Sonar Y-branch
                Y_rgb_gt = Y_rgb_gt.detach()

            # Compute Y consistency for dual-modality samples
            y_diff = self.loss_fn(Y_sonar_est, Y_rgb_gt)  # [B, 1]
            y_diff_masked = y_diff[dual_modality_mask]  # [N_dual, 1]

            if y_diff_masked.numel() > 0:
                y_consistency_loss = y_diff_masked.mean()
                total_loss = total_loss + self.y_consistency_weight * curriculum_weight * y_consistency_loss

        # ========================================
        # Loss 3: Horizontal Consistency (X)
        # ========================================
        # Optional: Both RGB and Sonar are good at X, so this is a weaker constraint
        if 'X_rgb_bypass' in intermediate_outputs and 'X_sonar_bypass' in intermediate_outputs:
            X_rgb = intermediate_outputs['X_rgb_bypass']  # [B, 1]
            X_sonar = intermediate_outputs['X_sonar_bypass']  # [B, 1]

            # Mutual consistency (no clear teacher/student)
            x_diff = self.loss_fn(X_rgb, X_sonar)  # [B, 1]
            x_diff_masked = x_diff[dual_modality_mask]  # [N_dual, 1]

            if x_diff_masked.numel() > 0:
                x_consistency_loss = x_diff_masked.mean()
                total_loss = total_loss + self.x_consistency_weight * curriculum_weight * x_consistency_loss

        return {
            'total_loss': total_loss,
            'depth_consistency': depth_consistency_loss,
            'y_consistency': y_consistency_loss,
            'x_consistency': x_consistency_loss
        }


def build_weak3d_consistency_loss(
    depth_consistency_weight: float = 1.0,
    y_consistency_weight: float = 1.0,
    x_consistency_weight: float = 0.1,
    **kwargs
) -> Weak3DConsistencyLoss:
    """
    Factory function to build Weak3DConsistencyLoss.

    Args:
        depth_consistency_weight: Weight for depth consistency
        y_consistency_weight: Weight for Y consistency
        x_consistency_weight: Weight for X consistency
        **kwargs: Additional arguments

    Returns:
        Weak3DConsistencyLoss instance

    Example:
        >>> consistency_loss = build_weak3d_consistency_loss(
        ...     depth_consistency_weight=1.0,
        ...     y_consistency_weight=1.0,
        ...     curriculum_warmup_epochs=50
        ... )
    """
    return Weak3DConsistencyLoss(
        depth_consistency_weight=depth_consistency_weight,
        y_consistency_weight=y_consistency_weight,
        x_consistency_weight=x_consistency_weight,
        **kwargs
    )
