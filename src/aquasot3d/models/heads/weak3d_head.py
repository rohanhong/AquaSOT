"""
Weak3DHead: Angle-Based 3D Center Estimation with Single-Modality Awareness

CRITICAL FIX (Audit 2025112101 Phase 3.1):
This module extends Weak3DHead with single-modality bypass branches to handle
scenarios where only RGB or only Sonar is available.

CRITICAL FIX (Audit 2025112102):
Implements physically-grounded angle-based prediction:
- **RGB measures**: viewing angles (theta_x, theta_y) from pixel positions
- **Sonar measures**: horizontal angle (theta_x) + radial distance (converted to Z)
- **Coordinates derived**: X = Z * tan(theta_x), Y = Z * tan(theta_y)

This design enforces the physical constraints of the sensors:
1. Cameras measure directions (angles), not absolute 3D positions
2. Sonar measures angle-distance pairs in polar coordinates
3. Conversion to Cartesian coordinates uses perspective geometry

CRITICAL FIX (Audit 2025112103):
Hardware-specific FOV parameters and coordinate units:
- **RGB Camera (HD Zoom SeaCam)**: 47deg horizontal x 27deg vertical FOV
- **Sonar (Oculus MD-Series)**: 130deg horizontal FOV
- **Coordinate Units**: All outputs in pixels (not physical units like meters)

Key Features:
1. **Angle-based prediction**: Predicts viewing angles (theta_x, theta_y) in [-max_angle, max_angle]
2. **Physical correctness**: Respects sensor measurement capabilities
3. **Numerical stability**: Bounded angles prevent extreme coordinate values
4. **Single-RGB bypass**: RGB-based depth estimator when sonar unavailable
5. **Single-Sonar bypass**: Sonar-based angle estimators when RGB unavailable
6. **Modality-aware routing**: Dynamically selects estimation path
7. **Zero-modality handling**: Default predictions when both modalities invalid

Architecture:
    is_rgb_valid, is_sonar_valid
           ↓
    [Modality Router]
           ↓
    ┌──────────┴──────────┬─────────────────┬─────────────────┐
    │  Dual-Modality      │  Single-RGB     │  Single-Sonar   │  Zero-Modality
    │                     │                 │                 │
    │  - Z from Sonar     │  - Z from RGB   │  - Z from Sonar │  - Z = default
    │  - θx: RGB+Sonar    │  - θx from RGB  │  - θx from Sonar│  - X = 0
    │  - θy from RGB      │  - θy from RGB  │  - θy from Sonar│  - Y = 0
    │                     │                 │                 │
    │  Convert:           │  Convert:       │  Convert:       │
    │  - X=Z*tan(θx)      │  - X=Z*tan(θx)  │  - X=Z*tan(θx)  │
    │  - Y=Z*tan(θy)      │  - Y=Z*tan(θy)  │  - Y=Z*tan(θy)  │
    └────────┬────────────┴─────────────────┴─────────────────┘
             ↓
    3D center (X, Y, Z) + Routing info

Mathematical Foundation:
    For a point at 3D position (X, Y, Z) relative to sensor:
        theta_x = arctan(X / Z)  # Horizontal viewing angle
        theta_y = arctan(Y / Z)  # Vertical viewing angle

    Inverse (what we implement):
        X = Z * tan(theta_x)
        Y = Z * tan(theta_y)

Author: AquaSOT-3D Team
Date: 2025-11-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from aquasot3d.core import BaseHead, PREDICTION_HEAD_REGISTRY


def extract_radial_activation(
    sonar_features: torch.Tensor,
    num_radial_bins: int = 50,
    radial_range: Tuple[float, float] = (50.0, 500.0)
) -> torch.Tensor:
    """
    Extract radial activation from sonar feature maps (already in rectangular coordinates).

    CRITICAL: This function expects preprocessed sonar data in rectangular coordinates,
    NOT raw polar data. The radial activation represents how strongly the target appears
    at different depth bins.

    Args:
        sonar_features: [B, C, H, W] sonar feature maps in rectangular coordinates
        num_radial_bins: Number of radial bins for depth discretization
        radial_range: (min_range, max_range) in pixels

    Returns:
        radial_activation: [B, num_radial_bins] activation per depth bin
    """
    B, C, H, W = sonar_features.shape

    # Global average pooling across spatial dimensions to get channel activations
    # This aggregates spatial information into channel-wise responses
    channel_activation = F.adaptive_avg_pool2d(sonar_features, (1, 1))  # [B, C, 1, 1]
    channel_activation = channel_activation.squeeze(-1).squeeze(-1)  # [B, C]

    # Project channel activations to radial bins
    # Each radial bin corresponds to a depth range
    radial_projection = nn.Linear(C, num_radial_bins).to(sonar_features.device)
    radial_activation = radial_projection(channel_activation)  # [B, num_radial_bins]

    # Apply softmax to get probability distribution over depth bins
    radial_activation = F.softmax(radial_activation, dim=-1)

    return radial_activation


@PREDICTION_HEAD_REGISTRY.register("Weak3DHead")
class Weak3DHead(BaseHead):
    """
    Weak 3D Head with Single-Modality Awareness (Version 2).

    CRITICAL FIX (Audit 2025112101 Phase 3.1):
    Extends Weak3DHead to handle single/zero-modality scenarios by adding bypass paths:
    - Dual-modality: Use original logic (Z from sonar, X from both, Y from RGB)
    - Single-RGB: Use RGB depth estimator for Z, RGB for X and Y
    - Single-Sonar: Use sonar for Z, sonar for X, sonar Y-branch for Y
    - Zero-modality: Return default predictions (center of image)

    Args:
        rgb_feat_dim: RGB feature dimension
        sonar_feat_dim: Sonar feature dimension
        num_radial_bins: Number of radial bins for sonar depth discretization
        radial_range: (min_range, max_range) in pixels
        hidden_dim: Hidden layer dimension for MLP heads
        dropout: Dropout probability
        default_depth: Default depth for zero-modality scenarios (pixels)
        max_angle_rgb_horizontal: Maximum horizontal viewing angle for RGB (radians)
                                   Default: 0.410 rad (23.5deg, half of 47deg FOV)
        max_angle_rgb_vertical: Maximum vertical viewing angle for RGB (radians)
                                Default: 0.236 rad (13.5deg, half of 27deg FOV)
        max_angle_sonar_horizontal: Maximum horizontal viewing angle for Sonar (radians)
                                    Default: 1.134 rad (65deg, half of 130deg FOV)

    Returns:
        dict containing:
            - center_3d: Predicted 3D center [B, 3] in (X, Y, Z) format (pixels)
            - routing_info: Information about which path was taken
            - (intermediate outputs for debugging)
    """

    def __init__(
        self,
        rgb_feat_dim: int = 256,
        sonar_feat_dim: int = 256,
        num_radial_bins: int = 50,
        radial_range: Tuple[float, float] = (50.0, 500.0),  # Depth range in pixels
        hidden_dim: int = 512,
        dropout: float = 0.1,
        default_depth: float = 200.0,  # Default depth in pixels
        max_angle_rgb_horizontal: float = 0.41015,  # 23.5 degrees (half of 47deg FOV)
        max_angle_rgb_vertical: float = 0.23562,    # 13.5 degrees (half of 27deg FOV)
        max_angle_sonar_horizontal: float = 1.13446,  # 65 degrees (half of 130deg FOV)
        **kwargs
    ):
        super().__init__()

        self.rgb_feat_dim = rgb_feat_dim
        self.sonar_feat_dim = sonar_feat_dim
        self.num_radial_bins = num_radial_bins
        self.radial_range = radial_range
        self.default_depth = default_depth

        # Hardware-specific FOV parameters (Audit 2025112103)
        self.max_angle_rgb_horizontal = max_angle_rgb_horizontal
        self.max_angle_rgb_vertical = max_angle_rgb_vertical
        self.max_angle_sonar_horizontal = max_angle_sonar_horizontal

        # ========================================
        # Dual-Modality Branches (Original Logic)
        # ========================================

        # Z estimation from sonar radial activation
        self.radial_weights = nn.Parameter(torch.randn(num_radial_bins))
        self.radial_encoder = nn.Sequential(
            nn.Linear(num_radial_bins, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Horizontal viewing angle (theta_x) estimation: Dual-branch with adaptive fusion
        # CRITICAL (Audit 2025112102): Predicts horizontal viewing angle theta_x (radians)
        # Output range: [-1, 1] (via Tanh), scaled to [-max_angle, max_angle]
        self.rgb_theta_x_head = nn.Sequential(
            nn.Linear(rgb_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Bound angles to [-1, 1]
        )

        self.sonar_theta_x_head = nn.Sequential(
            nn.Linear(sonar_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Bound angles to [-1, 1]
        )

        # Adaptive fusion gate for theta_x
        self.theta_x_fusion_gate = nn.Sequential(
            nn.Linear(rgb_feat_dim + sonar_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

        # Vertical viewing angle (theta_y) estimation: RGB-dominant
        # CRITICAL (Audit 2025112102): Predicts vertical viewing angle theta_y (radians)
        # Output range: [-1, 1] (via Tanh), scaled to [-max_angle, max_angle]
        self.theta_y_head = nn.Sequential(
            nn.Linear(rgb_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Bound angles to [-1, 1]
        )

        # ========================================
        # Single-RGB Bypass Branches
        # ========================================

        # RGB depth estimator: Estimate Z from RGB features alone
        # This is a challenging task as RGB lacks direct depth cues underwater
        # We use a deeper network with multi-scale reasoning
        self.rgb_depth_estimator = nn.Sequential(
            nn.Linear(rgb_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Normalize to [0, 1], will scale by radial_range
        )

        # RGB theta_x estimator (single-modality)
        # CRITICAL (Audit 2025112102): Predicts horizontal viewing angle
        self.rgb_theta_x_only_head = nn.Sequential(
            nn.Linear(rgb_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Bound angles to [-1, 1]
        )

        # RGB theta_y estimator (same as dual-modality, just for consistency)
        # We'll reuse self.theta_y_head for this

        # ========================================
        # Single-Sonar Bypass Branches
        # ========================================

        # Sonar theta_y estimator: Estimate theta_y from sonar features
        # Sonar has limited vertical resolution, so this is challenging
        # CRITICAL (Audit 2025112102): Predicts vertical viewing angle
        self.sonar_theta_y_branch = nn.Sequential(
            nn.Linear(sonar_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Bound angles to [-1, 1]
        )

        # Sonar theta_x estimator (single-modality)
        # CRITICAL (Audit 2025112102): Predicts horizontal viewing angle
        self.sonar_theta_x_only_head = nn.Sequential(
            nn.Linear(sonar_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Bound angles to [-1, 1]
        )

        # Sonar Z estimator (same as dual-modality)
        # We'll reuse radial_weights and radial_encoder

    def _convert_angles_to_coordinates(
        self,
        theta_x: torch.Tensor,
        theta_y: torch.Tensor,
        Z: torch.Tensor,
        max_angle_horizontal: float,
        max_angle_vertical: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert viewing angles (theta_x, theta_y) and depth Z to Cartesian coordinates (X, Y).

        CRITICAL (Audit 2025112102): This implements the physical constraint that sensors
        measure angles, not absolute coordinates.

        CRITICAL (Audit 2025112103): Uses hardware-specific FOV parameters for different
        modalities (RGB 47x27deg, Sonar 130deg).

        Physical relationship:
            X = Z * tan(theta_x)  # Horizontal displacement
            Y = Z * tan(theta_y)  # Vertical displacement

        Args:
            theta_x: Horizontal viewing angle (Tanh output), shape [B, 1], range [-1, 1]
            theta_y: Vertical viewing angle (Tanh output), shape [B, 1], range [-1, 1]
            Z: Depth in pixels, shape [B, 1]
            max_angle_horizontal: Maximum horizontal viewing angle in radians (FOV/2)
            max_angle_vertical: Maximum vertical viewing angle in radians (FOV/2)

        Returns:
            X: Horizontal coordinate in pixels, shape [B, 1]
            Y: Vertical coordinate in pixels, shape [B, 1]
        """
        # Scale tanh output [-1, 1] to actual angle range [-max_angle, max_angle]
        theta_x_scaled = theta_x * max_angle_horizontal
        theta_y_scaled = theta_y * max_angle_vertical

        # Convert angles to coordinates using perspective geometry
        # X = Z * tan(theta_x), Y = Z * tan(theta_y)
        X = Z * torch.tan(theta_x_scaled)
        Y = Z * torch.tan(theta_y_scaled)

        return X, Y

    def forward(
        self,
        rgb_features: torch.Tensor,
        sonar_features: torch.Tensor,
        is_rgb_valid: Optional[torch.Tensor] = None,
        is_sonar_valid: Optional[torch.Tensor] = None,
        sonar_radial_activation: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Conditional 3D center estimation with modality-aware routing.

        CRITICAL FIX (Audit 2025112101 Phase 3.1):
        Routes samples to appropriate estimation paths based on modality availability:
        - Dual-modality: Original Weak3D logic
        - Single-RGB: RGB depth estimator + RGB X/Y
        - Single-Sonar: Sonar radial Z + Sonar X + Sonar Y
        - Zero-modality: Default predictions (0, 0, default_depth)

        CRITICAL FIX (Audit 2025112101 Phase 3.3):
        When return_intermediate=True, returns intermediate predictions from all branches
        for 3D consistency loss computation.

        Args:
            rgb_features: [B, rgb_feat_dim] pooled RGB features
            sonar_features: [B, sonar_feat_dim] pooled sonar features
            is_rgb_valid: [B] boolean tensor, True if RGB has valid target (optional)
            is_sonar_valid: [B] boolean tensor, True if Sonar has valid target (optional)
            sonar_radial_activation: [B, num_radial_bins] sonar radial activation (optional)
            return_intermediate: If True, return intermediate predictions from all branches

        Returns:
            dict containing:
                - center_3d: Predicted 3D center [B, 3] in (X, Y, Z) format
                - routing_info: Dict with routing statistics
                - X, Y, Z: Individual coordinate predictions
                - intermediate (if return_intermediate=True): Dict with bypass branch outputs
        """
        B = rgb_features.shape[0]
        device = rgb_features.device

        # If no validity information provided, assume all samples are dual-modality
        if is_rgb_valid is None:
            is_rgb_valid = torch.ones(B, dtype=torch.bool, device=device)
        if is_sonar_valid is None:
            is_sonar_valid = torch.ones(B, dtype=torch.bool, device=device)

        # Detect modality availability for each sample
        zero_modality_mask = (~is_rgb_valid) & (~is_sonar_valid)  # [B]
        single_rgb_mask = is_rgb_valid & (~is_sonar_valid)  # [B]
        single_sonar_mask = (~is_rgb_valid) & is_sonar_valid  # [B]
        dual_modality_mask = is_rgb_valid & is_sonar_valid  # [B]

        # Initialize output tensors
        X_pred = torch.zeros(B, 1, device=device, dtype=rgb_features.dtype)
        Y_pred = torch.zeros(B, 1, device=device, dtype=rgb_features.dtype)
        Z_pred = torch.full((B, 1), self.default_depth, device=device, dtype=rgb_features.dtype)

        # Routing info for debugging and analysis
        routing_info = {
            'num_dual': dual_modality_mask.sum().item(),
            'num_single_rgb': single_rgb_mask.sum().item(),
            'num_single_sonar': single_sonar_mask.sum().item(),
            'num_zero': zero_modality_mask.sum().item()
        }

        # ========================================
        # Route 1: Dual-Modality → Original Weak3D Logic
        # ========================================
        if dual_modality_mask.any():
            dual_idx = dual_modality_mask.nonzero(as_tuple=False).squeeze(-1)  # [N_dual]
            rgb_dual = rgb_features[dual_idx]  # [N_dual, rgb_feat_dim]
            sonar_dual = sonar_features[dual_idx]  # [N_dual, sonar_feat_dim]

            # Z from sonar radial activation
            if sonar_radial_activation is None:
                # If not provided, extract from sonar features
                # NOTE: This is a simplified version; in practice, you'd pass it from backbone
                raise ValueError("sonar_radial_activation must be provided for dual-modality samples")

            radial_dual = sonar_radial_activation[dual_idx]  # [N_dual, num_radial_bins]

            # Weighted integral with learnable weights
            weights = F.softmax(self.radial_weights, dim=0)  # [num_radial_bins]
            Z_weighted = torch.sum(radial_dual * weights.unsqueeze(0), dim=-1, keepdim=True)  # [N_dual, 1]

            # MLP refinement with residual connection
            Z_mlp = self.radial_encoder(radial_dual)  # [N_dual, 1]
            Z_dual = Z_mlp + Z_weighted  # [N_dual, 1]
            Z_dual = torch.clamp(Z_dual, min=0.0)  # Ensure non-negative depth

            # Scale to actual depth range
            Z_dual = Z_dual * (self.radial_range[1] - self.radial_range[0]) + self.radial_range[0]

            # Predict viewing angles from dual-branch with adaptive fusion
            theta_x_rgb = self.rgb_theta_x_head(rgb_dual)  # [N_dual, 1] in [-1, 1]
            theta_x_sonar = self.sonar_theta_x_head(sonar_dual)  # [N_dual, 1] in [-1, 1]

            # Adaptive fusion weights
            fusion_features = torch.cat([rgb_dual, sonar_dual], dim=-1)  # [N_dual, rgb_feat_dim + sonar_feat_dim]
            fusion_weights = self.theta_x_fusion_gate(fusion_features)  # [N_dual, 2]

            theta_x_fused = fusion_weights[:, 0:1] * theta_x_rgb + fusion_weights[:, 1:2] * theta_x_sonar  # [N_dual, 1]

            # Predict vertical viewing angle from RGB (better vertical resolution)
            theta_y_dual = self.theta_y_head(rgb_dual)  # [N_dual, 1] in [-1, 1]

            # CRITICAL (Audit 2025112102): Convert angles to Cartesian coordinates
            # CRITICAL (Audit 2025112103): Use RGB FOV for dual-modality
            # X = Z * tan(theta_x), Y = Z * tan(theta_y)
            X_dual, Y_dual = self._convert_angles_to_coordinates(
                theta_x_fused, theta_y_dual, Z_dual,
                max_angle_horizontal=self.max_angle_rgb_horizontal,
                max_angle_vertical=self.max_angle_rgb_vertical
            )

            # Place predictions back to output tensors
            X_pred[dual_idx] = X_dual
            Y_pred[dual_idx] = Y_dual
            Z_pred[dual_idx] = Z_dual

        # ========================================
        # Route 2: Single-RGB → RGB Bypass
        # ========================================
        if single_rgb_mask.any():
            rgb_idx = single_rgb_mask.nonzero(as_tuple=False).squeeze(-1)  # [N_rgb]
            rgb_single = rgb_features[rgb_idx]  # [N_rgb, rgb_feat_dim]

            # Z from RGB depth estimator
            Z_rgb_norm = self.rgb_depth_estimator(rgb_single)  # [N_rgb, 1] in [0, 1]
            Z_rgb = Z_rgb_norm * (self.radial_range[1] - self.radial_range[0]) + self.radial_range[0]  # Scale to depth range

            # Predict viewing angles from RGB-only heads
            theta_x_rgb = self.rgb_theta_x_only_head(rgb_single)  # [N_rgb, 1] in [-1, 1]
            theta_y_rgb = self.theta_y_head(rgb_single)  # [N_rgb, 1] in [-1, 1]

            # CRITICAL (Audit 2025112102): Convert angles to Cartesian coordinates
            # CRITICAL (Audit 2025112103): Use RGB FOV for single-RGB path
            # X = Z * tan(theta_x), Y = Z * tan(theta_y)
            X_rgb, Y_rgb = self._convert_angles_to_coordinates(
                theta_x_rgb, theta_y_rgb, Z_rgb,
                max_angle_horizontal=self.max_angle_rgb_horizontal,
                max_angle_vertical=self.max_angle_rgb_vertical
            )

            # Place predictions back to output tensors
            X_pred[rgb_idx] = X_rgb
            Y_pred[rgb_idx] = Y_rgb
            Z_pred[rgb_idx] = Z_rgb

        # ========================================
        # Route 3: Single-Sonar → Sonar Bypass
        # ========================================
        if single_sonar_mask.any():
            sonar_idx = single_sonar_mask.nonzero(as_tuple=False).squeeze(-1)  # [N_sonar]
            sonar_single = sonar_features[sonar_idx]  # [N_sonar, sonar_feat_dim]

            # Z from sonar radial activation (same as dual-modality)
            if sonar_radial_activation is None:
                raise ValueError("sonar_radial_activation must be provided for single-sonar samples")

            radial_sonar = sonar_radial_activation[sonar_idx]  # [N_sonar, num_radial_bins]

            # Weighted integral with learnable weights
            weights = F.softmax(self.radial_weights, dim=0)  # [num_radial_bins]
            Z_weighted = torch.sum(radial_sonar * weights.unsqueeze(0), dim=-1, keepdim=True)  # [N_sonar, 1]

            # MLP refinement with residual connection
            Z_mlp = self.radial_encoder(radial_sonar)  # [N_sonar, 1]
            Z_sonar = Z_mlp + Z_weighted  # [N_sonar, 1]
            Z_sonar = torch.clamp(Z_sonar, min=0.0)  # Ensure non-negative depth

            # Scale to actual depth range
            Z_sonar = Z_sonar * (self.radial_range[1] - self.radial_range[0]) + self.radial_range[0]

            # Predict viewing angles from sonar-only heads
            theta_x_sonar = self.sonar_theta_x_only_head(sonar_single)  # [N_sonar, 1] in [-1, 1]
            theta_y_sonar = self.sonar_theta_y_branch(sonar_single)  # [N_sonar, 1] in [-1, 1]

            # CRITICAL (Audit 2025112102): Convert angles to Cartesian coordinates
            # CRITICAL (Audit 2025112103): Use Sonar FOV for single-Sonar path
            # NOTE: Sonar vertical FOV is typically smaller; using horizontal/2 as conservative estimate
            # X = Z * tan(theta_x), Y = Z * tan(theta_y)
            X_sonar, Y_sonar = self._convert_angles_to_coordinates(
                theta_x_sonar, theta_y_sonar, Z_sonar,
                max_angle_horizontal=self.max_angle_sonar_horizontal,
                max_angle_vertical=self.max_angle_sonar_horizontal / 2.0  # Conservative estimate
            )

            # Place predictions back to output tensors
            X_pred[sonar_idx] = X_sonar
            Y_pred[sonar_idx] = Y_sonar
            Z_pred[sonar_idx] = Z_sonar

        # ========================================
        # Route 4: Zero-Modality → Default Predictions
        # ========================================
        # Already initialized to (0, 0, default_depth), no additional processing needed

        # ========================================
        # Intermediate Outputs for 3D Consistency Loss (Phase 3.3)
        # ========================================
        # If return_intermediate=True and training mode, compute bypass branch predictions
        # for dual-modality samples to enable consistency loss
        intermediate = None
        if return_intermediate and dual_modality_mask.any():
            dual_idx = dual_modality_mask.nonzero(as_tuple=False).squeeze(-1)
            rgb_dual = rgb_features[dual_idx]
            sonar_dual = sonar_features[dual_idx]

            # Compute RGB bypass predictions (for all dual-modality samples)
            X_rgb_bypass_full = torch.zeros(B, 1, device=device, dtype=rgb_features.dtype)
            Y_rgb_bypass_full = torch.zeros(B, 1, device=device, dtype=rgb_features.dtype)
            Z_rgb_bypass_full = torch.zeros(B, 1, device=device, dtype=rgb_features.dtype)

            X_rgb_bypass_full[dual_idx] = self.rgb_x_only_head(rgb_dual)
            Y_rgb_bypass_full[dual_idx] = self.y_head(rgb_dual)
            Z_rgb_norm = self.rgb_depth_estimator(rgb_dual)
            Z_rgb_bypass_full[dual_idx] = Z_rgb_norm * (self.radial_range[1] - self.radial_range[0]) + self.radial_range[0]

            # Compute Sonar bypass predictions (for all dual-modality samples)
            X_sonar_bypass_full = torch.zeros(B, 1, device=device, dtype=rgb_features.dtype)
            Y_sonar_bypass_full = torch.zeros(B, 1, device=device, dtype=rgb_features.dtype)
            Z_sonar_bypass_full = torch.zeros(B, 1, device=device, dtype=rgb_features.dtype)

            if sonar_radial_activation is not None:
                radial_dual = sonar_radial_activation[dual_idx]
                weights = F.softmax(self.radial_weights, dim=0)
                Z_weighted = torch.sum(radial_dual * weights.unsqueeze(0), dim=-1, keepdim=True)
                Z_mlp = self.radial_encoder(radial_dual)
                Z_sonar = Z_mlp + Z_weighted
                Z_sonar = torch.clamp(Z_sonar, min=0.0)  # Ensure non-negative depth
                Z_sonar_bypass_full[dual_idx] = Z_sonar * (self.radial_range[1] - self.radial_range[0]) + self.radial_range[0]

            X_sonar_bypass_full[dual_idx] = self.sonar_x_only_head(sonar_dual)
            Y_sonar_bypass_full[dual_idx] = self.sonar_y_branch(sonar_dual)

            intermediate = {
                'X_rgb_bypass': X_rgb_bypass_full,
                'Y_rgb_bypass': Y_rgb_bypass_full,
                'Z_rgb_bypass': Z_rgb_bypass_full,
                'X_sonar_bypass': X_sonar_bypass_full,
                'Y_sonar_bypass': Y_sonar_bypass_full,
                'Z_sonar_bypass': Z_sonar_bypass_full
            }

        # Assemble 3D center prediction
        center_3d = torch.cat([X_pred, Y_pred, Z_pred], dim=-1)  # [B, 3]

        output = {
            'center_3d': center_3d,
            'routing_info': routing_info,
            'X': X_pred,
            'Y': Y_pred,
            'Z': Z_pred
        }

        if intermediate is not None:
            output['intermediate'] = intermediate

        return output


def build_weak3d_head(
    rgb_feat_dim: int = 256,
    sonar_feat_dim: int = 256,
    **kwargs
) -> Weak3DHead:
    """
    Factory function to build Weak3DHead.

    Args:
        rgb_feat_dim: RGB feature dimension
        sonar_feat_dim: Sonar feature dimension
        **kwargs: Additional arguments for Weak3DHead

    Returns:
        Weak3DHead instance

    Example:
        >>> head = build_weak3d_head(
        ...     rgb_feat_dim=256,
        ...     sonar_feat_dim=256,
        ...     num_radial_bins=50
        ... )
    """
    return Weak3DHead(
        rgb_feat_dim=rgb_feat_dim,
        sonar_feat_dim=sonar_feat_dim,
        **kwargs
    )
