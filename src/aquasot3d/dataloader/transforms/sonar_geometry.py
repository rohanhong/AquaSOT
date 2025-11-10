"""
Sonar Geometry Handler

This module handles geometric properties of Forward-Looking Sonar (FLS) images:
- Fan-shape field of view masks
- Radial resolution weighting (distance-dependent)
- Angular sector sampling
- Sensor position and orientation modeling

Sonar Characteristics:
    - Resolution: 682×376 pixels (standard FLS)
    - Field of view: Fan-shaped sector (typically 130° FOV)
    - Sensor position: Bottom-center for upward-facing configuration
    - Resolution degradation: Decreases with range from sensor

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_fan_mask(
    height: int,
    width: int,
    max_range: Optional[float] = None,
    fov_angle: float = 130.0,
    sensor_pos: Optional[Tuple[float, float]] = None,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Generate binary mask for sonar fan-shaped field of view.

    The mask indicates valid scanning regions (1) vs. background regions (0).
    Default configuration assumes upward-facing fan from bottom-center.

    Args:
        height: Sonar image height in pixels
        width: Sonar image width in pixels
        max_range: Maximum effective range in pixels (None = use full height)
        fov_angle: Field of view angle in degrees (e.g., 130°)
        sensor_pos: Sensor position (cx, cy), None = auto-compute
        device: Torch device for computation

    Returns:
        Binary mask tensor (1, height, width) where 1=valid, 0=background

    Example:
        >>> mask = generate_fan_mask(376, 682, fov_angle=130)
        >>> print(mask.shape)
        torch.Size([1, 376, 682])
        >>> print(mask.sum() / mask.numel())  # Valid pixel ratio
        tensor(0.6234)
    """
    # Auto-determine sensor position (bottom-center for upward-facing)
    if sensor_pos is None:
        cx = width / 2
        cy = height - 1  # Bottom edge
    else:
        cx, cy = sensor_pos

    # Auto-determine max range if not specified
    if max_range is None:
        max_range = height  # Use full image height

    # Create coordinate grids
    y = torch.arange(height, dtype=torch.float32, device=device)
    x = torch.arange(width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Calculate radial distance from sensor
    r = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

    # Calculate angle from vertical axis (0° = upward)
    # For upward-facing fan: use atan2(x - cx, cy - y)
    # This gives angle relative to upward direction
    theta = torch.atan2(xx - cx, cy - yy)  # Range: [-π, π], 0 = upward

    # Convert FOV angle to radians
    fov_angle_rad = torch.deg2rad(torch.tensor(fov_angle, device=device))

    # Apply fan-shape constraints
    # 1. Range constraint: r <= max_range
    # 2. Angular constraint: |theta| <= fov_angle / 2
    range_mask = r <= max_range
    angle_mask = torch.abs(theta) <= (fov_angle_rad / 2)

    # Combine constraints
    mask = (range_mask & angle_mask).float()

    # Add batch dimension
    mask = mask.unsqueeze(0)  # (1, H, W)

    return mask


def compute_radial_weights(
    height: int,
    width: int,
    sensor_pos: Optional[Tuple[float, float]] = None,
    decay_factor: float = 0.5,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Compute position-dependent quality weights based on radial distance.

    Near-range sonar returns have better resolution than far-range.
    This function generates a weight map with exponential decay.

    Args:
        height: Sonar image height
        width: Sonar image width
        sensor_pos: Sensor position (cx, cy), None = auto-compute
        decay_factor: Controls decay rate (larger = faster decay)
        device: Torch device

    Returns:
        Weight tensor (1, height, width) with values in [0, 1]

    Example:
        >>> weights = compute_radial_weights(376, 682, decay_factor=0.5)
        >>> print(weights.max(), weights.min())
        tensor(1.0000) tensor(0.0183)
    """
    # Auto-determine sensor position
    if sensor_pos is None:
        cx = width / 2
        cy = height - 1
    else:
        cx, cy = sensor_pos

    # Create coordinate grids
    y = torch.arange(height, dtype=torch.float32, device=device)
    x = torch.arange(width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Calculate radial distance
    r = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

    # Normalize distance to [0, 1] range
    r_normalized = r / (height * decay_factor)

    # Exponential decay: weights = exp(-r_normalized)
    weights = torch.exp(-r_normalized)

    # Add batch dimension
    weights = weights.unsqueeze(0)  # (1, H, W)

    return weights


def compute_angular_resolution(
    height: int,
    width: int,
    fov_angle: float = 130.0,
    num_angular_bins: int = 64,
    sensor_pos: Optional[Tuple[float, float]] = None,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Compute angular resolution map for sonar image.

    Angular resolution varies across the field of view in sonar imaging.
    This function creates a map indicating angular bin assignment for each pixel.

    Args:
        height: Sonar image height
        width: Sonar image width
        fov_angle: Field of view angle in degrees
        num_angular_bins: Number of discrete angular bins
        sensor_pos: Sensor position (cx, cy)
        device: Torch device

    Returns:
        Angular bin indices (1, height, width) with values in [0, num_angular_bins-1]
    """
    # Auto-determine sensor position
    if sensor_pos is None:
        cx = width / 2
        cy = height - 1
    else:
        cx, cy = sensor_pos

    # Create coordinate grids
    y = torch.arange(height, dtype=torch.float32, device=device)
    x = torch.arange(width, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Calculate angle from vertical axis
    theta = torch.atan2(xx - cx, cy - yy)  # Range: [-π, π]

    # Convert FOV angle to radians
    fov_angle_rad = torch.deg2rad(torch.tensor(fov_angle, device=device))

    # Normalize theta to [0, 1] range within FOV
    # theta_normalized = (theta + fov_angle_rad/2) / fov_angle_rad
    theta_normalized = (theta + fov_angle_rad / 2) / fov_angle_rad
    theta_normalized = torch.clamp(theta_normalized, 0, 1)

    # Map to angular bins
    angular_bins = (theta_normalized * (num_angular_bins - 1)).long()

    # Add batch dimension
    angular_bins = angular_bins.unsqueeze(0)  # (1, H, W)

    return angular_bins


class SonarGeometryHandler(nn.Module):
    """
    PyTorch module for handling sonar geometric properties.

    This module generates and caches geometric masks and weight maps for
    sonar images, enabling efficient geometric-aware processing.

    Args:
        sonar_height: Sonar image height (default 376)
        sonar_width: Sonar image width (default 682)
        fov_angle: Field of view angle in degrees (default 130°)
        max_range: Maximum effective range in pixels (None = full height)
        decay_factor: Radial weight decay factor (default 0.5)
        sensor_pos: Sensor position (cx, cy), None = auto-compute

    Attributes:
        fan_mask: Binary fan-shape mask (1, H, W)
        radial_weights: Radial quality weights (1, H, W)
        angular_bins: Angular bin indices (1, H, W)

    Example:
        >>> geometry = SonarGeometryHandler(sonar_height=376, sonar_width=682)
        >>> mask = geometry.fan_mask
        >>> weights = geometry.radial_weights
        >>> weighted_img = sonar_img * weights
    """

    def __init__(
        self,
        sonar_height: int = 376,
        sonar_width: int = 682,
        fov_angle: float = 130.0,
        max_range: Optional[float] = None,
        decay_factor: float = 0.5,
        sensor_pos: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()

        self.sonar_height = sonar_height
        self.sonar_width = sonar_width
        self.fov_angle = fov_angle
        self.max_range = max_range
        self.decay_factor = decay_factor
        self.sensor_pos = sensor_pos

        # Register buffers (non-trainable tensors that move with module)
        self.register_buffer(
            'fan_mask',
            generate_fan_mask(
                sonar_height, sonar_width, max_range, fov_angle, sensor_pos
            )
        )

        self.register_buffer(
            'radial_weights',
            compute_radial_weights(
                sonar_height, sonar_width, sensor_pos, decay_factor
            )
        )

        self.register_buffer(
            'angular_bins',
            compute_angular_resolution(
                sonar_height, sonar_width, fov_angle, sensor_pos=sensor_pos
            )
        )

    def apply_fan_mask(self, sonar_img: torch.Tensor) -> torch.Tensor:
        """
        Apply fan-shape mask to sonar image.

        Args:
            sonar_img: Sonar image tensor (B, C, H, W)

        Returns:
            Masked sonar image (B, C, H, W)
        """
        # Expand mask to match batch and channel dimensions
        mask = self.fan_mask.unsqueeze(0)  # (1, 1, H, W)

        # Check if mask size matches input image size
        if mask.shape[2:] != sonar_img.shape[2:]:
            # Resize mask to match input dimensions
            mask = F.interpolate(
                mask,
                size=sonar_img.shape[2:],
                mode='nearest'
            )

        return sonar_img * mask

    def apply_radial_weighting(self, sonar_img: torch.Tensor) -> torch.Tensor:
        """
        Apply radial quality weighting to sonar image.

        Args:
            sonar_img: Sonar image tensor (B, C, H, W)

        Returns:
            Radially weighted sonar image (B, C, H, W)
        """
        weights = self.radial_weights.unsqueeze(0)  # (1, 1, H, W)

        # Check if weights size matches input image size
        if weights.shape[2:] != sonar_img.shape[2:]:
            # Resize weights to match input dimensions
            weights = F.interpolate(
                weights,
                size=sonar_img.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        return sonar_img * weights

    def get_radial_profile(
        self,
        sonar_img: torch.Tensor,
        num_bins: int = 50
    ) -> torch.Tensor:
        """
        Extract radial activation profile from sonar image.

        This computes the average activation at different radial distances,
        useful for depth estimation in weak 3D prediction.

        Args:
            sonar_img: Sonar image tensor (B, C, H, W)
            num_bins: Number of radial bins

        Returns:
            Radial profile (B, num_bins)
        """
        B, C, H, W = sonar_img.shape

        # Get sensor position
        if self.sensor_pos is None:
            cx = W / 2
            cy = H - 1
        else:
            cx, cy = self.sensor_pos

        # Create radial distance map
        y = torch.arange(H, dtype=torch.float32, device=sonar_img.device)
        x = torch.arange(W, dtype=torch.float32, device=sonar_img.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        r = torch.sqrt((xx - cx)**2 + (yy - cy)**2)

        # Normalize to [0, num_bins) range
        r_max = r.max()
        r_normalized = (r / r_max) * (num_bins - 1)
        r_bins = r_normalized.long().clamp(0, num_bins - 1)

        # Average activation per bin across all channels
        radial_profile = []
        for b in range(B):
            profile = []
            for bin_idx in range(num_bins):
                mask = (r_bins == bin_idx)
                if mask.sum() > 0:
                    # Mean activation in this bin (average across channels)
                    bin_activation = sonar_img[b, :, mask].mean()
                else:
                    bin_activation = torch.tensor(0.0, device=sonar_img.device)
                profile.append(bin_activation)
            radial_profile.append(torch.stack(profile))

        return torch.stack(radial_profile)  # (B, num_bins)

    def forward(self, sonar_img: torch.Tensor) -> dict:
        """
        Process sonar image with geometric handling.

        Args:
            sonar_img: Sonar image tensor (B, C, H, W)

        Returns:
            Dictionary containing:
                - masked: Fan-masked sonar image
                - weighted: Radially weighted sonar image
                - radial_profile: Radial activation profile
        """
        masked = self.apply_fan_mask(sonar_img)
        weighted = self.apply_radial_weighting(sonar_img)
        radial_profile = self.get_radial_profile(sonar_img)

        return {
            'masked': masked,
            'weighted': weighted,
            'radial_profile': radial_profile,
        }