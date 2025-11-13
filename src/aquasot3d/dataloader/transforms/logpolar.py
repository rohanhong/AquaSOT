"""
Differentiable Log-Polar Transform for Sonar Preprocessing

This module implements a differentiable log-polar coordinate transformation
specifically designed for Forward-Looking Sonar (FLS) images. The transformation:
- Converts fan-shaped sonar images to rectangular log-polar grids
- Encodes radial distance and angular priors
- Supports learnable scale parameters (α, β) for end-to-end optimization
- Enables calibration-free geometric adaptation

Mathematical Formulation:
    Given Cartesian coordinates (x, y) relative to sensor position:
        r = sqrt((x - cx)² + (y - cy)²)
        θ = atan2(x - cx, cy - y)

    Log-polar coordinates:
        ρ = α * log(r + 1)     # Logarithmic radial scaling
        φ = β * θ              # Angular scaling

Reference:
    Based on traditional sonar log-polar registration methods, implemented
    as a differentiable operator for end-to-end deep learning.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LogPolarTransform(nn.Module):
    """
    Differentiable log-polar transformation for sonar images.

    This module transforms fan-shaped sonar images from Cartesian coordinates
    to log-polar coordinates, making geometric patterns more regular and
    easier to process with standard CNNs.

    Args:
        sonar_height: Input sonar image height (default 376)
        sonar_width: Input sonar image width (default 682)
        output_height: Output log-polar image height (default 256)
        output_width: Output log-polar image width (default 256)
        fov_angle: Field of view angle in degrees (default 130°)
        max_range: Maximum range in pixels (None = auto-compute)
        sensor_pos: Sensor position (cx, cy), None = auto-compute
        learnable_scales: If True, α and β are learnable parameters
        init_alpha: Initial value for radial scale α (default 1.0)
        init_beta: Initial value for angular scale β (default 1.0)
        align_corners: If True, align corners of grid to boundaries (default True for backward compatibility)
                       Recommended: False for modern practice (pixel-center alignment)

    Learnable Parameters:
        alpha: Logarithmic radial scaling factor (if learnable_scales=True)
        beta: Angular scaling factor (if learnable_scales=True)

    Example:
        >>> transform = LogPolarTransform(
        ...     sonar_height=376, sonar_width=682,
        ...     output_height=256, output_width=256,
        ...     learnable_scales=True
        ... )
        >>> sonar_img = torch.randn(1, 3, 376, 682)
        >>> logpolar_img = transform(sonar_img)
        >>> print(logpolar_img.shape)
        torch.Size([1, 3, 256, 256])
    """

    def __init__(
        self,
        sonar_height: int = 376,
        sonar_width: int = 682,
        output_height: int = 256,
        output_width: int = 256,
        fov_angle: float = 130.0,
        max_range: Optional[float] = None,
        sensor_pos: Optional[Tuple[float, float]] = None,
        learnable_scales: bool = True,
        init_alpha: float = 1.0,
        init_beta: float = 1.0,
        align_corners: bool = True,
    ):
        super().__init__()

        self.sonar_height = sonar_height
        self.sonar_width = sonar_width
        self.output_height = output_height
        self.output_width = output_width
        self.fov_angle = fov_angle
        self.learnable_scales = learnable_scales
        self.align_corners = align_corners

        # Determine sensor position (bottom-center for upward-facing)
        if sensor_pos is None:
            self.cx = sonar_width / 2
            self.cy = sonar_height - 1
        else:
            self.cx, self.cy = sensor_pos

        # Determine max range
        if max_range is None:
            self.max_range = sonar_height
        else:
            self.max_range = max_range

        # Initialize learnable scale parameters
        if learnable_scales:
            self.alpha = nn.Parameter(torch.tensor(init_alpha))
            self.beta = nn.Parameter(torch.tensor(init_beta))
        else:
            self.register_buffer('alpha', torch.tensor(init_alpha))
            self.register_buffer('beta', torch.tensor(init_beta))

        # Precompute sampling grid (will be modulated by learnable scales)
        self._init_sampling_grid()

    def _init_sampling_grid(self):
        """
        Initialize base sampling grid for log-polar transformation.

        The grid maps output log-polar coordinates back to input Cartesian coordinates
        for bilinear sampling.
        """
        # Create log-polar coordinate grids
        # ρ: radial (log-scaled), φ: angular
        rho = torch.linspace(0, 1, self.output_height)  # Normalized log-radius [0, 1]
        phi = torch.linspace(-1, 1, self.output_width)  # Normalized angle [-1, 1]

        rho_grid, phi_grid = torch.meshgrid(rho, phi, indexing='ij')

        # IMPORTANT: Clone grids to avoid memory sharing issues
        # torch.meshgrid returns views that share memory, which causes problems during
        # checkpoint loading. Creating contiguous clones ensures independent memory.
        rho_grid = rho_grid.clone().contiguous()
        phi_grid = phi_grid.clone().contiguous()

        # Register as buffers (non-trainable, but move with module)
        self.register_buffer('rho_grid', rho_grid)  # (H_out, W_out)
        self.register_buffer('phi_grid', phi_grid)  # (H_out, W_out)

    def _compute_cartesian_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Cartesian sampling coordinates from log-polar grid.

        Returns:
            Tuple of (x_grid, y_grid) in Cartesian coordinates
        """
        # Convert FOV angle to radians
        fov_angle_rad = torch.deg2rad(torch.tensor(self.fov_angle, device=self.rho_grid.device))

        # Inverse log-polar transformation
        # ρ = α * log(r + 1) => r = exp(ρ / α) - 1
        # φ = β * θ => θ = φ / β

        # Denormalize rho and phi (WITHOUT multiplying by alpha/beta)
        # rho_grid is normalized [0,1], denormalize to actual ρ values
        rho_denorm = self.rho_grid * np.log(self.max_range + 1)
        # phi_grid is normalized [-1,1], denormalize to actual φ values
        phi_denorm = self.phi_grid * (fov_angle_rad / 2)

        # Apply inverse transformation with learnable scales
        # This ensures alpha and beta affect the sampling grid
        r = torch.exp(rho_denorm / (self.alpha + 1e-6)) - 1

        # Compute angle
        theta = phi_denorm / (self.beta + 1e-6)

        # Convert polar (r, θ) to Cartesian (x, y)
        # For upward-facing fan: x = cx + r * sin(θ), y = cy - r * cos(θ)
        x = self.cx + r * torch.sin(theta)
        y = self.cy - r * torch.cos(theta)

        return x, y

    def _normalize_grid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Normalize Cartesian coordinates to grid_sample format [-1, 1].

        Args:
            x: X coordinates in pixel space
            y: Y coordinates in pixel space

        Returns:
            Normalized grid (H_out, W_out, 2) in range [-1, 1]
        """
        # Normalize x to [-1, 1]
        x_norm = 2 * (x / (self.sonar_width - 1)) - 1

        # Normalize y to [-1, 1]
        y_norm = 2 * (y / (self.sonar_height - 1)) - 1

        # Stack to create grid (H, W, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1)

        return grid

    def _compute_cartesian_grid_dynamic(
        self, H: int, W: int, scale_h: float, scale_w: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Cartesian sampling coordinates for dynamically sized inputs.

        This method preserves geometric relationships by scaling sensor position
        and max range proportionally, while keeping FOV angle and learnable
        parameters (alpha, beta) unchanged.

        Mathematical Principle:
            When input size changes from (H_init, W_init) to (H, W):
            - Sensor position scales: cx_new = cx * scale_w, cy_new = cy * scale_h
            - Max range scales: max_range_new = max_range * scale_h
            - FOV angle, alpha, beta remain constant (intrinsic geometric properties)

        Args:
            H: Dynamic input height
            W: Dynamic input width
            scale_h: Height scaling factor (H / H_init)
            scale_w: Width scaling factor (W / W_init)

        Returns:
            Tuple of (x_grid, y_grid) in Cartesian coordinates
        """
        # Scale sensor position proportionally to maintain geometric center
        cx_scaled = self.cx * scale_w
        cy_scaled = self.cy * scale_h

        # Scale max range proportionally
        max_range_scaled = self.max_range * scale_h

        # Convert FOV angle to radians (same for all sizes)
        fov_angle_rad = torch.deg2rad(
            torch.tensor(self.fov_angle, device=self.rho_grid.device)
        )

        # Use base log-polar grids (output size remains constant)
        # Denormalize rho and phi using scaled max_range (WITHOUT multiplying by alpha/beta)
        rho_denorm = self.rho_grid * np.log(max_range_scaled + 1)
        phi_denorm = self.phi_grid * (fov_angle_rad / 2)

        # Apply inverse transformation with learnable scales
        # This ensures alpha and beta affect the sampling grid
        r = torch.exp(rho_denorm / (self.alpha + 1e-6)) - 1

        # Compute angle
        theta = phi_denorm / (self.beta + 1e-6)

        # Convert polar (r, θ) to Cartesian (x, y) with scaled sensor position
        # For upward-facing fan: x = cx + r * sin(θ), y = cy - r * cos(θ)
        x = cx_scaled + r * torch.sin(theta)
        y = cy_scaled - r * torch.cos(theta)

        return x, y

    def _normalize_grid_dynamic(
        self, x: torch.Tensor, y: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Normalize Cartesian coordinates for dynamic input size to grid_sample format.

        Args:
            x: X coordinates in pixel space
            y: Y coordinates in pixel space
            H: Dynamic input height
            W: Dynamic input width

        Returns:
            Normalized grid (H_out, W_out, 2) in range [-1, 1]
        """
        # Normalize x to [-1, 1] using dynamic width
        x_norm = 2 * (x / (W - 1)) - 1

        # Normalize y to [-1, 1] using dynamic height
        y_norm = 2 * (y / (H - 1)) - 1

        # Stack to create grid (H, W, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1)

        return grid

    def forward(self, sonar_img: torch.Tensor) -> torch.Tensor:
        """
        Apply log-polar transformation to sonar image.

        Args:
            sonar_img: Input sonar image (B, C, H, W) in Cartesian coordinates

        Returns:
            Transformed image (B, C, H_out, W_out) in log-polar coordinates
        """
        B, C, H, W = sonar_img.shape

        # High-generalization: adapt to input size if different from initialized size
        if H != self.sonar_height or W != self.sonar_width:
            # Dynamically compute grid for new dimensions
            # Scale parameters proportionally
            scale_h = H / self.sonar_height
            scale_w = W / self.sonar_width

            # Recompute sampling grid with scaled dimensions
            # This preserves geometric relationships
            x_grid, y_grid = self._compute_cartesian_grid_dynamic(H, W, scale_h, scale_w)
            grid_normalized = self._normalize_grid_dynamic(x_grid, y_grid, H, W)
        else:
            # Use cached grids for standard size (performance optimization)
            x_grid, y_grid = self._compute_cartesian_grid()
            grid_normalized = self._normalize_grid(x_grid, y_grid)

        # Expand grid to match batch size
        grid_batch = grid_normalized.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H_out, W_out, 2)

        # Apply bilinear sampling (differentiable)
        logpolar_img = F.grid_sample(
            sonar_img,
            grid_batch,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=self.align_corners
        )

        return logpolar_img

    def inverse_transform(self, logpolar_img: torch.Tensor) -> torch.Tensor:
        """
        Inverse log-polar transformation (log-polar -> Cartesian).

        Args:
            logpolar_img: Image in log-polar coordinates (B, C, H_out, W_out)

        Returns:
            Image in Cartesian coordinates (B, C, H, W)
        """
        B, C, H_out, W_out = logpolar_img.shape

        # Create Cartesian coordinate grid for output
        y_cart = torch.arange(self.sonar_height, dtype=torch.float32, device=logpolar_img.device)
        x_cart = torch.arange(self.sonar_width, dtype=torch.float32, device=logpolar_img.device)
        yy, xx = torch.meshgrid(y_cart, x_cart, indexing='ij')

        # Convert Cartesian to polar coordinates
        r = torch.sqrt((xx - self.cx)**2 + (yy - self.cy)**2)
        theta = torch.atan2(xx - self.cx, self.cy - yy)

        # Convert polar to log-polar
        rho = self.alpha * torch.log(r + 1)
        phi = self.beta * theta

        # Normalize to [-1, 1] for grid_sample
        fov_angle_rad = torch.deg2rad(torch.tensor(self.fov_angle, device=logpolar_img.device))
        rho_max = self.alpha * np.log(self.max_range + 1)

        rho_norm = 2 * (rho / rho_max) - 1
        phi_norm = 2 * (phi / (self.beta * fov_angle_rad / 2)) - 1

        # Clamp to valid range
        rho_norm = torch.clamp(rho_norm, -1, 1)
        phi_norm = torch.clamp(phi_norm, -1, 1)

        # Create sampling grid
        grid = torch.stack([phi_norm, rho_norm], dim=-1)  # Note: grid_sample uses (x, y) order
        grid_batch = grid.unsqueeze(0).expand(B, -1, -1, -1)

        # Sample from log-polar image
        cartesian_img = F.grid_sample(
            logpolar_img,
            grid_batch,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=self.align_corners
        )

        return cartesian_img

    def get_scale_parameters(self) -> dict:
        """
        Get current scale parameters.

        Returns:
            Dictionary with 'alpha' and 'beta' values
        """
        return {
            'alpha': self.alpha.item() if isinstance(self.alpha, nn.Parameter) else self.alpha.item(),
            'beta': self.beta.item() if isinstance(self.beta, nn.Parameter) else self.beta.item(),
        }


class LogPolarPreprocessor(nn.Module):
    """
    Convenience wrapper for log-polar preprocessing with optional normalization.

    This module combines log-polar transformation with optional intensity
    normalization and geometric masking.

    Args:
        transform_config: Configuration for LogPolarTransform
        normalize: If True, apply intensity normalization
        apply_mask: If True, apply fan-shape mask before transform

    Example:
        >>> preprocessor = LogPolarPreprocessor(
        ...     transform_config={'learnable_scales': True},
        ...     normalize=True,
        ...     apply_mask=True
        ... )
        >>> sonar_img = torch.randn(1, 3, 376, 682)
        >>> processed = preprocessor(sonar_img)
    """

    def __init__(
        self,
        transform_config: Optional[dict] = None,
        normalize: bool = True,
        apply_mask: bool = True,
    ):
        super().__init__()

        if transform_config is None:
            transform_config = {}

        self.logpolar_transform = LogPolarTransform(**transform_config)
        self.normalize = normalize
        self.apply_mask = apply_mask

        # Optional: import geometry handler for masking
        if apply_mask:
            from .sonar_geometry import SonarGeometryHandler
            self.geometry_handler = SonarGeometryHandler(
                sonar_height=transform_config.get('sonar_height', 376),
                sonar_width=transform_config.get('sonar_width', 682),
            )

    def forward(self, sonar_img: torch.Tensor) -> torch.Tensor:
        """
        Preprocess sonar image with log-polar transform.

        Args:
            sonar_img: Input sonar image (B, C, H, W)

        Returns:
            Preprocessed log-polar image (B, C, H_out, W_out)
        """
        # Apply fan mask if requested
        if self.apply_mask:
            sonar_img = self.geometry_handler.apply_fan_mask(sonar_img)

        # Apply normalization if requested
        if self.normalize:
            # Normalize to [0, 1] range per image
            B, C, H, W = sonar_img.shape
            sonar_flat = sonar_img.view(B, C, -1)
            min_vals = sonar_flat.min(dim=2, keepdim=True)[0].unsqueeze(-1)
            max_vals = sonar_flat.max(dim=2, keepdim=True)[0].unsqueeze(-1)
            sonar_img = (sonar_img - min_vals) / (max_vals - min_vals + 1e-6)

        # Apply log-polar transform
        logpolar_img = self.logpolar_transform(sonar_img)

        return logpolar_img