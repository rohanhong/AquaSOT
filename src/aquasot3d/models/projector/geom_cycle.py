"""
Geometric Cyclic Projector for Calibration-Free 3D-2D Projection

This module implements learnable geometric projectors that map 3D center coordinates
to 2D bounding boxes without requiring extrinsic calibration. It enforces cyclic
consistency (3D→2D→3D) to ensure geometric coherence.

Key Features:
- Learnable 3D→2D projection without calibration parameters
- Separate projectors for RGB and Sonar (different viewpoints)
- Cyclic consistency loss for geometric supervision
- End-to-end differentiable

Coordinate Systems:
- 3D: Fusion Frame (X→right, Y→down, Z→forward)
- 2D: Image Frame (cx, cy, w, h) in cxcywh format

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from aquasot3d.core import BaseModule


class GeometricCyclicProjector(BaseModule):
    """
    Learnable 3D→2D Geometric Projector with Cyclic Consistency.

    Projects 3D center (X, Y, Z) to 2D bounding box (cx, cy, w, h) using
    a learnable projection function. Enforces cyclic consistency to ensure
    that projecting to 2D and back to 3D preserves the original coordinates.

    Args:
        input_dim: Dimension of additional context features (default: 0)
        hidden_dim: Hidden layer dimension (default: 256)
        use_context: Whether to use context features for projection (default: False)
        projection_type: Type of projection ('mlp', 'linear', or 'perspective')

    Returns:
        Dictionary containing:
            - 'bbox_2d': Projected 2D bounding box (cx, cy, w, h), shape (B, 4)
            - 'center_3d_reconstructed': Reconstructed 3D center (optional)
            - 'projection_matrix': Learned projection parameters (for analysis)

    Example:
        >>> projector = GeometricCyclicProjector()
        >>> center_3d = torch.randn(4, 3)  # (X, Y, Z)
        >>> outputs = projector(center_3d)
        >>> print(outputs['bbox_2d'].shape)  # (4, 4)
    """

    def __init__(
        self,
        input_dim: int = 0,
        hidden_dim: int = 256,
        use_context: bool = False,
        projection_type: str = 'mlp',
        **kwargs
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_context = use_context
        self.projection_type = projection_type

        # Forward projection: 3D → 2D
        if projection_type == 'mlp':
            # Non-linear projection with MLP
            projection_input_dim = 3 + input_dim if use_context else 3
            self.forward_projector = nn.Sequential(
                nn.Linear(projection_input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 4)  # Output: (cx, cy, w, h)
            )
        elif projection_type == 'linear':
            # Simple linear projection
            projection_input_dim = 3 + input_dim if use_context else 3
            self.forward_projector = nn.Linear(projection_input_dim, 4)
        elif projection_type == 'perspective':
            # Perspective projection with learnable intrinsics
            self.focal_length = nn.Parameter(torch.tensor([1.0, 1.0]))
            self.principal_point = nn.Parameter(torch.tensor([0.0, 0.0]))
            self.size_scale = nn.Parameter(torch.tensor([1.0, 1.0]))
        else:
            raise ValueError(f"Unknown projection_type: {projection_type}")

        # Backward projection: 2D → 3D (for cyclic consistency)
        if projection_type in ['mlp', 'linear']:
            if projection_type == 'mlp':
                self.backward_projector = nn.Sequential(
                    nn.Linear(4, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 3)  # Output: (X, Y, Z)
                )
            else:
                self.backward_projector = nn.Linear(4, 3)

        self.logger.info(
            f"Initialized GeometricCyclicProjector: input_dim={input_dim}, "
            f"hidden_dim={hidden_dim}, projection_type={projection_type}"
        )

    def project_3d_to_2d(
        self,
        center_3d: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Project 3D center to 2D bounding box with metadata-aware geometry adjustment.

        CRITICAL (Audit 2025111411 Finding 3): Output is in cxcywh format.
        Callers should wrap output using BBoxCXCYWH for type-safe handling.

        CRITICAL FIX (Audit 2025111413 Finding 2): Now accepts metadata to adjust
        projection geometry based on actual crop scale/padding, reconciling resized
        crops with true sensor geometry.

        Args:
            center_3d: 3D center coordinates (X, Y, Z), shape (B, 3)
            context: Optional context features, shape (B, C)
            metadata: Optional metadata list with per-sample scale/padding info

        Returns:
            2D bounding box in cxcywh format (cx, cy, w, h), shape (B, 4)
        """
        if self.projection_type == 'perspective':
            # Perspective projection
            X, Y, Z = center_3d.unbind(-1)

            # Avoid division by zero
            Z_safe = Z.clamp(min=0.1)

            # Project to image plane
            fx, fy = self.focal_length
            cx_offset, cy_offset = self.principal_point
            sx, sy = self.size_scale

            # CRITICAL FIX (Audit 2025111414 Finding 2): Apply metadata scale correction
            # Use scale_x/scale_y instead of unified 'scale_factor'
            if metadata is not None and len(metadata) > 0:
                batch_size = center_3d.shape[0]

                # CRITICAL FIX (Audit 2025111413 Finding 4): Validate metadata list
                assert isinstance(metadata, (list, tuple)), \
                    f"Metadata must be list/tuple for projector, got {type(metadata)}"
                assert len(metadata) >= batch_size, \
                    f"Metadata list too short: {len(metadata)} < {batch_size}"

                scale_correction_x = torch.ones(batch_size, device=center_3d.device)
                scale_correction_y = torch.ones(batch_size, device=center_3d.device)

                for i in range(min(batch_size, len(metadata))):
                    if metadata[i]:
                        # Support both scale_factor (unified) and scale_x/scale_y (per-axis)
                        if 'scale_factor' in metadata[i]:
                            scale_correction_x[i] = metadata[i]['scale_factor']
                            scale_correction_y[i] = metadata[i]['scale_factor']
                        else:
                            scale_correction_x[i] = metadata[i].get('scale_x', 1.0)
                            scale_correction_y[i] = metadata[i].get('scale_y', 1.0)

                # Adjust focal length and size scale based on crop rescaling
                fx_adjusted = fx * scale_correction_x
                fy_adjusted = fy * scale_correction_y
                sx_adjusted = sx * scale_correction_x
                sy_adjusted = sy * scale_correction_y
            else:
                fx_adjusted = fx
                fy_adjusted = fy
                sx_adjusted = sx
                sy_adjusted = sy

            cx = (fx_adjusted * X / Z_safe) + cx_offset
            cy = (fy_adjusted * Y / Z_safe) + cy_offset

            # Estimate bounding box size based on depth
            # Objects closer (smaller Z) appear larger
            w = sx_adjusted / Z_safe
            h = sy_adjusted / Z_safe

            bbox_2d = torch.stack([cx, cy, w, h], dim=-1)
        else:
            # MLP or linear projection
            if self.use_context and context is not None:
                proj_input = torch.cat([center_3d, context], dim=-1)
            else:
                proj_input = center_3d

            bbox_2d = self.forward_projector(proj_input)

            # CRITICAL FIX (Audit 2025111414 Finding 2): Apply metadata scale correction
            # Use scale_x/scale_y instead of unified 'scale_factor'
            if metadata is not None and len(metadata) > 0:
                batch_size = center_3d.shape[0]

                # CRITICAL FIX (Audit 2025111413 Finding 4): Validate metadata list
                assert isinstance(metadata, (list, tuple)), \
                    f"Metadata must be list/tuple for projector, got {type(metadata)}"
                assert len(metadata) >= batch_size, \
                    f"Metadata list too short: {len(metadata)} < {batch_size}"

                scale_correction_x = torch.ones(batch_size, 1, device=center_3d.device)
                scale_correction_y = torch.ones(batch_size, 1, device=center_3d.device)

                for i in range(min(batch_size, len(metadata))):
                    if metadata[i]:
                        # Support both scale_factor (unified) and scale_x/scale_y (per-axis)
                        if 'scale_factor' in metadata[i]:
                            scale_correction_x[i, 0] = metadata[i]['scale_factor']
                            scale_correction_y[i, 0] = metadata[i]['scale_factor']
                        else:
                            scale_correction_x[i, 0] = metadata[i].get('scale_x', 1.0)
                            scale_correction_y[i, 0] = metadata[i].get('scale_y', 1.0)

                # Adjust projected coordinates and sizes (separate x and y scaling)
                bbox_2d_cx = bbox_2d[:, 0:1] * scale_correction_x
                bbox_2d_cy = bbox_2d[:, 1:2] * scale_correction_y
                bbox_2d_w = F.softplus(bbox_2d[:, 2:3]) * scale_correction_x
                bbox_2d_h = F.softplus(bbox_2d[:, 3:4]) * scale_correction_y
                bbox_2d_xy = torch.cat([bbox_2d_cx, bbox_2d_cy], dim=-1)
                bbox_2d_wh = torch.cat([bbox_2d_w, bbox_2d_h], dim=-1)
            else:
                # Ensure positive width and height (avoid inplace for gradient)
                bbox_2d_xy = bbox_2d[:, :2]
                bbox_2d_wh = F.softplus(bbox_2d[:, 2:])

            bbox_2d = torch.cat([bbox_2d_xy, bbox_2d_wh], dim=-1)

        return bbox_2d

    def project_2d_to_3d(self, bbox_2d: torch.Tensor) -> torch.Tensor:
        """
        Project 2D bounding box back to 3D center (for cyclic consistency).

        CRITICAL (Audit 2025111411 Finding 3): Input must be in cxcywh format.
        Callers should pass BBoxCXCYWH.tensor() for type-safe handling.

        Args:
            bbox_2d: 2D bounding box in cxcywh format (cx, cy, w, h), shape (B, 4)

        Returns:
            Reconstructed 3D center (X, Y, Z), shape (B, 3)
        """
        if self.projection_type == 'perspective':
            # Inverse perspective projection
            cx, cy, w, h = bbox_2d.unbind(-1)

            fx, fy = self.focal_length
            cx_offset, cy_offset = self.principal_point
            sx, sy = self.size_scale

            # Estimate depth from size
            Z = (sx / w.clamp(min=1e-4) + sy / h.clamp(min=1e-4)) / 2

            # Back-project to 3D
            X = (cx - cx_offset) * Z / fx
            Y = (cy - cy_offset) * Z / fy

            center_3d = torch.stack([X, Y, Z], dim=-1)
        else:
            # MLP or linear back-projection
            center_3d = self.backward_projector(bbox_2d)

        return center_3d

    def forward(
        self,
        center_3d: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        compute_cyclic_loss: bool = False,
        metadata: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: Project 3D to 2D with metadata-aware geometry.

        CRITICAL FIX (Audit 2025111413 Finding 2): Now accepts metadata to ensure
        cyclic losses operate on real crop geometry instead of ideal assumptions.

        Args:
            center_3d: 3D center coordinates (X, Y, Z), shape (B, 3)
            context: Optional context features, shape (B, C)
            compute_cyclic_loss: Whether to compute cyclic consistency loss
            metadata: Optional metadata list with per-sample scale/padding info

        Returns:
            Dictionary with projection results
        """
        # Forward projection: 3D → 2D with metadata awareness
        bbox_2d = self.project_3d_to_2d(center_3d, context, metadata)

        outputs = {
            'bbox_2d': bbox_2d,
        }

        if compute_cyclic_loss:
            # Backward projection: 2D → 3D
            center_3d_reconstructed = self.project_2d_to_3d(bbox_2d)

            # Cyclic consistency loss
            cyclic_loss = F.l1_loss(center_3d_reconstructed, center_3d)

            outputs['center_3d_reconstructed'] = center_3d_reconstructed
            outputs['cyclic_loss'] = cyclic_loss

        return outputs

    def compute_projection_loss(
        self,
        pred_bbox_2d: torch.Tensor,
        target_bbox_2d: torch.Tensor,
        loss_type: str = 'l1'
    ) -> torch.Tensor:
        """
        Compute 2D projection loss with enforced type-safe format checking.

        CRITICAL (Audit 2025111411 Finding 3): Enforces cxcywh format using
        type-safe wrappers to prevent silent format confusion.

        Args:
            pred_bbox_2d: Predicted 2D box in cxcywh format, shape (B, 4)
            target_bbox_2d: Target 2D box in cxcywh format, shape (B, 4)
            loss_type: Loss type ('l1', 'l2', or 'smooth_l1')

        Returns:
            Projection loss (scalar)

        Raises:
            ValueError: If inputs are not [B, 4] tensors in cxcywh format
        """
        # CRITICAL: Type-safe format enforcement (Audit 2025111411 Finding 3)
        from aquasot3d.utils.bbox import BBoxCXCYWH

        # Wrap inputs to enforce format contract and validate shape
        try:
            pred_bbox_wrapper = BBoxCXCYWH(pred_bbox_2d)
            target_bbox_wrapper = BBoxCXCYWH(target_bbox_2d)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Bbox format validation failed in compute_projection_loss: {e}. "
                f"Expected [B, 4] tensors in cxcywh format. "
                f"Got pred shape={pred_bbox_2d.shape}, target shape={target_bbox_2d.shape}"
            )

        # Extract validated tensors for loss computation
        pred_tensor = pred_bbox_wrapper.tensor()
        target_tensor = target_bbox_wrapper.tensor()

        # Compute loss using validated tensors
        if loss_type == 'l1':
            loss = F.l1_loss(pred_tensor, target_tensor)
        elif loss_type == 'l2':
            loss = F.mse_loss(pred_tensor, target_tensor)
        elif loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred_tensor, target_tensor)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return loss


class RGBProjector(GeometricCyclicProjector):
    """
    RGB-specific Geometric Projector.

    Specialized projector for RGB modality, optimized for camera-based
    perspective projection characteristics.
    """

    def __init__(self, **kwargs):
        # Default to perspective projection for RGB
        if 'projection_type' not in kwargs:
            kwargs['projection_type'] = 'perspective'
        super().__init__(**kwargs)
        self.logger.info("Initialized RGB-specific Geometric Projector")


class SonarProjector(GeometricCyclicProjector):
    """
    Sonar-specific Geometric Projector.

    Specialized projector for Sonar modality, accounting for sonar-specific
    geometric properties (fan-shaped FOV, radial coordinates).
    """

    def __init__(self, **kwargs):
        # Default to MLP projection for Sonar (non-standard geometry)
        if 'projection_type' not in kwargs:
            kwargs['projection_type'] = 'mlp'
        super().__init__(**kwargs)
        self.logger.info("Initialized Sonar-specific Geometric Projector")


def build_geometric_projector(
    modality: str = 'rgb',
    **kwargs
) -> GeometricCyclicProjector:
    """
    Factory function to build geometric projector.

    Args:
        modality: Modality type ('rgb', 'sonar', or 'generic')
        **kwargs: Additional arguments for projector

    Returns:
        Geometric projector instance

    Example:
        >>> rgb_proj = build_geometric_projector('rgb')
        >>> sonar_proj = build_geometric_projector('sonar', hidden_dim=256)
    """
    if modality == 'rgb':
        return RGBProjector(**kwargs)
    elif modality == 'sonar':
        return SonarProjector(**kwargs)
    else:
        return GeometricCyclicProjector(**kwargs)
