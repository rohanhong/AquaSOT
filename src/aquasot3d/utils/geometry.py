"""
Geometry Utilities for 3D-2D Projection and Coordinate Transforms

Provides coordinate system definitions, transformations, and cyclic consistency
for weak 3D center estimation without extrinsic calibration.

Coordinate System Convention:
- Fusion Frame: X→right, Y→down, Z→forward (RGB-centric, right-handed)
- RGB Frame: Same as Fusion Frame (identity transform)
- Sonar Frame: X→right, Y→down, Z→forward (sensor-centric)

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class CoordinateTransform(nn.Module):
    """
    Learnable coordinate transformation for Sonar→Fusion without calibration.

    Uses a learnable 3x4 affine transformation matrix [R|t].

    Args:
        init_identity: Whether to initialize as identity transform
        learnable: Whether transformation is learnable
    """

    def __init__(
        self,
        init_identity: bool = True,
        learnable: bool = True
    ):
        super().__init__()

        # Initialize 3x4 affine matrix [R|t]
        if init_identity:
            # Identity rotation + zero translation
            init_matrix = torch.eye(3, 4)
        else:
            # Random initialization
            init_matrix = torch.randn(3, 4) * 0.01

        if learnable:
            self.transform_matrix = nn.Parameter(init_matrix)
        else:
            self.register_buffer('transform_matrix', init_matrix)

        self.learnable = learnable

    def forward(self, points_3d: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to 3D points.

        Args:
            points_3d: Input 3D points (B, 3) or (B, N, 3)

        Returns:
            Transformed 3D points
        """
        if points_3d.dim() == 2:
            # (B, 3) -> (B, 4) homogeneous
            ones = torch.ones(points_3d.shape[0], 1, device=points_3d.device)
            points_homo = torch.cat([points_3d, ones], dim=1)  # (B, 4)

            # Transform
            transformed = torch.matmul(points_homo, self.transform_matrix.t())  # (B, 3)

        elif points_3d.dim() == 3:
            # (B, N, 3) -> (B, N, 4)
            B, N, _ = points_3d.shape
            ones = torch.ones(B, N, 1, device=points_3d.device)
            points_homo = torch.cat([points_3d, ones], dim=2)  # (B, N, 4)

            # Transform
            transformed = torch.matmul(points_homo, self.transform_matrix.t())  # (B, N, 3)

        else:
            raise ValueError(f"Unsupported input shape: {points_3d.shape}")

        return transformed


def project_3d_to_2d_rgb(
    center_3d: torch.Tensor,
    focal_length: float = 1000.0,
    principal_point: Optional[Tuple[float, float]] = None
) -> torch.Tensor:
    """
    Weak perspective projection for RGB.

    Assumption: Weak perspective assumes that Z variation is small relative
    to the average distance from camera to object, allowing orthographic
    approximation. This is valid for underwater tracking where depth changes
    are limited.

    Args:
        center_3d: 3D center in Fusion Frame (B, 3) as [X, Y, Z]
        focal_length: Approximate focal length (pixels, unused in weak projection)
        principal_point: Principal point (cx0, cy0), if None assumes (0, 0)

    Returns:
        2D center (B, 2) as [cx, cy] in RGB image coordinates
    """
    X = center_3d[:, 0]  # (B,)
    Y = center_3d[:, 1]  # (B,)
    Z = center_3d[:, 2]  # (B,)

    # Weak perspective: ignore Z in projection (orthographic approximation)
    # Full perspective would be: cx = focal_length * X / Z + cx0
    # But we assume Z ≈ constant for weak 3D estimation
    cx = X
    cy = Y

    # Add principal point offset if provided
    if principal_point is not None:
        cx = cx + principal_point[0]
        cy = cy + principal_point[1]

    center_2d = torch.stack([cx, cy], dim=1)  # (B, 2)

    return center_2d


def project_3d_to_2d_sonar(
    center_3d: torch.Tensor,
    transform_matrix: torch.Tensor
) -> torch.Tensor:
    """
    Learned projection for Sonar (no calibration).

    Uses learnable 3x4 affine transformation matrix to project
    3D points to 2D sonar image coordinates.

    Args:
        center_3d: 3D center in Fusion Frame (B, 3) as [X, Y, Z]
        transform_matrix: Learnable 3x4 affine matrix [R|t]

    Returns:
        2D center (B, 2) as [cx, cy] in Sonar image coordinates
    """
    # Convert to homogeneous coordinates
    B = center_3d.shape[0]
    ones = torch.ones(B, 1, device=center_3d.device)
    points_homo = torch.cat([center_3d, ones], dim=1)  # (B, 4)

    # Apply transformation
    points_2d_homo = torch.matmul(points_homo, transform_matrix.t())  # (B, 3)

    # Extract 2D coordinates (ignore third dimension)
    cx = points_2d_homo[:, 0]
    cy = points_2d_homo[:, 1]

    center_2d = torch.stack([cx, cy], dim=1)  # (B, 2)

    return center_2d


def bbox_to_center(bbox: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding box to center coordinates.

    Args:
        bbox: Bounding boxes (B, 4) in format [x, y, w, h]

    Returns:
        Centers (B, 2) as [cx, cy]
    """
    cx = bbox[:, 0] + bbox[:, 2] / 2.0
    cy = bbox[:, 1] + bbox[:, 3] / 2.0

    return torch.stack([cx, cy], dim=1)


def center_to_bbox(
    center: torch.Tensor,
    size: torch.Tensor
) -> torch.Tensor:
    """
    Convert center and size to bounding box.

    Args:
        center: Centers (B, 2) as [cx, cy]
        size: Sizes (B, 2) as [w, h]

    Returns:
        Bounding boxes (B, 4) in format [x, y, w, h]
    """
    x = center[:, 0] - size[:, 0] / 2.0
    y = center[:, 1] - size[:, 1] / 2.0
    w = size[:, 0]
    h = size[:, 1]

    return torch.stack([x, y, w, h], dim=1)


class CyclicConsistencyLoss(nn.Module):
    """
    3D→2D→3D cyclic consistency loss.

    Projects predicted 3D center to 2D for both modalities and compares
    with ground truth 2D centers. Enforces geometric consistency without
    requiring extrinsic calibration.

    Args:
        loss_type: Type of loss ('l1', 'l2', 'smooth_l1')
        weight_rgb: Weight for RGB projection loss
        weight_sonar: Weight for Sonar projection loss
        normalize: Whether to normalize by image size

    Example:
        >>> criterion = CyclicConsistencyLoss(loss_type='l2')
        >>> pred_3d = torch.randn(4, 3)
        >>> gt_rgb_bbox = torch.tensor([[64, 64, 128, 128]] * 4, dtype=torch.float32)
        >>> gt_sonar_bbox = torch.tensor([[100, 200, 150, 120]] * 4, dtype=torch.float32)
        >>> transform = torch.eye(3, 4)
        >>> loss = criterion(pred_3d, gt_rgb_bbox, gt_sonar_bbox, transform)
    """

    def __init__(
        self,
        loss_type: str = 'l2',
        weight_rgb: float = 1.0,
        weight_sonar: float = 1.0,
        normalize: bool = False
    ):
        super().__init__()

        self.loss_type = loss_type
        self.weight_rgb = weight_rgb
        self.weight_sonar = weight_sonar
        self.normalize = normalize

    def forward(
        self,
        pred_3d: torch.Tensor,
        gt_rgb_bbox: torch.Tensor,
        gt_sonar_bbox: torch.Tensor,
        sonar_transform_matrix: torch.Tensor,
        rgb_img_size: Optional[Tuple[int, int]] = None,
        sonar_img_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute cyclic consistency loss.

        Args:
            pred_3d: Predicted 3D center (B, 3) as [X, Y, Z]
            gt_rgb_bbox: Ground truth RGB bbox (B, 4) as [x, y, w, h]
            gt_sonar_bbox: Ground truth Sonar bbox (B, 4) as [x, y, w, h]
            sonar_transform_matrix: Learnable sonar projection (3, 4)
            rgb_img_size: RGB image size (H, W) for normalization
            sonar_img_size: Sonar image size (H, W) for normalization

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Project 3D to 2D
        proj_rgb_2d = project_3d_to_2d_rgb(pred_3d)  # (B, 2)
        proj_sonar_2d = project_3d_to_2d_sonar(pred_3d, sonar_transform_matrix)  # (B, 2)

        # Ground truth 2D centers
        gt_rgb_center = bbox_to_center(gt_rgb_bbox)  # (B, 2)
        gt_sonar_center = bbox_to_center(gt_sonar_bbox)  # (B, 2)

        # Normalize if requested
        if self.normalize:
            if rgb_img_size is not None:
                rgb_h, rgb_w = rgb_img_size
                proj_rgb_2d = proj_rgb_2d / torch.tensor(
                    [rgb_w, rgb_h], device=proj_rgb_2d.device
                )
                gt_rgb_center = gt_rgb_center / torch.tensor(
                    [rgb_w, rgb_h], device=gt_rgb_center.device
                )

            if sonar_img_size is not None:
                sonar_h, sonar_w = sonar_img_size
                proj_sonar_2d = proj_sonar_2d / torch.tensor(
                    [sonar_w, sonar_h], device=proj_sonar_2d.device
                )
                gt_sonar_center = gt_sonar_center / torch.tensor(
                    [sonar_w, sonar_h], device=gt_sonar_center.device
                )

        # Compute losses
        if self.loss_type == 'l1':
            loss_rgb = F.l1_loss(proj_rgb_2d, gt_rgb_center, reduction='mean')
            loss_sonar = F.l1_loss(proj_sonar_2d, gt_sonar_center, reduction='mean')

        elif self.loss_type == 'l2':
            loss_rgb = F.mse_loss(proj_rgb_2d, gt_rgb_center, reduction='mean')
            loss_sonar = F.mse_loss(proj_sonar_2d, gt_sonar_center, reduction='mean')

        elif self.loss_type == 'smooth_l1':
            loss_rgb = F.smooth_l1_loss(proj_rgb_2d, gt_rgb_center, reduction='mean')
            loss_sonar = F.smooth_l1_loss(proj_sonar_2d, gt_sonar_center, reduction='mean')

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Weighted combination
        total_loss = self.weight_rgb * loss_rgb + self.weight_sonar * loss_sonar

        # Loss dictionary
        loss_dict = {
            'cyclic_total': total_loss,
            'cyclic_rgb': loss_rgb,
            'cyclic_sonar': loss_sonar,
            'proj_rgb_2d': proj_rgb_2d,
            'proj_sonar_2d': proj_sonar_2d,
        }

        return total_loss, loss_dict


def compute_reprojection_error(
    pred_3d: torch.Tensor,
    gt_rgb_bbox: torch.Tensor,
    gt_sonar_bbox: torch.Tensor,
    sonar_transform_matrix: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute reprojection errors for evaluation.

    Args:
        pred_3d: Predicted 3D center (B, 3)
        gt_rgb_bbox: Ground truth RGB bbox (B, 4)
        gt_sonar_bbox: Ground truth Sonar bbox (B, 4)
        sonar_transform_matrix: Sonar projection matrix (3, 4)

    Returns:
        Dictionary with reprojection errors
    """
    # Project to 2D
    proj_rgb_2d = project_3d_to_2d_rgb(pred_3d)
    proj_sonar_2d = project_3d_to_2d_sonar(pred_3d, sonar_transform_matrix)

    # GT centers
    gt_rgb_center = bbox_to_center(gt_rgb_bbox)
    gt_sonar_center = bbox_to_center(gt_sonar_bbox)

    # Euclidean distances
    error_rgb = torch.norm(proj_rgb_2d - gt_rgb_center, dim=1)  # (B,)
    error_sonar = torch.norm(proj_sonar_2d - gt_sonar_center, dim=1)  # (B,)

    return {
        'error_rgb': error_rgb,
        'error_sonar': error_sonar,
        'error_mean': (error_rgb + error_sonar) / 2.0,
    }


# Factory functions
def create_coordinate_transform(
    init_identity: bool = True,
    learnable: bool = True
) -> CoordinateTransform:
    """Create coordinate transformation."""
    return CoordinateTransform(init_identity=init_identity, learnable=learnable)


def create_cyclic_consistency_loss(
    loss_type: str = 'l2',
    weight_rgb: float = 1.0,
    weight_sonar: float = 1.0
) -> CyclicConsistencyLoss:
    """Create cyclic consistency loss."""
    return CyclicConsistencyLoss(
        loss_type=loss_type,
        weight_rgb=weight_rgb,
        weight_sonar=weight_sonar
    )
