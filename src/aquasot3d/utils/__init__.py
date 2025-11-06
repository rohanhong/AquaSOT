"""
Utility Modules

Geometry utilities for coordinate transformations, 3D-2D projections,
and cyclic consistency loss.

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

from .geometry import (
    CoordinateTransform,
    project_3d_to_2d_rgb,
    project_3d_to_2d_sonar,
    bbox_to_center,
    center_to_bbox,
    CyclicConsistencyLoss,
    compute_reprojection_error,
    create_coordinate_transform,
    create_cyclic_consistency_loss,
)

__all__ = [
    # Coordinate Transform
    'CoordinateTransform',
    'create_coordinate_transform',

    # Projection Functions
    'project_3d_to_2d_rgb',
    'project_3d_to_2d_sonar',

    # Bbox Utilities
    'bbox_to_center',
    'center_to_bbox',

    # Cyclic Consistency Loss
    'CyclicConsistencyLoss',
    'create_cyclic_consistency_loss',
    'compute_reprojection_error',
]
