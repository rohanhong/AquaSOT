"""
Utility Modules

Geometry utilities for coordinate transformations, 3D-2D projections,
cyclic consistency loss, and configuration management.

Author: AquaSOT-3D Team
Date: 2025-11-10
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

from .config import (
    ConfigDict,
    load_config,
    save_config,
    validate_config,
    print_config,
    setup_config_logging,
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

    # Configuration
    'ConfigDict',
    'load_config',
    'save_config',
    'validate_config',
    'print_config',
    'setup_config_logging',
]
