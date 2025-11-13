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

from .bbox_utils import (
    convert_bbox_format,
    normalize_bbox,
    denormalize_bbox,
    scale_bbox,
    convert_boxes,
)

__all__ = [
    # Coordinate Transform
    'CoordinateTransform',
    'create_coordinate_transform',

    # Projection Functions
    'project_3d_to_2d_rgb',
    'project_3d_to_2d_sonar',

    # Bbox Utilities (legacy from geometry)
    'bbox_to_center',
    'center_to_bbox',

    # Bbox Format Conversion (Audit 2025111405 contract helpers)
    'convert_bbox_format',
    'normalize_bbox',
    'denormalize_bbox',
    'scale_bbox',
    'convert_boxes',

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
