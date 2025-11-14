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

from .bbox import (
    BBoxXYWH,
    BBoxCXCYWH,
    BBoxXYXY,
    bbox_from_xywh,
    bbox_from_cxcywh,
    bbox_from_xyxy,
    convert_bbox_format as convert_bbox_format_typed,
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

    # Type-safe Bbox Wrappers (Audit 2025111411 Finding 3)
    'BBoxXYWH',
    'BBoxCXCYWH',
    'BBoxXYXY',
    'bbox_from_xywh',
    'bbox_from_cxcywh',
    'bbox_from_xyxy',
    'convert_bbox_format_typed',

    # BEST PRACTICE: Use typed wrappers at module boundaries to prevent format confusion
    # Example:
    #   # In model forward (outputs normalized cxcywh):
    #   pred_bbox_cxcywh = BBoxCXCYWH(model_output['boxes'])
    #
    #   # In evaluation (expects xywh pixels):
    #   pred_bbox_xywh = pred_bbox_cxcywh.to_xywh()
    #   pred_pixels = denormalize_bbox(pred_bbox_xywh.tensor(), image_size)
    #   pred_bbox = BBoxXYWH(pred_pixels)  # Type-safe pixel coordinates

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
