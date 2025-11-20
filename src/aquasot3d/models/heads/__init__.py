"""
Prediction Heads Module

This module contains all prediction heads for the tracking system:
- 2D Box Heads: RGB, Sonar, and Fusion box prediction
- Fusion Head: Multi-modal feature fusion with uncertainty weighting
- Weak 3D Head: Calibration-free 3D center estimation

Author: AquaSOT-3D Team
Date: 2025-11-10
"""

from .box_head import (
    BoxHead,
    RGBBoxHead,
    SonarBoxHead,
    FusionBoxHead,
    build_box_head,
    # Loss functions
    GIoULoss,
    CIoULoss,
    QualityFocalLoss,
    # Utility functions
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_xywh_to_cxcywh,
    compute_iou,
    compute_giou,
    compute_ciou,
)

from .fusion_head import (
    FusionHead,
    UncertaintyEstimator,
    AdaptiveFusionGate,
    build_fusion_head,
)

from .weak3d_head import (
    Weak3DHead,
    extract_radial_activation,
    build_weak3d_head,
)

# Weak3D Consistency Loss (Audit 2025112101 Phase 3.3)
from .weak3d_loss import (
    Weak3DConsistencyLoss,
    build_weak3d_consistency_loss,
)

__all__ = [
    # Box heads
    'BoxHead',
    'RGBBoxHead',
    'SonarBoxHead',
    'FusionBoxHead',
    'build_box_head',
    # Fusion head
    'FusionHead',
    'UncertaintyEstimator',
    'AdaptiveFusionGate',
    'build_fusion_head',
    # Weak 3D head
    'Weak3DHead',
    'extract_radial_activation',
    'build_weak3d_head',
    # Weak 3D consistency loss (Audit 2025112101 Phase 3.3)
    'Weak3DConsistencyLoss',
    'build_weak3d_consistency_loss',
    # Losses
    'GIoULoss',
    'CIoULoss',
    'QualityFocalLoss',
    # Utils
    'box_cxcywh_to_xyxy',
    'box_xyxy_to_cxcywh',
    'box_xywh_to_cxcywh',
    'compute_iou',
    'compute_giou',
    'compute_ciou',
]
