"""
Prediction Heads Module

This module contains all prediction heads for the tracking system:
- 2D Box Heads: RGB and Sonar box prediction

ARCHITECTURE UPDATE (Audit 2025112313): Fusion heads removed (dual-view fusion architecture).

Author: AquaSOT-3D Team
Date: 2025-11-23
"""

from .box_head import (
    BoxHead,
    RGBBoxHead,
    SonarBoxHead,
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

__all__ = [
    # Box heads
    'BoxHead',
    'RGBBoxHead',
    'SonarBoxHead',
    'build_box_head',
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
