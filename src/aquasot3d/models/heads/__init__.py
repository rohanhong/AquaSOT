"""
Prediction Heads Module

This module contains OSTrack-style CENTER prediction heads for visual object tracking.
The CENTER head uses spatial convolutions to preserve location information.

Author: AquaSOT-3D Team
"""

from .box_head import (
    CenterHead,
    RGBCenterHead,
    SonarCenterHead,
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
    'CenterHead',
    'RGBCenterHead',
    'SonarCenterHead',
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
