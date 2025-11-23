"""
Prediction Heads Module

This module contains all prediction heads for the tracking system:
- 2D Box Heads: RGB, Sonar, and Fusion box prediction
- Fusion Head: Multi-modal feature fusion with uncertainty weighting

Author: AquaSOT-3D Team
Date: 2025-11-23
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
