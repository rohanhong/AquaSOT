"""
Neural Network Models for AquaSOT-3D

This package contains all neural network components:
- backbones: Unified ViT for RGB and Sonar feature extraction
- fusion: Cross-modal fusion modules (G2-DCMA)
- heads: Prediction heads for tracking and 3D estimation
- projector: 3D-2D cyclic projection modules
- quality: Quality estimation modules
- tracker: Complete end-to-end tracking system

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

from aquasot3d.models.backbones import (
    ViTBackbone,
    BackboneFactory,
    MultiModalBackbone,
    build_backbone,
    build_multimodal_backbone,
)

from aquasot3d.models.tracker import (
    AquaSOT3DTracker,
    build_tracker,
)


__all__ = [
    # Backbones
    'ViTBackbone',
    'BackboneFactory',
    'MultiModalBackbone',
    'build_backbone',
    'build_multimodal_backbone',
    # Tracker
    'AquaSOT3DTracker',
    'build_tracker',
]
