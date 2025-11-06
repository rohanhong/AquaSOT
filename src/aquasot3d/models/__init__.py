"""
Neural Network Models for AquaSOT-3D

This package contains all neural network components:
- backbones: RGB and Sonar feature extractors
- fusion: Cross-modal fusion modules (G2-DCMA)
- temporal: Temporal modeling heads (Vision-Mamba, ARTrack)
- heads: Prediction heads for tracking and 3D estimation
- projector: 3D-2D cyclic projection modules
- quality: Quality estimation modules

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

from aquasot3d.models.backbones import (
    ViTRGBBackbone,
    ConvNeXtRGBBackbone,
    SonarEncoder,
    BackboneFactory,
    MultiModalBackbone,
    build_backbone,
    build_multimodal_backbone,
)


__all__ = [
    # Backbones
    'ViTRGBBackbone',
    'ConvNeXtRGBBackbone',
    'SonarEncoder',
    'BackboneFactory',
    'MultiModalBackbone',
    'build_backbone',
    'build_multimodal_backbone',
]
