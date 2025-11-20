"""
Backbone Networks for AquaSOT-3D

This package provides ViT backbone for unified RGB and Sonar feature extraction:
- ViT: Unified Vision Transformer for RGB (3-channel) and Sonar (1-channel) images
- Factory: Unified interface for creating backbones

Author: AquaSOT-3D Team
Date: 2025-11-19
"""

from aquasot3d.models.backbones.vit import (
    ViTBackbone,
    create_vit_backbone,
)

from aquasot3d.models.backbones.factory import (
    BackboneFactory,
    MultiModalBackbone,
    build_backbone,
    build_multimodal_backbone,
)


__all__ = [
    # ViT (unified for RGB and Sonar)
    'ViTBackbone',
    'create_vit_backbone',

    # Factory
    'BackboneFactory',
    'MultiModalBackbone',
    'build_backbone',
    'build_multimodal_backbone',
]
