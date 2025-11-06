"""
Backbone Networks for AquaSOT-3D

This package provides various backbone architectures for RGB and Sonar feature extraction:
- ViT RGB: Vision Transformer for RGB images
- ConvNeXt RGB: ConvNeXt architecture for RGB images
- Sonar Encoder: Specialized encoder for sonar images with log-polar preprocessing
- Backbone Factory: Unified interface for creating backbones

Author: AquaSOT-3D Team
Date: 2025-11-06
"""

from aquasot3d.models.backbones.vit_rgb import (
    ViTRGBBackbone,
    PositionEmbeddingAdapter,
    create_vit_rgb_backbone,
)

from aquasot3d.models.backbones.convnext_rgb import (
    ConvNeXtRGBBackbone,
    ConvNeXtFeaturePyramid,
    create_convnext_rgb_backbone,
)

from aquasot3d.models.backbones.sonar_encoder import (
    SonarEncoder,
    LightweightSonarCNN,
    RadialProfileExtractor,
    create_sonar_encoder,
)

from aquasot3d.models.backbones.backbone_factory import (
    BackboneFactory,
    MultiModalBackbone,
    build_backbone,
    build_multimodal_backbone,
)


__all__ = [
    # ViT RGB
    'ViTRGBBackbone',
    'PositionEmbeddingAdapter',
    'create_vit_rgb_backbone',

    # ConvNeXt RGB
    'ConvNeXtRGBBackbone',
    'ConvNeXtFeaturePyramid',
    'create_convnext_rgb_backbone',

    # Sonar Encoder
    'SonarEncoder',
    'LightweightSonarCNN',
    'RadialProfileExtractor',
    'create_sonar_encoder',

    # Factory
    'BackboneFactory',
    'MultiModalBackbone',
    'build_backbone',
    'build_multimodal_backbone',
]
